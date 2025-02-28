import hashlib

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ..DB.db_sqlite import KVDB

cache_db = KVDB("sqlite_cache.db", "text_similarity_cache")


class SentenceSimilarity:
    def __init__(self, model_path="google/flan-t5-base", cache_expiration=None):
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self.cache_expiration = cache_expiration
        self.device = torch.device("cpu")

    def init(self):
        if self.model is None:
            print("Loading model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path, torch_dtype=torch.float32
            )
            print("Model loaded!")

    def _create_cache_key(self, prefix, text1, text2):
        """创建缓存键"""
        combined = f"{text1}||{text2}"
        return f"{prefix}:{hashlib.md5(combined.encode()).hexdigest()}"

    async def _get_core_meaning(self, text: str) -> str:
        """理解并提取文本的核心含义"""
        # 检查核心含义缓存
        cache_key = self._create_cache_key("core", text, "")
        cached = await cache_db.get(cache_key, expire=self.cache_expiration)
        if cached is not None:
            return cached

        # 构建提示词
        summarize_prompt = (
            "Understand and summarize the core meaning of this text in a brief, standardized way.\n"
            "Keep key information, numbers, and actions.\n"
            f"Text: {text}\n"
            "Core meaning:"
        )

        # 生成核心含义
        inputs = self.tokenizer(
            summarize_prompt, return_tensors="pt", max_length=512, truncation=True
        )

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=50,
                min_length=1,
                num_beams=2,
                temperature=0.3,
                early_stopping=True,
            )

        core_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 缓存核心含义
        await cache_db.put(cache_key, core_text)

        return core_text

    async def get_similarity(self, text1: str, text2: str) -> float:
        """获取两段文本的相似度"""
        # 检查相似度缓存
        cache_key = self._create_cache_key("sim", text1, text2)
        cached = await cache_db.get(cache_key, expire=self.cache_expiration)
        if cached is not None:
            return cached

        # 获取两段文本的核心含义
        core_text1 = await self._get_core_meaning(text1)
        core_text2 = await self._get_core_meaning(text2)
        print(core_text1, core_text2)
        # 比较核心含义
        compare_prompt = (
            "You are a precise text comparison expert. Your task is to analyze exact differences between two texts.\n"
            "CRITICAL: Never output 1.0 unless texts are 100% identical. Even slight differences must reduce the score.\n\n"
            "Scoring rules (STRICT ENFORCEMENT REQUIRED):\n"
            "1.0 = 100% identical (exact same words, numbers, and order)\n"
            "0.9-0.99 = Same meaning with minor word order changes only\n"
            "0.8-0.89 = Same core meaning but different numbers/years (e.g., 2024 vs 2025)\n"
            "0.6-0.79 = Similar meaning but significant detail changes\n"
            "0.4-0.59 = Major detail differences but same topic\n"
            "0.2-0.39 = Very different meanings or outcomes\n"
            "0.0-0.19 = Opposite meanings (win vs lose)\n\n"
            "Examples (MUST FOLLOW):\n"
            "'trump win 2024' vs 'trump win 2025' = 0.85\n"
            "'trump win 2024' vs 'trump lose 2024' = 0.15\n"
            "'trump win 2024' vs 'trump win election' = 0.75\n"
            "'trump win 2024' vs 'trump 2024 win' = 0.95\n\n"
            f"Text 1: {core_text1}\n"
            f"Text 2: {core_text2}\n\n"
            "Steps:\n"
            "1. List EXACT differences in words, numbers, and order\n"
            "2. Output ONLY a number between 0 and 1 (never output 1 unless 100% identical)\n"
            "Score:"
        )

        inputs = self.tokenizer(
            compare_prompt, return_tensors="pt", max_length=512, truncation=True
        )

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=50,
                min_length=1,
                num_beams=4,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.2,
                early_stopping=True,
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取相似度分数
        try:
            similarity = float(result.strip())
            similarity = max(0.0, min(1.0, similarity))
        except ValueError:
            similarity = 0.5

        # 缓存相似度结果
        await cache_db.put(cache_key, similarity)

        return similarity

    async def get_similarity_with_analysis(self, text1: str, text2: str) -> dict:
        """获取相似度和详细分析"""
        cache_key = self._create_cache_key("analysis", text1, text2)
        cached = await cache_db.get(cache_key, expire=self.cache_expiration)
        if cached is not None:
            return cached

        # 构建提示词
        prompt = (
            "Analyze these two texts and provide:\n"
            "1. A similarity score (0-1)\n"
            "2. Key similarities and differences\n"
            "3. Brief explanation\n\n"
            f"Text 1: {text1}\n"
            f"Text 2: {text2}"
        )

        # 生成分析
        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        )

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=200,
                min_length=50,
                num_beams=2,
                temperature=0.7,
                early_stopping=True,
            )

        analysis = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 尝试从分析中提取相似度分数
        try:
            # 假设分数在文本的某处以数字形式出现
            import re

            score_match = re.search(
                r"(?:score|similarity):\s*(\d*\.?\d+)", analysis.lower()
            )
            similarity = float(score_match.group(1)) if score_match else 0.5
            similarity = max(0.0, min(1.0, similarity))
        except:
            similarity = 0.5

        result = {
            "similarity": similarity,
            "analysis": analysis,
            "text1": text1,
            "text2": text2,
        }

        # 存入缓存
        await cache_db.put(cache_key, result)

        return result
