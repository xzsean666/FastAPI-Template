import hashlib

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from ..DB.db_sqlite import KVDB

cache_db = KVDB("sqlite_cache.db", "distilbert_sentence_similarity")


#
class SentenceSimilarity:
    def __init__(self, model_path="microsoft/e5-base-v2", cache_expiration=None):
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self.cache_expiration = cache_expiration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init(self):
        if self.model is None:
            print("Loading model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            self.model.to(self.device)
            print("Model loaded!")

    def _create_cache_key(self, prefix, text):
        """创建唯一的缓存键"""
        # 对于长文本，使用哈希值
        if len(text) > 100:
            text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
            return f"{prefix}:{text_hash}"
        return f"{prefix}:{text}"

    async def get_embedding(self, text: str) -> np.ndarray:
        """获取文本的嵌入向量，使用cache_db缓存"""
        cache_key = self._create_cache_key("embed", text)

        # 尝试从缓存获取
        cached = await cache_db.get(cache_key, expire=self.cache_expiration)
        if cached is not None:
            return np.array(cached)

        # 缓存未命中，计算嵌入向量
        if self.model is None:
            raise RuntimeError("Model not initialized")

        # 对输入文本进行编码
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 获取模型输出
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用[CLS]标记的输出作为句子表示
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

        # 存储到缓存
        await cache_db.put(cache_key, embedding.tolist())

        return embedding

    async def get_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度，使用cache_db缓存"""
        # 确保排序以保持一致性（text1,text2 和 text2,text1 应返回相同结果）
        sorted_texts = sorted([text1, text2])
        cache_key = self._create_cache_key(
            "sim", f"{sorted_texts[0]}||{sorted_texts[1]}"
        )

        # 尝试从缓存获取
        cached = await cache_db.get(cache_key, expire=self.cache_expiration)
        if cached is not None:
            return cached

        # 缓存未命中，计算相似度
        vec1 = await self.get_embedding(text1)
        vec2 = await self.get_embedding(text2)
        similarity = await self._cosine_similarity(vec1, vec2)

        # 将 numpy.float64 转换为 Python float 后再存储
        similarity_float = float(similarity)

        # 存储到缓存
        await cache_db.put(cache_key, similarity_float)

        return similarity_float

    async def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)
