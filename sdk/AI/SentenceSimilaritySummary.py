import hashlib

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

from ..DB.db_sqlite import KVDB

cache_db = KVDB("sqlite_cache.db", "distilbert_sentence_similarity")

model_list = [
    "distilbert-base-multilingual-cased",  # 混合
    "BAAI/bge-base-en-v1.5",  # 英语
    "BAAI/bge-base-zh-v1.5",
]

# # 1. 最小型号 (~250MB)
# model_path = "sentence-transformers/all-MiniLM-L6-v2"      # 250MB

# # 2. 小型号 (~500MB)
# model_path = "sentence-transformers/all-MiniLM-L12-v2"     # 420MB
# model_path = "BAAI/bge-small-en"                          # 440MB
# model_path = "BAAI/bge-small-en-v1.5"                     # 450MB
# model_path = "thenlper/gte-small"                         # 460MB
# # 3. 中等型号 (~800MB)
# model_path = "BAAI/bge-base-en"  # 780MB
# model_path = "BAAI/bge-base-en-v1.5"  # 790MB
# model_path = "thenlper/gte-base"  # 800MB
# model_path = "sentence-transformers/all-mpnet-base-v2"  # 850MB
# # 4. 大型号 (~1.3GB)
# model_path = "BAAI/bge-large-en"  # 1.2GB
# model_path = "BAAI/bge-large-en-v1.5"  # 1.3GB
# model_path = "thenlper/gte-large"  # 1.3GB
# # 1. 性能最好（1.3GB）
# "BAAI/bge-large-zh",  # 最佳性能，适合算力充足
# # 2. 平衡型（800MB）
# "BAAI/bge-base-zh",  # 性能与资源平衡，推荐选择
# # 3. 轻量级（500MB）
# "BAAI/bge-small-zh",  #


# # summary model
# # 1. BART 系列
# "facebook/bart-large-cnn"  # 1.6GB
# "facebook/bart-large-xsum"  # 1.6GB
# # 2. PEGASUS 系列
# "google/pegasus-cnn_dailymail"  # 2.2GB
# "google/pegasus-xsum"  # 2.2GB
# # 3. T5 系列
# "t5-base"  # 850MB
# # 4. 轻量级
# "sshleifer/distilbart-cnn-12-6"  # 400MB
# # 1. mT5 系列
# "google/mt5-base"                      # 1.2GB
# # ROUGE-1: 41.12 | ROUGE-2: 19.56 | ROUGE-L: 38.23
# # 优点：支持101种语言，性能最好
# # 缺点：速度较慢

# "google/mt5-small"                     # 500MB
# # ROUGE-1: 39.24 | ROUGE-2: 18.21 | ROUGE-L: 36.12
# # 优点：轻量级，速度快
# # 缺点：性能略低


#
class SentenceSimilarity:
    def __init__(
        self,
        similarity_model="BAAI/bge-base-en-v1.5",  # 相似度模型
        cache_expiration=None,
    ):
        self.model = None
        self.tokenizer = None
        self.model_path = similarity_model
        self.cache_expiration = cache_expiration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init(self):
        if self.model is None:
            print("Loading similarity model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            self.model.to(self.device)
            print("Similarity model loaded!")

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
        """计算文本相似度"""
        # 对排序后的文本创建缓存键
        sorted_texts = sorted([text1, text2])
        cache_key = self._create_cache_key(
            "sim", f"{sorted_texts[0]}||{sorted_texts[1]}"
        )

        # 尝试从缓存获取
        cached = await cache_db.get(cache_key, expire=self.cache_expiration)
        if cached is not None:
            return cached

        # 计算相似度
        vec1 = await self.get_embedding(text1)
        vec2 = await self.get_embedding(text2)
        similarity = await self._cosine_similarity(vec1, vec2)
        similarity = float(similarity)

        # 存入缓存
        await cache_db.put(cache_key, similarity)
        return similarity

    async def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)
