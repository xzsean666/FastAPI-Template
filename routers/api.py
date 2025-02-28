import asyncio

from fastapi import APIRouter, Request

from helper.limiter import limiter
from modules.api import *
from modules.base import *
from sdk.AI.SentenceSimilaritySummary import SentenceSimilarity

# from xzutils.AI.SentenceSimilarity import SentenceSimilarity
from sdk.AI.SentenceSimilarityTensorflow import (
    SentenceSimilarity as SentenceSimilarityTensorflow,
)

router = APIRouter()


sentenceSimilarity = SentenceSimilarity()
sentenceSimilarity.init()

sentenceSimilarityTensorflow = SentenceSimilarityTensorflow(
    model_path="/home/sean/AI/universal-sentence-encoder-v2"
)
sentenceSimilarityTensorflow.init()


@router.post(
    "/api/sentence_compare_torch", response_model=BaseResponse[SentenceCompareResponse]
)
@limiter.limit("5/second")
async def sentence_compare_torch(request: Request, data: SentenceCompareInput):
    response = SentenceCompareResponse(
        base_sentence=data.base_sentence, compare_results=[]
    )

    # 创建所有相似度计算的任务列表
    similarity_tasks = [
        sentenceSimilarity.get_similarity(data.base_sentence, compare_sentence)
        for compare_sentence in data.compare_sentences
    ]

    # 并行执行所有任务
    similarities = await asyncio.gather(*similarity_tasks)

    # 将结果组合
    response.compare_results = [
        CompareResult(sentence=sentence, result=similarity)
        for sentence, similarity in zip(data.compare_sentences, similarities)
    ]

    return BaseResponse(
        code=StatusCode.OK,
        message="success",
        data=response,
    )


@router.post(
    "/api/sentence_compare_tensorflow",
    response_model=BaseResponse[SentenceCompareResponse],
)
@limiter.limit("5/second")
async def sentence_compare_tensorflow(request: Request, data: SentenceCompareInput):
    response = SentenceCompareResponse(
        base_sentence=data.base_sentence, compare_results=[]
    )

    # 创建所有相似度计算的任务列表
    similarity_tasks = [
        sentenceSimilarityTensorflow.get_similarity(
            data.base_sentence, compare_sentence
        )
        for compare_sentence in data.compare_sentences
    ]

    # 并行执行所有任务
    similarities = await asyncio.gather(*similarity_tasks)

    # 将结果组合
    response.compare_results = [
        CompareResult(sentence=sentence, result=similarity)
        for sentence, similarity in zip(data.compare_sentences, similarities)
    ]

    return BaseResponse(
        code=StatusCode.OK,
        message="success",
        data=response,
    )
