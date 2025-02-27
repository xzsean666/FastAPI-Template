from fastapi import APIRouter, Request

from helper.limiter import limiter
from modules.api import *
from modules.base import *

router = APIRouter()


@router.post(
    "/api/sentence_compare", response_model=BaseResponse[SentenceCompareResponse]
)
@limiter.limit("5/second")
def sentence_compare(request: Request, data: SentenceCompareInput):

    response = SentenceCompareResponse(
        base_sentence=data.base_sentence, compare_results=[]
    )
    return BaseResponse(
        code=StatusCode.OK,
        message="success",
        data=response,
    )
