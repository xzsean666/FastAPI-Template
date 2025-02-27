from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class SentenceCompareInput(BaseModel, Generic[T]):
    """Base response model for API endpoints"""

    base_sentence: str


class CompareResult(BaseModel):
    """Model for a single comparison result"""

    sentence: str
    result: str


class SentenceCompareResponse(BaseModel):
    """Response model for sentence comparison"""

    base_sentence: str
    compare_results: List[CompareResult]
