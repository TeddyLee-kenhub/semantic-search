"""Validated API request and response models."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ContentType = Literal["article", "study_unit"]
Locale = Literal["zh", "en"]
RecordId = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._~-]{0,127}$",
    ),
]


class StrictModel(BaseModel):
    """Base schema rejecting fields not in the published API contract."""

    model_config = ConfigDict(extra="forbid")


class Chunk(StrictModel):
    """A single content chunk submitted for indexing."""

    text: str = Field(min_length=1, max_length=16000)

    @field_validator("text")
    @classmethod
    def text_must_not_be_whitespace(cls, value: str) -> str:
        """Reject chunks with no searchable content."""
        if not value.strip():
            raise ValueError("text must contain non-whitespace characters")
        return value


class ReplaceDocumentRequest(StrictModel):
    """Complete replacement payload for a document's chunk set."""

    locale: Locale
    slug: str = Field(max_length=512)
    chunks: list[Chunk] = Field(max_length=256)


class ReplaceDocumentResponse(StrictModel):
    """Summary of a completed document replacement."""

    type: ContentType
    record_id: str
    chunks_count: int = Field(
        ge=0, le=256, description="Actual indexed chunk count after splitting."
    )


class SearchRequest(StrictModel):
    """Vector search request, filtered by locale and content type."""

    q: str = Field(min_length=1, max_length=2048)
    locale: Locale
    types: list[ContentType] = Field(
        default_factory=lambda: ["article", "study_unit"],
        min_length=1,
        max_length=2,
    )
    limit: int = Field(default=10, ge=1, le=100)

    @field_validator("q")
    @classmethod
    def query_must_not_be_whitespace(cls, value: str) -> str:
        """Reject queries that contain no search terms."""
        if not value.strip():
            raise ValueError("q must contain non-whitespace characters")
        return value

    @field_validator("types")
    @classmethod
    def content_types_must_be_unique(
        cls, value: list[ContentType]
    ) -> list[ContentType]:
        """Reject duplicate content types."""
        if len(set(value)) != len(value):
            raise ValueError("types must not contain duplicates")
        return value


class SearchResult(StrictModel):
    """A distinct content result and its best chunk similarity."""

    type: ContentType
    record_id: str
    slug: str = Field(max_length=512)
    locale: Locale
    score: float = Field(ge=-1, le=1)


class SearchResponse(StrictModel):
    """Distinct matching content, ordered by score descending."""

    results: list[SearchResult] = Field(max_length=100)


class ErrorDetail(StrictModel):
    """Safe, machine-readable API error."""

    code: str
    message: str


class ErrorResponse(StrictModel):
    """Consistent error envelope."""

    error: ErrorDetail
