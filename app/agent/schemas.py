"""Structured output schema for the LangGraph agent final turn."""

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class AgentAnswerPayload(BaseModel):
    """
    Returned after ReAct tool loops; the model fills this in a dedicated structured-output step.
    Field descriptions guide behavior so a separate long system prompt is optional.
    """

    model_config = ConfigDict(populate_by_name=True)

    reply: str = Field(
        ...,
        description=(
            "User-facing answer in plain text only. "
            "Do not include URLs, markdown links, or numeric document ids in the prose—"
            "list sources only in source_document_ids."
        ),
    )
    source_document_ids: list[int] = Field(
        default_factory=list,
        validation_alias=AliasChoices("source_document_ids", "sourceDocumentIds"),
        serialization_alias="source_document_ids",
        description=(
            "Integer document_id values from semantic_search tool results you actually used for facts in 'reply'. "
            "Copy each number exactly as in the tool JSON (field document_id). "
            "One entry per distinct document; omit ids you did not rely on. "
            "Empty if you did not call semantic_search or no indexed sources applied."
        ),
    )

    @field_validator("source_document_ids", mode="before")
    @classmethod
    def _coerce_document_ids(cls, v):
        if v is None:
            return []
        if not isinstance(v, list):
            return []
        out: list[int] = []
        for x in v:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                continue
        return out
