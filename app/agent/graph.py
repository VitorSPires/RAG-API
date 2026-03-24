import json
import logging
from typing import Optional, Tuple, Type, Union

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.agent.schemas import AgentAnswerPayload
from app.core.config import get_chat_model, get_optional_agent_system_prompt
from app.services.semantic_search_service import semantic_search_for_llm

logger = logging.getLogger(__name__)

# Prepended only to the final structured-output LLM call (not the whole ReAct loop).
_STRUCTURED_OUTPUT_GUIDE = (
    "Fill the schema from the conversation above. "
    "In source_document_ids, list every integer document_id from semantic_search tool JSON that you used for reply—"
    "same numbers as in the tool output, one id per document (deduplicate). "
    "Do not invent ids; only ids that appeared in a tool message. "
    "Leave source_document_ids empty if you did not use search results. "
    "Put the answer only in reply; do not paste URLs or ids in reply."
)

StructuredFormatArg = Union[Type[AgentAnswerPayload], Tuple[str, Type[AgentAnswerPayload]]]


def build_agent_graph(checkpointer):
    """
    Build a compiled LangGraph react agent with Postgres checkpointing.
    Retrieval uses the same in-process service as GET /search/llm.
    Final user-visible content uses structured output (AgentAnswerPayload).
    """

    @tool
    def semantic_search(query: str, limit: int = 5) -> str:
        """Search uploaded documents for text relevant to the question. Use for factual questions."""
        try:
            lim = max(1, min(int(limit), 20))
            data = semantic_search_for_llm(query=query, limit=lim)
            return json.dumps(data.get("sources", []), ensure_ascii=False)
        except Exception:
            logger.exception("semantic_search tool failed")
            return json.dumps({"error": "search_failed"})

    model = ChatOpenAI(model=get_chat_model(), temperature=0.2)

    extra_system: Optional[str] = get_optional_agent_system_prompt()
    response_format: StructuredFormatArg = (_STRUCTURED_OUTPUT_GUIDE, AgentAnswerPayload)

    kwargs = dict(
        model=model,
        tools=[semantic_search],
        checkpointer=checkpointer,
        response_format=response_format,
    )
    if extra_system:
        kwargs["prompt"] = SystemMessage(content=extra_system)

    return create_react_agent(**kwargs)
