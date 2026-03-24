import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import convert_to_messages
from pydantic import BaseModel, Field

from app.agent.schemas import AgentAnswerPayload

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")
    thread_id: str = Field(..., min_length=1, description="Stable id for the conversation (checkpoints)")


class SourceRef(BaseModel):
    file_name: str
    file_url: str


class ChatResponse(BaseModel):
    reply: str
    thread_id: str
    sources: List[SourceRef] = Field(
        default_factory=list,
        description="Documents the model cited via source_document_ids, resolved from semantic_search tool JSON.",
    )


def _extract_text_from_message_content(content: Union[str, List[dict], None]) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    return str(content)


def _last_ai_text(messages: List[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            text = _extract_text_from_message_content(m.content)
            if text:
                return text
    return ""


def _tool_message_text(content: Union[str, List[Any], None]) -> str:
    """Flatten tool message content; providers may use text blocks, json blocks, or plain str."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(str(block.get("text", "")))
            elif btype == "json" or "json" in block:
                j = block.get("json")
                if isinstance(j, str):
                    parts.append(j)
                else:
                    parts.append(json.dumps(j, ensure_ascii=False))
            elif isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content)


def _strip_json_fences(raw: str) -> str:
    s = raw.strip()
    if not s.startswith("```"):
        return s
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _tool_name_and_content(m: Any) -> tuple[Optional[str], Any]:
    """ToolMessage instance or OpenAI/LC dict after checkpoint load."""
    if isinstance(m, ToolMessage):
        return getattr(m, "name", None), m.content
    if isinstance(m, dict):
        role = str(m.get("type") or m.get("role") or "").lower()
        if role != "tool":
            return None, None
        return m.get("name"), m.get("content")
    return None, None


def _parse_doc_id(item: dict) -> Optional[int]:
    raw = item.get("document_id")
    if raw is None:
        raw = item.get("documentId")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _allowed_sources_by_document_id(new_messages: List[Any]) -> Dict[int, SourceRef]:
    """document_id -> SourceRef from semantic_search tool payloads in this invoke (first chunk wins per doc)."""
    mapping: Dict[int, SourceRef] = {}
    for m in new_messages:
        tname, content = _tool_name_and_content(m)
        if content is None and not isinstance(m, ToolMessage):
            continue
        if tname is not None and tname != "semantic_search":
            continue
        raw = _strip_json_fences(_tool_message_text(content))
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            doc_id = _parse_doc_id(item)
            if doc_id is None or doc_id in mapping:
                continue
            url = (item.get("file_url") or item.get("fileUrl") or "").strip()
            name = (item.get("file_name") or item.get("fileName") or "Document").strip() or "Document"
            mapping[doc_id] = SourceRef(file_name=name, file_url=url)
    return mapping


def _parse_structured_response(raw) -> Optional[AgentAnswerPayload]:
    """LangGraph may return a Pydantic model, dict, or provider-specific object."""
    if raw is None:
        return None
    if isinstance(raw, AgentAnswerPayload):
        return raw
    if isinstance(raw, dict):
        try:
            return AgentAnswerPayload.model_validate(raw)
        except Exception as e:
            logger.warning("AgentAnswerPayload model_validate(dict) failed: %s", e)
            return None
    if hasattr(raw, "model_dump"):
        try:
            return AgentAnswerPayload.model_validate(raw.model_dump())
        except Exception as e:
            logger.warning("AgentAnswerPayload model_validate(model_dump) failed: %s", e)
            return None
    try:
        return AgentAnswerPayload.model_validate(raw)
    except Exception as e:
        logger.warning("AgentAnswerPayload model_validate failed: %s; type=%s", e, type(raw))
        return None


def _sources_from_structured(
    structured: AgentAnswerPayload,
    allowed_by_doc_id: Dict[int, SourceRef],
) -> List[SourceRef]:
    out: List[SourceRef] = []
    seen: set[str] = set()
    for raw_id in structured.source_document_ids or []:
        try:
            did = int(raw_id)
        except (TypeError, ValueError):
            continue
        ref = allowed_by_doc_id.get(did)
        if ref is None:
            continue
        if ref.file_url in seen:
            continue
        seen.add(ref.file_url)
        out.append(ref)
    return out


def _strip_embedded_urls(reply: str, urls: List[str]) -> str:
    """Remove URLs the UI already shows as pills (model sometimes pastes them anyway)."""
    out = reply
    for u in sorted(set(urls), key=len, reverse=True):
        out = out.replace(u, "")
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\(\s*\)", "", out)
    return out.strip()


def _prior_message_count(graph, config: dict) -> int:
    try:
        snap = graph.get_state(config)
        if snap and snap.values:
            msgs = snap.values.get("messages") or []
            return len(msgs)
    except Exception:
        logger.debug("get_state before chat invoke failed", exc_info=True)
    return 0


def _messages_from_checkpoint(graph, config: dict) -> Optional[List[BaseMessage]]:
    try:
        snap = graph.get_state(config)
        if snap and snap.values:
            raw = snap.values.get("messages")
            if raw is not None:
                return list(raw)
    except Exception:
        logger.debug("get_state after chat invoke failed", exc_info=True)
    return None


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    graph = getattr(request.app.state, "agent_graph", None)
    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="Chat agent is unavailable (database or LangGraph checkpointer not initialized).",
        )

    config = {"configurable": {"thread_id": body.thread_id}}

    prev_len = _prior_message_count(graph, config)

    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=body.message)]},
            config=config,
        )
    except Exception as e:
        logger.exception("Agent invoke failed")
        raise HTTPException(status_code=500, detail="The assistant failed to complete the request.") from e

    messages_raw = _messages_from_checkpoint(graph, config) or list(result.get("messages") or [])
    try:
        messages = convert_to_messages(messages_raw)
    except Exception:
        logger.debug("convert_to_messages failed; using raw list", exc_info=True)
        messages = list(messages_raw)
    new_messages = messages[prev_len:] if len(messages) >= prev_len else messages
    # Use full history for tool payloads: prev_len slicing can miss ToolMessages vs checkpoint
    # ordering; ids are still validated (only tool-returned docs can appear in the map).
    allowed_by_doc_id = _allowed_sources_by_document_id(messages)

    raw_structured = result.get("structured_response")
    structured = _parse_structured_response(raw_structured)

    if structured is not None:
        sources = _sources_from_structured(structured, allowed_by_doc_id)
        reply = (structured.reply or "").strip()
        if sources:
            reply = _strip_embedded_urls(reply, [s.file_url for s in sources])
        logger.info(
            "chat turn: tool_docs=%s model_doc_ids=%s matched_sources=%s",
            len(allowed_by_doc_id),
            len(structured.source_document_ids or []),
            len(sources),
        )
    else:
        logger.warning(
            "structured_response missing or unparseable (type=%s); falling back to last AI message, no sources",
            type(raw_structured).__name__,
        )
        reply = _last_ai_text(messages)
        sources = []

    if not reply:
        raise HTTPException(status_code=500, detail="The assistant returned an empty response.")

    return ChatResponse(reply=reply, thread_id=body.thread_id, sources=sources)
