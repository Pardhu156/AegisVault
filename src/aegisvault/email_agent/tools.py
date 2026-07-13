"""Email-specific tools for the Stage 4.1 agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aegisvault.agent_runtime.tools import ToolDefinition, ToolRegistry
from aegisvault.email_agent.store import EmailStore

EMAIL_AGENT_SYSTEM_PROMPT = """You are a business email assistant operating only on a synthetic local email dataset.
Use the available email tools to search, read, summarize, draft, send, archive, delete, and list email threads.
When a task needs email facts, call tools instead of guessing.
For threaded conversations, prefer list_threads or summarize_email with a thread_id when appropriate.
Never claim that a simulated sent email was actually delivered outside the local dataset.
After tool results, answer concisely with the useful email details and mention any simulated action."""


def build_email_tool_registry(dataset_path: str | Path = "datasets/email", *, persist_sent: bool = True) -> ToolRegistry:
    """Build a tool registry backed by the synthetic email dataset."""

    store = EmailStore(dataset_path, persist_sent=persist_sent)
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="search_email",
            description="Search synthetic emails by query, label, sender, unread status, folder, and limit.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "label": {"type": "string"},
                    "sender": {"type": "string"},
                    "unread_only": {"type": "boolean"},
                    "folder": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": [],
            },
            function=lambda query="", label=None, sender=None, unread_only=False, folder=None, limit=10: store.search(
                query=str(query or ""),
                label=_optional(label),
                sender=_optional(sender),
                unread_only=_bool(unread_only),
                folder=_optional(folder),
                limit=_int(limit, default=10),
            ),
        )
    )
    registry.register(
        ToolDefinition(
            name="read_email",
            description="Read a full email by id and mark it as read in memory.",
            parameters={
                "type": "object",
                "properties": {"email_id": {"type": "string"}},
                "required": ["email_id"],
            },
            function=lambda email_id: store.read_email(str(email_id)),
        )
    )
    registry.register(
        ToolDefinition(
            name="summarize_email",
            description="Summarize one email, a thread, search results, or unread inbox messages.",
            parameters={
                "type": "object",
                "properties": {
                    "email_id": {"type": "string"},
                    "thread_id": {"type": "string"},
                    "query": {"type": "string"},
                },
                "required": [],
            },
            function=lambda email_id=None, thread_id=None, query="": store.summarize(
                email_id=_optional(email_id), thread_id=_optional(thread_id), query=str(query or "")
            ),
        )
    )
    registry.register(
        ToolDefinition(
            name="draft_email",
            description="Create a simulated draft email in the local dataset state.",
            parameters={
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                    "context_email_id": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
            function=lambda to, subject, body, context_email_id=None: store.draft_email(
                to=str(to), subject=str(subject), body=str(body), context_email_id=_optional(context_email_id)
            ),
        )
    )
    registry.register(
        ToolDefinition(
            name="send_email",
            description="Simulate sending an email by appending it to datasets/email/sent; no real email is sent.",
            parameters={
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                    "context_email_id": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
            function=lambda to, subject, body, context_email_id=None: store.send_email(
                to=str(to), subject=str(subject), body=str(body), context_email_id=_optional(context_email_id)
            ),
        )
    )
    registry.register(
        ToolDefinition(
            name="delete_email",
            description="Mark a synthetic email as deleted in memory.",
            parameters={
                "type": "object",
                "properties": {"email_id": {"type": "string"}},
                "required": ["email_id"],
            },
            function=lambda email_id: store.delete_email(str(email_id)),
        )
    )
    registry.register(
        ToolDefinition(
            name="archive_email",
            description="Mark a synthetic email as archived in memory.",
            parameters={
                "type": "object",
                "properties": {"email_id": {"type": "string"}},
                "required": ["email_id"],
            },
            function=lambda email_id: store.archive_email(str(email_id)),
        )
    )
    registry.register(
        ToolDefinition(
            name="list_unread",
            description="List unread inbox emails, optionally filtered by label or sender.",
            parameters={
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "sender": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": [],
            },
            function=lambda label=None, sender=None, limit=10: store.list_unread(
                label=_optional(label), sender=_optional(sender), limit=_int(limit, default=10)
            ),
        )
    )
    registry.register(
        ToolDefinition(
            name="list_threads",
            description="List conversation threads with participants, unread counts, labels, and latest snippets.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": [],
            },
            function=lambda query="", limit=10: store.list_threads(query=str(query or ""), limit=_int(limit, default=10)),
        )
    )
    return registry


def _optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default
