"""Stage 4.1 Email Agent built on the generic agent runtime."""

from aegisvault.email_agent.store import Contact, EmailMessage, EmailStore
from aegisvault.email_agent.tools import EMAIL_AGENT_SYSTEM_PROMPT, build_email_tool_registry

__all__ = [
    "Contact",
    "EMAIL_AGENT_SYSTEM_PROMPT",
    "EmailMessage",
    "EmailStore",
    "build_email_tool_registry",
]
