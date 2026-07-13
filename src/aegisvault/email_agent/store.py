"""Synthetic email dataset store for Stage 4.1."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from aegisvault.agent_runtime.exceptions import ToolExecutionError


@dataclass(frozen=True, slots=True)
class Contact:
    id: str
    name: str
    email: str
    role: str
    aliases: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EmailMessage:
    id: str
    from_email: str
    to: list[str]
    subject: str
    body: str
    timestamp: str
    labels: list[str]
    priority: str
    read: bool
    attachments: list[str]
    thread_id: str
    folder: str = "inbox"
    deleted: bool = False
    archived: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, folder: str) -> "EmailMessage":
        return cls(
            id=str(payload["id"]),
            from_email=str(payload["from"]),
            to=[str(item) for item in payload.get("to", [])],
            subject=str(payload["subject"]),
            body=str(payload["body"]),
            timestamp=str(payload["timestamp"]),
            labels=[str(item) for item in payload.get("labels", [])],
            priority=str(payload.get("priority", "normal")),
            read=bool(payload.get("read", False)),
            attachments=[str(item) for item in payload.get("attachments", [])],
            thread_id=str(payload["thread_id"]),
            folder=folder,
            deleted=bool(payload.get("deleted", False)),
            archived=bool(payload.get("archived", False)),
        )

    def to_public_dict(self, *, include_body: bool = False) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "from": self.from_email,
            "to": self.to,
            "subject": self.subject,
            "timestamp": self.timestamp,
            "labels": self.labels,
            "priority": self.priority,
            "read": self.read,
            "attachments": self.attachments,
            "thread_id": self.thread_id,
            "folder": self.folder,
            "deleted": self.deleted,
            "archived": self.archived,
        }
        if include_body:
            payload["body"] = self.body
        else:
            payload["snippet"] = _snippet(self.body)
        return payload


class EmailStore:
    """In-memory email store backed by synthetic JSONL files."""

    def __init__(self, dataset_path: str | Path = "datasets/email", *, persist_sent: bool = True) -> None:
        self.dataset_path = Path(dataset_path)
        self.persist_sent = persist_sent
        self.inbox = self._load_messages(self.dataset_path / "inbox" / "emails.jsonl", folder="inbox")
        self.sent = self._load_messages(self.dataset_path / "sent" / "sent_emails.jsonl", folder="sent")
        self.drafts = self._load_messages(self.dataset_path / "drafts" / "drafts.jsonl", folder="drafts")
        self.contacts = self._load_contacts(self.dataset_path / "contacts" / "contacts.json")

    @property
    def messages(self) -> list[EmailMessage]:
        return [*self.inbox, *self.sent, *self.drafts]

    def search(
        self,
        *,
        query: str = "",
        label: str | None = None,
        sender: str | None = None,
        unread_only: bool = False,
        folder: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        terms = _terms(query)
        results: list[EmailMessage] = []
        for message in self.messages:
            if message.deleted:
                continue
            if folder and message.folder != folder:
                continue
            if unread_only and message.read:
                continue
            if label and label.lower() not in {item.lower() for item in message.labels}:
                continue
            if sender and not self._sender_matches(message, sender):
                continue
            haystack = _message_haystack(message, self.contact_name_for(message.from_email))
            if terms and not all(term in haystack for term in terms):
                continue
            results.append(message)
        results.sort(key=lambda item: item.timestamp, reverse=True)
        return [message.to_public_dict() for message in results[: max(1, limit)]]

    def read_email(self, email_id: str) -> dict[str, Any]:
        message = self._find_message(email_id)
        message.read = True
        return message.to_public_dict(include_body=True)

    def summarize(self, *, email_id: str | None = None, thread_id: str | None = None, query: str = "") -> dict[str, Any]:
        if email_id:
            messages = [self._find_message(email_id)]
        elif thread_id:
            messages = [item for item in self.messages if item.thread_id == thread_id and not item.deleted]
        elif query:
            ids = {item["id"] for item in self.search(query=query, limit=6)}
            messages = [item for item in self.messages if item.id in ids]
        else:
            messages = [item for item in self.inbox if not item.deleted and not item.read][:6]
        if not messages:
            raise ToolExecutionError("no emails matched the summary request")
        messages.sort(key=lambda item: item.timestamp)
        summary = " ".join(_first_sentence(item.body) for item in messages)
        return {
            "message_count": len(messages),
            "thread_id": messages[0].thread_id if messages else None,
            "subjects": sorted({item.subject for item in messages}),
            "summary": summary[:900],
            "latest_timestamp": max(item.timestamp for item in messages),
        }

    def draft_email(
        self,
        *,
        to: str,
        subject: str,
        body: str,
        context_email_id: str | None = None,
    ) -> dict[str, Any]:
        recipients = self.resolve_recipients(to)
        draft = EmailMessage(
            id=f"draft_{uuid4().hex[:8]}",
            from_email="me@aegisvault.local",
            to=recipients,
            subject=subject,
            body=body,
            timestamp=_utc_now(),
            labels=["draft"],
            priority="normal",
            read=True,
            attachments=[],
            thread_id=self._context_thread(context_email_id),
            folder="drafts",
        )
        self.drafts.append(draft)
        return draft.to_public_dict(include_body=True)

    def send_email(self, *, to: str, subject: str, body: str, context_email_id: str | None = None) -> dict[str, Any]:
        recipients = self.resolve_recipients(to)
        message = EmailMessage(
            id=f"sent_{uuid4().hex[:8]}",
            from_email="me@aegisvault.local",
            to=recipients,
            subject=subject,
            body=body,
            timestamp=_utc_now(),
            labels=["sent", "simulated"],
            priority="normal",
            read=True,
            attachments=[],
            thread_id=self._context_thread(context_email_id),
            folder="sent",
        )
        self.sent.append(message)
        if self.persist_sent:
            self._append_sent(message)
        return message.to_public_dict(include_body=True)

    def delete_email(self, email_id: str) -> dict[str, Any]:
        message = self._find_message(email_id)
        message.deleted = True
        return {"id": message.id, "deleted": True, "subject": message.subject}

    def archive_email(self, email_id: str) -> dict[str, Any]:
        message = self._find_message(email_id)
        message.archived = True
        if "archived" not in message.labels:
            message.labels.append("archived")
        return {"id": message.id, "archived": True, "subject": message.subject}

    def list_unread(self, *, label: str | None = None, sender: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        return self.search(label=label, sender=sender, unread_only=True, folder="inbox", limit=limit)

    def list_threads(self, *, query: str = "", limit: int = 10) -> list[dict[str, Any]]:
        matched = self.search(query=query, limit=100) if query else [item.to_public_dict() for item in self.messages if not item.deleted]
        thread_ids = {str(item["thread_id"]) for item in matched}
        threads: list[dict[str, Any]] = []
        for thread_id in thread_ids:
            messages = [item for item in self.messages if item.thread_id == thread_id and not item.deleted]
            messages.sort(key=lambda item: item.timestamp)
            if not messages:
                continue
            threads.append(
                {
                    "thread_id": thread_id,
                    "subject": messages[-1].subject,
                    "message_count": len(messages),
                    "participants": sorted({item.from_email for item in messages} | {email for item in messages for email in item.to}),
                    "latest_timestamp": messages[-1].timestamp,
                    "unread_count": sum(1 for item in messages if not item.read),
                    "labels": sorted({label for item in messages for label in item.labels}),
                    "snippet": _snippet(messages[-1].body),
                }
            )
        threads.sort(key=lambda item: item["latest_timestamp"], reverse=True)
        return threads[: max(1, limit)]

    def resolve_recipients(self, recipient: str) -> list[str]:
        resolved: list[str] = []
        for part in [item.strip() for item in recipient.split(",") if item.strip()]:
            contact = self._find_contact(part)
            resolved.append(contact.email if contact else part)
        if not resolved:
            raise ToolExecutionError("at least one recipient is required")
        invalid = [item for item in resolved if "@" not in item]
        if invalid:
            raise ToolExecutionError(f"invalid contact or email address: {', '.join(invalid)}")
        return resolved

    def contact_name_for(self, email: str) -> str:
        for contact in self.contacts:
            if contact.email.lower() == email.lower():
                return contact.name
        return ""

    def _find_message(self, email_id: str) -> EmailMessage:
        for message in self.messages:
            if message.id == email_id and not message.deleted:
                return message
        raise ToolExecutionError(f"email {email_id!r} was not found")

    def _find_contact(self, value: str) -> Contact | None:
        lower = value.lower()
        for contact in self.contacts:
            names = {contact.id.lower(), contact.name.lower(), contact.email.lower(), *(alias.lower() for alias in contact.aliases)}
            if lower in names:
                return contact
        return None

    def _sender_matches(self, message: EmailMessage, sender: str) -> bool:
        contact = self._find_contact(sender)
        if contact is not None:
            return message.from_email.lower() == contact.email.lower()
        haystack = f"{message.from_email} {self.contact_name_for(message.from_email)}".lower()
        normalized_sender = sender.lower().replace("@", " ").replace(".", " ").replace("-", " ")
        if sender.lower() in haystack:
            return True
        tokens = [token for token in normalized_sender.split() if len(token) > 3 and token not in {"mail", "email", "example"}]
        return any(token in haystack for token in tokens)

    def _context_thread(self, context_email_id: str | None) -> str:
        if not context_email_id:
            return f"thread_{uuid4().hex[:8]}"
        return self._find_message(context_email_id).thread_id

    def _append_sent(self, message: EmailMessage) -> None:
        sent_path = self.dataset_path / "sent" / "sent_emails.jsonl"
        sent_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _message_to_file_dict(message)
        with sent_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _load_messages(path: Path, *, folder: str) -> list[EmailMessage]:
        if not path.exists():
            return []
        messages: list[EmailMessage] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    messages.append(EmailMessage.from_dict(payload, folder=folder))
                except Exception as exc:
                    raise ToolExecutionError(f"invalid email dataset row {path}:{line_no}: {exc}") from exc
        return messages

    @staticmethod
    def _load_contacts(path: Path) -> list[Contact]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return [Contact(**item) for item in payload.get("contacts", [])]


def _message_to_file_dict(message: EmailMessage) -> dict[str, Any]:
    payload = asdict(message)
    payload["from"] = payload.pop("from_email")
    payload.pop("folder", None)
    return payload


def _message_haystack(message: EmailMessage, contact_name: str) -> str:
    return " ".join([message.subject, message.body, message.from_email, contact_name, *message.labels]).lower()


def _terms(query: str) -> list[str]:
    return [part.lower() for part in query.replace('"', " ").split() if part.strip()]


def _snippet(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def _first_sentence(text: str) -> str:
    compact = " ".join(text.split())
    for separator in (". ", "! ", "? "):
        if separator in compact:
            return compact.split(separator, 1)[0] + separator.strip()
    return compact[:220]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
