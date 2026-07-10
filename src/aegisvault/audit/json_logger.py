"""JSON Lines audit sink."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from aegisvault.audit.base import AuditSink


class JsonLineAuditSink(AuditSink):
    """Append audit events to a UTF-8 JSON Lines file."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, event: dict[str, Any]) -> None:
        """Append one JSON object as a single line."""

        line = json.dumps(event, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self.output_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
