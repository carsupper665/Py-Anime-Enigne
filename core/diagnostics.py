from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, Optional


def generate_diagnostic_id(prefix: str = "diag") -> str:
    """Return a short, human-readable diagnostic identifier."""
    token = uuid.uuid4().hex[:10]
    return f"{prefix}-{token}"


def _build_payload(diagnostic_id: str, event: str, extra: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"diagnostic_id": diagnostic_id, "event": event}
    for key, value in extra.items():
        if value is None:
            continue
        payload[key] = value
    return payload


def log_structured(
    logger: logging.Logger,
    level: int,
    diagnostic_id: Optional[str],
    event: str,
    **fields: Any,
) -> None:
    """Emit a structured JSON log if a diagnostic id is present."""
    if not diagnostic_id:
        return
    payload = _build_payload(diagnostic_id, event, fields)
    logger.log(level, json.dumps(payload, ensure_ascii=False))


def attach_diagnostic(exc: BaseException, diagnostic_id: Optional[str]) -> BaseException:
    """Annotate an exception with a diagnostic id for downstream handlers."""
    if diagnostic_id:
        setattr(exc, "diagnostic_id", diagnostic_id)
    return exc


def format_user_message(message: str, diagnostic_id: Optional[str]) -> str:
    """Append diagnostic id to user-facing message when available."""
    if not diagnostic_id:
        return message
    return f"{message}\n追蹤代碼：{diagnostic_id}"
