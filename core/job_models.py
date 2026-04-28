from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class JobRequest:
    src: str
    prefer: str
    opts: Dict[str, Any]
    diagnostic_id: Optional[str] = None

    def to_payload(self) -> tuple[str, str, Dict[str, Any]]:
        return self.src, self.prefer, self.opts

    @classmethod
    def from_payload(
        cls, src: str, prefer: str, opts: Dict[str, Any], diagnostic_id: Optional[str] = None
    ) -> "JobRequest":
        return cls(src=src, prefer=prefer, opts=opts, diagnostic_id=diagnostic_id)


@dataclass
class JobResult:
    input: str
    output: str
    kind: str
    frames: Optional[int] = None
    manifest: Optional[str] = None
    diagnostic_id: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "input": self.input,
            "output": self.output,
            "kind": self.kind,
        }
        if self.frames is not None:
            payload["frames"] = self.frames
        if self.manifest is not None:
            payload["manifest"] = self.manifest
        if self.diagnostic_id is not None:
            payload["diagnostic_id"] = self.diagnostic_id
        return payload

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "JobResult":
        return cls(
            input=str(payload.get("input", "")),
            output=str(payload.get("output", "")),
            kind=str(payload.get("kind", "")),
            frames=payload.get("frames"),
            manifest=payload.get("manifest"),
            diagnostic_id=payload.get("diagnostic_id"),
        )
