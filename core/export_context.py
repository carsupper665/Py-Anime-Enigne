from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple


class ExportContextError(ValueError):
    """Raised when export context cannot be constructed due to invalid input."""


@dataclass(frozen=True)
class ExportOptions:
    quality: int = 75
    max_fps: int = 15  # 0 代表自動
    loop: bool = True
    target_path: Optional[str] = None
    direct_copy: bool = False
    profile: str = "webp"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExportRequest:
    src_path: str
    prefer: str
    engine: str
    output_dir: str
    image_format: str
    anim_format: str
    range_ms: Optional[Tuple[Optional[int], Optional[int]]]
    options: ExportOptions
    hsv_config: Dict[str, Any]
    wand_config: Optional[Dict[str, Any]]


@dataclass(frozen=True)
class ExportContext:
    request: ExportRequest
    env: Dict[str, str]


def parse_time_range(opts: Optional[Dict[str, Any]]) -> Optional[Tuple[Optional[int], Optional[int]]]:
    if not isinstance(opts, dict):
        return None
    range_cfg = opts.get("range")
    if not isinstance(range_cfg, dict):
        return None
    rin = range_cfg.get("in_ms")
    rout = range_cfg.get("out_ms")
    if rin is None and rout is None:
        return None

    def _to_int(value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ExportContextError(f"Invalid time range value: {value}") from exc

    return _to_int(rin), _to_int(rout)


def _get_export_options(config: Dict[str, Any]) -> Dict[str, Any]:
    export_cfg = {}
    if isinstance(config.get("export"), dict):
        export_cfg = config["export"].get("options", {})
    if not isinstance(export_cfg, dict):
        export_cfg = {}
    # 回退舊版 output 設定
    legacy = {}
    if isinstance(config.get("output"), dict):
        legacy = {
            "quality": config["output"].get("quality"),
            "max_fps": config["output"].get("max_fps"),
            "loop": config["output"].get("loop"),
            "profile": config["output"].get("image"),
        }
    result = {
        "quality": export_cfg.get("quality", legacy.get("quality", 75)),
        "max_fps": export_cfg.get("max_fps", legacy.get("max_fps", 15)),
        "loop": export_cfg.get("loop", legacy.get("loop", True)),
        "profile": export_cfg.get("profile", legacy.get("profile", "webp")),
        "target_path": export_cfg.get("target_path"),
        "direct_copy": export_cfg.get("direct_copy", False),
    }
    return result


def merge_export_settings(config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> ExportOptions:
    base = _get_export_options(config)
    if overrides:
        base.update({k: overrides[k] for k in ("quality", "max_fps", "loop", "target_path", "direct_copy", "profile") if k in overrides})

    try:
        quality = int(base.get("quality", 75))
    except (TypeError, ValueError):
        quality = 75
    try:
        max_fps = int(base.get("max_fps", 15))
    except (TypeError, ValueError):
        max_fps = 15
    loop = bool(base.get("loop", True))
    target_path_value = base.get("target_path")
    target_path = str(target_path_value) if isinstance(target_path_value, str) and target_path_value else None
    direct_copy = bool(base.get("direct_copy", False))
    profile = str(base.get("profile", "webp") or "webp").lower()
    return ExportOptions(
        quality=quality,
        max_fps=max_fps,
        loop=loop,
        target_path=target_path,
        direct_copy=direct_copy,
        profile=profile,
    )


def apply_engine_overrides(engine: str, opts: Optional[Dict[str, Any]], config: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    hsv_cfg: Dict[str, Any] = {}
    base_hsv = config.get("hsv", {}) if isinstance(config.get("hsv"), dict) else {}
    hsv_cfg.update(base_hsv)
    wand_cfg: Optional[Dict[str, Any]] = None

    if engine == "hsv":
        if isinstance(opts, dict) and isinstance(opts.get("hsv"), dict):
            hsv_cfg = dict(opts["hsv"])
    elif engine == "wand":
        wand_cfg = dict(opts or {})

    return hsv_cfg, wand_cfg


def build_export_context(src: str, prefer: str, opts: Optional[Dict[str, Any]], config: Dict[str, Any]) -> ExportContext:
    if not src:
        raise ExportContextError("Source path is required.")
    prefer = (prefer or "auto") or "auto"
    engine = (prefer or config.get("engine", "hsv")).lower()
    output_cfg = config.get("output", {}) if isinstance(config.get("output"), dict) else {}
    output_dir = output_cfg.get("dir", "./animes") or "./animes"
    image_format = str(output_cfg.get("image", "webp")).lower()
    anim_format = str(output_cfg.get("anim", "webp")).lower()
    range_ms = parse_time_range(opts or {})

    export_overrides = (opts or {}).get("export") if isinstance(opts, dict) else None
    export_options = merge_export_settings(config, export_overrides if isinstance(export_overrides, dict) else None)

    hsv_cfg, wand_cfg = apply_engine_overrides(engine, opts if isinstance(opts, dict) else None, config)

    env_updates: Dict[str, str] = {}
    model_path = ""
    try:
        model_path = config.get("openvino", {}).get("model_path", "")  # type: ignore[attr-defined]
    except Exception:
        model_path = ""
    if model_path:
        env_updates["RMBG_MODEL_PATH"] = str(model_path).strip()

    request = ExportRequest(
        src_path=src,
        prefer=prefer,
        engine=engine,
        output_dir=output_dir,
        image_format=image_format,
        anim_format=anim_format,
        range_ms=range_ms,
        options=export_options,
        hsv_config=hsv_cfg,
        wand_config=wand_cfg,
    )
    return ExportContext(request=request, env=env_updates)
