import json
import os
from pathlib import Path
from typing import Any, Dict


APP_NAME = "PyAnimeEngine"


def _config_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / APP_NAME
    return Path.home() / ".py_anime_engine"


def get_config_path() -> Path:
    return _config_dir() / "config.json"


def get_default_config() -> Dict[str, Any]:
    return {
        "engine": "hsv",  # rembg | openvino | hsv | wand
        "hsv": {
            "tol_h": 10,
            "tol_s": 60,
            "tol_v": 60,
            "strength": 1.5,
            "erode_iter": 1,
            "dilate_iter": 0,
            "feather_px": 2.0,
        },
        "openvino": {
            "model_path": "",
        },
        "output": {
            "image": "webp",   # png | webp
            "anim": "webp",    # webp | gif
            "dir": "./animes",
        },
        "export": {
            "request": {
                "image_format": "webp",
                "anim_format": "webp",
            },
            "options": {
                "quality": 75,
                "max_fps": 15,
                "loop": True,
            },
        },
        # 規格：影片輸出固定為 animated-webp（不含音訊/背景合成選項）
        "video": {
            "format": "animated-webp"
        },
        # OSD 版面（關閉時保存、下次載入時還原）
        # 每筆：{name, path, x, y, w, h, visible}
        "osd": [],
    }


def migrate_config(data: Dict[str, Any], default: Dict[str, Any]) -> Dict[str, Any]:
    def _merge(target: Dict[str, Any], src: Dict[str, Any]):
        for k, v in src.items():
            if k not in target:
                target[k] = v
            elif isinstance(v, dict) and isinstance(target.get(k), dict):
                _merge(target[k], v)

    _merge(data, default)

    # Legacy output options -> export.options
    export_options = data.setdefault("export", {}).setdefault("options", {})
    legacy_output = data.get("output", {}) if isinstance(data.get("output"), dict) else {}
    for key in ("quality", "max_fps", "loop"):
        if key in legacy_output:
            export_options[key] = legacy_output[key]
    # Keep legacy keys in sync for舊程式碼；將 export options 回寫到 output
    for key in ("quality", "max_fps", "loop"):
        if key in export_options:
            legacy_output[key] = export_options[key]
    data["output"] = legacy_output

    # 確保 request 預設存在
    export_request = data.setdefault("export", {}).setdefault("request", {})
    image_fmt = data.get("output", {}).get("image")
    anim_fmt = data.get("output", {}).get("anim")
    if image_fmt:
        export_request["image_format"] = image_fmt
    if anim_fmt:
        export_request["anim_format"] = anim_fmt

    # 規格相容性處理：影片輸出固定為 animated-webp
    try:
        if isinstance(data.get("video"), dict):
            data["video"]["format"] = "animated-webp"
        else:
            data["video"] = {"format": "animated-webp"}
    except Exception:
        data["video"] = {"format": "animated-webp"}

    return data


def load_config() -> Dict[str, Any]:
    path = get_config_path()
    default = get_default_config()
    try:
        if not path.exists():
            return default
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return migrate_config(data, default)
    except Exception:
        # 壞掉時回退預設
        return default


def save_config(cfg: Dict[str, Any]) -> Path:
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return path
