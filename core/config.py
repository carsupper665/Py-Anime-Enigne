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
        # 規格：影片輸出固定為 animated-webp（不含音訊/背景合成選項）
        "video": {
            "format": "animated-webp"
        },
    }


def load_config() -> Dict[str, Any]:
    path = get_config_path()
    default = get_default_config()
    try:
        if not path.exists():
            return default
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # merge defaults to keep forward compatibility
        def _merge(d, src):
            for k, v in src.items():
                if k not in d:
                    d[k] = v
                elif isinstance(v, dict) and isinstance(d.get(k), dict):
                    _merge(d[k], v)
        _merge(data, default)
        # 規格相容性處理：強制影片輸出格式為 animated-webp；忽略舊版可能殘留的可變設定
        try:
            if isinstance(data.get("video"), dict):
                data["video"]["format"] = "animated-webp"
            else:
                data["video"] = {"format": "animated-webp"}
        except Exception:
            data["video"] = {"format": "animated-webp"}
        return data
    except Exception:
        # 壞掉時回退預設
        return default


def save_config(cfg: Dict[str, Any]) -> Path:
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return path
