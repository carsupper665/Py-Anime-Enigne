import pytest

from core.config import get_default_config
from core.export_context import (
    ExportContextError,
    ExportOptions,
    build_export_context,
    merge_export_settings,
    parse_time_range,
)


def test_parse_time_range_valid_values():
    range_info = {"range": {"in_ms": "150", "out_ms": "450"}}
    assert parse_time_range(range_info) == (150, 450)


def test_parse_time_range_invalid_value_raises():
    with pytest.raises(ExportContextError):
        parse_time_range({"range": {"in_ms": "abc"}})


def test_merge_export_settings_applies_overrides():
    cfg = get_default_config()
    overrides = {
        "quality": 90,
        "max_fps": 0,
        "loop": False,
        "profile": "png",
        "target_path": "/tmp/out.png",
    }
    options = merge_export_settings(cfg, overrides)
    assert options == ExportOptions(
        quality=90,
        max_fps=0,
        loop=False,
        target_path="/tmp/out.png",
        direct_copy=False,
        profile="png",
    )


def test_build_export_context_includes_env_and_options():
    cfg = get_default_config()
    cfg["openvino"]["model_path"] = "/models/rmbg.onnx"
    context = build_export_context(
        "input.webp",
        "hsv",
        {"export": {"quality": 80}},
        cfg,
    )
    assert context.request.engine == "hsv"
    assert context.request.options.quality == 80
    assert context.request.options.profile == "webp"
    assert context.request.output_dir == cfg["output"]["dir"]
    assert context.env["RMBG_MODEL_PATH"] == "/models/rmbg.onnx"
