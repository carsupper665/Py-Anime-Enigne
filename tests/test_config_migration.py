from core.config import get_default_config, migrate_config


def test_migrate_config_moves_legacy_output_options():
    legacy = {
        "engine": "hsv",
        "output": {
            "dir": "./animes",
            "image": "webp",
            "anim": "webp",
            "quality": 60,
            "max_fps": 0,
            "loop": False,
        },
        "video": {"format": "gif"},
    }
    default = get_default_config()
    migrated = migrate_config(legacy, default)

    export_opts = migrated["export"]["options"]
    assert export_opts["quality"] == 60
    assert export_opts["max_fps"] == 0
    assert export_opts["loop"] is False
    # legacy output remains in sync
    assert migrated["output"]["quality"] == 60
    assert migrated["video"]["format"] == "animated-webp"


def test_migrate_config_fills_missing_request_defaults():
    data = {"output": {"image": "png", "anim": "webp"}}
    migrated = migrate_config(data, get_default_config())
    assert migrated["export"]["request"]["image_format"] == "png"
    assert migrated["export"]["request"]["anim_format"] == "webp"
