from ui.services.rmbg_pipeline import clamp_fps


def test_clamp_fps_no_limit():
    assert clamp_fps(24, 0) == 24


def test_clamp_fps_with_limit_lower():
    assert clamp_fps(60, 30) == 30


def test_clamp_fps_with_limit_higher():
    assert clamp_fps(24, 120) == 24


def test_clamp_fps_invalid_source():
    assert clamp_fps(0, 30) == 1
    assert clamp_fps(-10, 0) == 1
