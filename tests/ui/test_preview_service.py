CV_IMPORT = False
try:
    import pytest  # noqa
    import cv2
    CV_IMPORT = True
except Exception:  # pragma: no cover
    import pytest
    pytest.skip('cv2 未安裝，略過 PreviewService 測試', allow_module_level=True)

import numpy as np
from PyQt6.QtGui import QImage
from ui.services.preview_service import PreviewService


def test_preview_service_static_frame(tmp_path):
    path = tmp_path / 'frame.png'
    cv2.imwrite(str(path), np.zeros((10, 10, 3), np.uint8))

    service = PreviewService()
    service.set_static_source(str(path))

    frame = service.load_current_frame()
    assert isinstance(frame.image, QImage)
    assert frame.meta['frame'] == 0


def test_preview_service_hsv_overlay(tmp_path):
    path = tmp_path / 'frame.png'
    img = np.zeros((10, 10, 3), np.uint8)
    img[:] = (0, 255, 0)
    cv2.imwrite(str(path), img)

    service = PreviewService()
    service.set_static_source(str(path))
    frame = service.load_current_frame(apply_hsv=True)
    assert isinstance(frame.image, QImage)
    assert isinstance(frame.overlay, QImage)
