# ui/add_page.py
# from __future__ import annotations
import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
    QFileDialog, QListWidget, QListWidgetItem, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from ui.osd import OSD
from logging import Logger

BTN_STYLE = """
    QPushButton {
        background-color: #333;
        color: #EEE;
        border: none;
        border-radius: 6px;
        padding: 6px 12px;
        width: 80px;
        font-size: 14px;
        font-family: "Source Han Sans TC"
        }
    QPushButton:hover {
        background-color: #444;
        }
    QPushButton:pressed {
        background-color: #555;
        }
"""

class AddPage(QWidget):
    on_exception = pyqtSignal(object)
    on_activated = pyqtSignal(QWidget)
    dataChanged = pyqtSignal(dict)  # 對外通知更新完成
    gif_closed = pyqtSignal(str)

    def __init__(self, parent, logger: Logger, data: dict | None = None):
        super().__init__(parent)
        self.setObjectName("AddPage")
        self.logger = logger
        self.osd: OSD | None = None
        self.data = {"dir": "./animes", "items": []}  # 內部狀態

        self.activated_gifs:dict[str:QWidget] =  {}

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Add to Desktop", self)
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        title.setStyleSheet("""
            background-color: none;
            font-family: "Source Han Sans TC";
            font-weight: 600;
            font-size: 30px;
            color: #FFFFFF;
        """)

        # 清單：顯示提供的 data.items
        self.list = QListWidget(self)
        self.list.itemDoubleClicked.connect(self._show_selected)  # 雙擊顯示 OSD（QListWidget 提供 itemDoubleClicked）。:contentReference[oaicite:1]{index=1}
        self.list.setStyleSheet("""
            QListWidget { background:#111; border:1px solid #333; border-radius:8px; color:#ddd; }
            QListWidget::item { padding:6px 10px; }
            QListWidget::item:selected { background:#2a2a2a; }
        """)

        # 按鈕列
        row = QHBoxLayout()
        self.btn_show = QPushButton("顯示到桌面", self)
        self.btn_show.clicked.connect(self._show_selected)

        # self.btn_pick = QPushButton("選擇其他檔案", self)
        # self.btn_pick.clicked.connect(self._pick_and_add)  # 仍可手動選擇（QFileDialog）。:contentReference[oaicite:2]{index=2}
        
        self.btn_show.setStyleSheet(BTN_STYLE)
        # self.btn_pick.setStyleSheet(BTN_STYLE)

        row.addStretch(1)
        row.addWidget(self.btn_show)
        # row.addWidget(self.btn_pick)

        v.addWidget(title, alignment=Qt.AlignmentFlag.AlignHCenter)
        v.addWidget(self.list, 1)
        v.addLayout(row)

        # 初始化資料
        if data:
            self.update_data(data, mode="replace")

    # ---- 對外 API ----
    @pyqtSlot(dict)
    def update_data(self, payload: dict, mode: str = "replace"):
        """
        payload: {"dir": str, "items": [{"path": str, "width": int, "height": int, "is_anim": bool, "bytes": int}, ...]}
        mode: "replace" | "merge"
        """
        self.logger.debug(f"AddPage.update_data() called with mode={mode}, payload={payload}")
        try:
            if not isinstance(payload, dict) or "items" not in payload:
                return
            if mode == "replace":
                self.data = {"dir": payload.get("dir", self.data["dir"]), "items": list(payload["items"])}
                self._rebuild_list()
            else:  # merge
                seen = {it["path"] for it in self.data["items"]}
                for it in payload["items"]:
                    if it["path"] not in seen:
                        self.data["items"].append(it)
                        seen.add(it["path"])
                if "dir" in payload:
                    self.data["dir"] = payload["dir"]
                self._rebuild_list()

            self.dataChanged.emit(self.data)
        except Exception as e:
            self.on_exception.emit(e)

    @pyqtSlot(dict)
    def on_sync(self, gifs: dict[str:QWidget]):
        self.activated_gifs = gifs.copy()

    def get_data(self) -> dict:
        return dict(self.data)

    # ---- 內部 ----
    def _rebuild_list(self):
        self.list.clear()
        for it in self.data["items"]:
            name = os.path.basename(it["path"])
            anim_tag = "🎞" if it.get("is_anim") else "🖼"
            wh = f'{it.get("width","?")}×{it.get("height","?")}'
            size_k = f'{(it.get("bytes",0)/1024):.1f} KB'
            text = f'{anim_tag} {name}  ·  {wh}  ·  {size_k}'
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, it)
            self.list.addItem(item)

    def _current_path(self) -> str | None:
        cur = self.list.currentItem()
        # self.logger.debug(f"Current selected item: {cur}")
        return cur.data(Qt.ItemDataRole.UserRole) if cur else None

    @pyqtSlot()
    def _show_selected(self):
        try:
            it = self._current_path()
            if not it:
                return
            path = it['path']
            self.logger.debug(f"Show selected path: {path}")
            self.logger.debug(f"Item info: {it}")
            name=it['path'].split("./animes\\")[-1]
            if name in self.activated_gifs:
                btn = QMessageBox.question(
                            self, "目標已存在", f"「{name}」已存在，創建副本?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.No
                        )
                if btn != QMessageBox.StandardButton.Yes:
                    return

                name = self._duplicate(name)
            osd = OSD(name=name)
            osd.on_exception.connect(self.on_exception)
            osd.set_media(path, size=(it['width'], it['height']))
            osd.show()
            osd.activateWindow()
            osd.on_closed.connect(self.gif_closed)
            self.set_activated_gifs(osd)
        except Exception as e:
            self.on_exception.emit(e)

    def set_activated_gifs(self, gif: OSD):
        self.on_activated.emit(gif)

    def _duplicate(self, name: str, num: int = 0) -> str:
        if name in self.activated_gifs:
            return self._duplicate(f"{name}_c{num}", num + 1)
        return name

    def _pick_and_add(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇動畫", self.data.get("dir", ""), "Images (*.gif *.webp *.apng *.png *.jpg *.jpeg)"
        )
        if not path:
            return
        # 最少化：只新增 path，其餘欄位留空或由你後端填補
        if not any(it["path"] == path for it in self.data["items"]):
            self.data["items"].append({"path": path, "width": 0, "height": 0, "is_anim": True, "bytes": 0})
            self._rebuild_list()
            self.dataChanged.emit(self.data)
