from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QToolButton,
    QSizePolicy,
    QSpacerItem,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QFont
class NavBar(QWidget):
    tabChanged = pyqtSignal(int)  # 自定義信號，發出選中的tab索引

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setObjectName("NavBar")
        self.setAutoFillBackground(True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)  # 關鍵
        self.setStyleSheet("""      
                #NavBar {
                    background: #212121;
                    border-right: 1px solid #CFCFCF;
                }
                QToolTip {
                    background-color: #2A3448;
                    color: #FFFFFF;
                    border: 1px solid #4B5870;
                    border-radius: 4px;
                    padding: 4px 6px;
                    margin: 0px;
                    font-family: "Inter";
                    font-size: 11px;
                    font-weight: normal;
                    white-space: nowrap;
                }
                """)

        self.setFixedWidth(55)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 12, 0, 12)
        layout.setSpacing(8)

        self.buttons = []

        self.dashboard_btn = self._make_button("", icon_path=r"./src/images/home_btn.png", page_index=0, tooltip="首頁")
        layout.addWidget(self.dashboard_btn)

        self.add = self._make_button("", icon_path=r"./src/images/icon_add.png", page_index=1, tooltip="Add Gif")
        layout.addWidget(self.add)

        self.set_gif = self._make_button("", icon_path=r"./src/images/view_btn.png", page_index=2, tooltip="Add gif to desktop")
        layout.addWidget(self.set_gif)

        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        # layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

    def _make_button(self, text: str, icon_path: str | None, page_index: int, tooltip: str = "") -> QToolButton:
        btn = QToolButton(self)
        btn.setText(text)
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon if text == "" else Qt.ToolButtonStyle.ToolButtonTextOnly)
        btn.setFixedHeight(55)
        btn.setFont(QFont("Inter", 10))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setCheckable(True)
        btn.page_index = page_index

        # Set tooltip if provided
        if tooltip:
            btn.setToolTip(tooltip)

        # optional icon
        if icon_path:
            icon = QIcon()
            icon.addFile(icon_path)
            btn.setIcon(icon)
            # set a reasonable icon size for text-beside-icon layout
            btn.setIconSize(QSize(20, 20))

        # base stylesheet; uses project's QTOOLBUTTON_HOVER for hover effect
        base = (
            "QToolButton {"
            "background-color: transparent;"
            "color: #E8EBF2;"
            "border: none;"
            "text-align: center;"
            # "padding-left: 12px;"
            "font-family: 'Inter';"
            "font-weight: 500;"
            "font-size: 16px;"
            "border-radius: 12px;"
            "}"
        )
        active = (
            "QToolButton:checked { background-color: rgba(255,255,255,18);"
            "border: 2px solid rgba(31, 156, 224, 255);"
            "font-weight: 700; color: #FFFFFF; }"
            # "QToolButton:checked {padding-left: 10px;}" if icon_path is not None else ""
        )
        c = """
QToolButton:hover {
        background-color:  rgba(255, 255, 255, 50); 
        border: 2px solid rgba(192, 192, 192, 255);
    }
"""
        border = """QToolButton{padding-left: 12px;}\nQToolButton:checked{padding: 10px;}\nQToolButton:hover {padding: 10px;}""" if icon_path is not None else ""

        btn.setStyleSheet(base + "\n" + c + "\n" + active + "\n" + border)

        btn.clicked.connect(self._on_nav_clicked)
        self.buttons.append(btn)
        return btn

    def _make_action_button(self, text: str, icon_path: str | None, tooltip: str = "") -> QToolButton:
        """Create an action button (like logout) that doesn't have checked state"""
        btn = QToolButton(self)
        btn.setText(text)
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon if text == "" else Qt.ToolButtonStyle.ToolButtonTextOnly)
        btn.setFixedHeight(56)
        btn.setFont(QFont("Inter", 10))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setCheckable(False)  # Action buttons are not checkable
        btn.page_index = -1

        # Set tooltip if provided
        if tooltip:
            btn.setToolTip(tooltip)

        # optional icon
        if icon_path:
            icon = QIcon()
            icon.addFile(icon_path)
            btn.setIcon(icon)
            # set a reasonable icon size for text-beside-icon layout
            btn.setIconSize(QSize(20, 20))

        # Different stylesheet for action buttons - no checked state, different hover color
        base = (
            "QToolButton {"
            "background-color: transparent;"
            "color: #E8EBF2;"
            "border: none;"
            "text-align: left;"
            "padding-left: 12px;"
            "font-family: 'Inter';"
            "font-weight: 500;"
            "font-size: 16px;"
            "border-radius: 12px;"
            "}"
        )
        hover = (
            "QToolButton:hover {"
            "background-color: rgba(229, 34, 34, 80);"  # Red-ish hover for logout
            "}"
        )
        btn.setStyleSheet(base + "\n" + hover)

        btn.clicked.connect(self._on_action_clicked)
        return btn

    def _on_nav_clicked(self):
        btn = self.sender()
        if not isinstance(btn, QToolButton):
            return
        # update active styles
        if btn.page_index >= 0:
            self.set_active(btn.page_index)
            self.tabChanged.emit(btn.page_index)
        else:
            # negative page index used for actions like logout
            self._handle_action(btn)

    def _on_action_clicked(self):
        """Handle action button clicks (like logout)"""
        btn = self.sender()
        if not isinstance(btn, QToolButton):
            return
        self._handle_action(btn)

    def _handle_action(self, btn: QToolButton):
        text = btn.text().lower()
        if hasattr(self.parent, "logout"):
            try:
                self.parent.LOGGER.warning("logout")
                self.parent.logout()
            except Exception as e:
                self.parent.LOGGER.error(f"Logout Fail. error: {e}")

    def set_active(self, page_index: int):
        for b in self.buttons:
            try:
                b.setChecked(b.page_index == page_index)
            except Exception as e:
                print(e)
                b.setChecked(False)

    def set_disabled(self, page_index: int = None, disabled: bool = True):
        for b in self.buttons:
            try:
                if b.page_index == page_index:
                    b.setDisabled(disabled)
                    return
                b.setDisabled(disabled)
            except Exception as e:
                print(e)
                b.setDisabled(False)
    
    def set_enabled(self, page_index: int = None, enabled: bool = True):
        self.set_disabled(page_index, not enabled)
