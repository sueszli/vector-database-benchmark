"""This is used to patch the QApplication style sheet.
It reads the current stylesheet, appends our modifications and sets the new stylesheet.
"""
import sys
from PyQt5 import QtWidgets
CUSTOM_PATCH_FOR_DARK_THEME = '\n/* PayToEdit text was being clipped */\nQAbstractScrollArea {\n    padding: 0px;\n}\n/* In History tab, labels while edited were being clipped (Windows) */\nQAbstractItemView QLineEdit {\n    padding: 0px;\n    show-decoration-selected: 1;\n}\n/* Checked item in dropdowns have way too much height...\n   see #6281 and https://github.com/ColinDuquesnoy/QDarkStyleSheet/issues/200\n   */\nQComboBox::item:checked {\n    font-weight: bold;\n    max-height: 30px;\n}\n'
CUSTOM_PATCH_FOR_DEFAULT_THEME_MACOS = '\n/* On macOS, main window status bar icons have ugly frame (see #6300) */\nStatusBarButton {\n    background-color: transparent;\n    border: 1px solid transparent;\n    border-radius: 4px;\n    margin: 0px;\n    padding: 2px;\n}\nStatusBarButton:checked {\n  background-color: transparent;\n  border: 1px solid #1464A0;\n}\nStatusBarButton:checked:disabled {\n  border: 1px solid #14506E;\n}\nStatusBarButton:pressed {\n  margin: 1px;\n  background-color: transparent;\n  border: 1px solid #1464A0;\n}\nStatusBarButton:disabled {\n  border: none;\n}\nStatusBarButton:hover {\n  border: 1px solid #148CD2;\n}\n'

def patch_qt_stylesheet(use_dark_theme: bool) -> None:
    if False:
        for i in range(10):
            print('nop')
    custom_patch = ''
    if use_dark_theme:
        custom_patch = CUSTOM_PATCH_FOR_DARK_THEME
    elif sys.platform == 'darwin':
        custom_patch = CUSTOM_PATCH_FOR_DEFAULT_THEME_MACOS
    app = QtWidgets.QApplication.instance()
    style_sheet = app.styleSheet() + custom_patch
    app.setStyleSheet(style_sheet)