from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget

def setFont(widget: QWidget, fontSize=14, weight=QFont.Normal):
    if False:
        i = 10
        return i + 15
    ' set the font of widget\n\n    Parameters\n    ----------\n    widget: QWidget\n        the widget to set font\n\n    fontSize: int\n        font pixel size\n\n    weight: `QFont.Weight`\n        font weight\n    '
    widget.setFont(getFont(fontSize, weight))

def getFont(fontSize=14, weight=QFont.Normal):
    if False:
        return 10
    ' create font\n\n    Parameters\n    ----------\n    fontSize: int\n        font pixel size\n\n    weight: `QFont.Weight`\n        font weight\n    '
    font = QFont()
    font.setFamilies(['Segoe UI', 'Microsoft YaHei', 'PingFang SC'])
    font.setPixelSize(fontSize)
    font.setWeight(weight)
    return font