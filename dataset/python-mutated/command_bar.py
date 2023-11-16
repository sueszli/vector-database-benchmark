from typing import Iterable, List, Tuple, Union
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QRectF, QRect, QPoint, QEvent
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QHoverEvent, QPainterPath
from PyQt5.QtWidgets import QAction, QLayoutItem, QWidget, QFrame, QHBoxLayout, QApplication
from ...common.font import setFont
from ...common.icon import FluentIcon, Icon, Action
from ...common.style_sheet import isDarkTheme
from .menu import RoundMenu, MenuAnimationType
from .button import TransparentToggleToolButton
from .tool_tip import ToolTipFilter
from .flyout import FlyoutViewBase, Flyout

class CommandButton(TransparentToggleToolButton):
    """ Command button

    Constructors
    ------------
    * CommandButton(`parent`: QWidget = None)
    * CommandButton(`icon`: QIcon | str | FluentIconBase = None, `parent`: QWidget = None)
    """

    def _postInit(self):
        if False:
            i = 10
            return i + 15
        super()._postInit()
        self.setCheckable(False)
        self.setToolButtonStyle(Qt.ToolButtonIconOnly)
        setFont(self, 12)
        self._text = ''
        self._action = None
        self._isTight = False

    def setTight(self, isTight: bool):
        if False:
            print('Hello World!')
        self._isTight = isTight
        self.update()

    def isTight(self):
        if False:
            for i in range(10):
                print('nop')
        return self._isTight

    def sizeHint(self) -> QSize:
        if False:
            i = 10
            return i + 15
        if self.isIconOnly():
            return QSize(36, 34) if self.isTight() else QSize(48, 34)
        tw = self.fontMetrics().width(self.text())
        style = self.toolButtonStyle()
        if style == Qt.ToolButtonTextBesideIcon:
            return QSize(tw + 47, 34)
        if style == Qt.ToolButtonTextOnly:
            return QSize(tw + 32, 34)
        return QSize(tw + 32, 50)

    def isIconOnly(self):
        if False:
            print('Hello World!')
        if not self.text():
            return True
        return self.toolButtonStyle() in [Qt.ToolButtonIconOnly, Qt.ToolButtonFollowStyle]

    def _drawIcon(self, icon, painter, rect):
        if False:
            print('Hello World!')
        pass

    def text(self):
        if False:
            i = 10
            return i + 15
        return self._text

    def setText(self, text: str):
        if False:
            for i in range(10):
                print('nop')
        self._text = text
        self.update()

    def setAction(self, action: QAction):
        if False:
            while True:
                i = 10
        self._action = action
        self._onActionChanged()
        self.clicked.connect(action.trigger)
        action.toggled.connect(self.setChecked)
        action.changed.connect(self._onActionChanged)
        self.installEventFilter(CommandToolTipFilter(self, 700))

    def _onActionChanged(self):
        if False:
            return 10
        action = self.action()
        self.setIcon(action.icon())
        self.setText(action.text())
        self.setToolTip(action.toolTip())
        self.setEnabled(action.isEnabled())
        self.setCheckable(action.isCheckable())
        self.setChecked(action.isChecked())

    def action(self):
        if False:
            while True:
                i = 10
        return self._action

    def paintEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        super().paintEvent(e)
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        if not self.isChecked():
            painter.setPen(Qt.white if isDarkTheme() else Qt.black)
        else:
            painter.setPen(Qt.black if isDarkTheme() else Qt.white)
        if not self.isEnabled():
            painter.setOpacity(0.43)
        elif self.isPressed:
            painter.setOpacity(0.63)
        style = self.toolButtonStyle()
        (iw, ih) = (self.iconSize().width(), self.iconSize().height())
        if self.isIconOnly():
            y = (self.height() - ih) / 2
            x = (self.width() - iw) / 2
            super()._drawIcon(self._icon, painter, QRectF(x, y, iw, ih))
        elif style == Qt.ToolButtonTextOnly:
            painter.drawText(self.rect(), Qt.AlignCenter, self.text())
        elif style == Qt.ToolButtonTextBesideIcon:
            y = (self.height() - ih) / 2
            super()._drawIcon(self._icon, painter, QRectF(11, y, iw, ih))
            rect = QRectF(26, 0, self.width() - 26, self.height())
            painter.drawText(rect, Qt.AlignCenter, self.text())
        elif style == Qt.ToolButtonTextUnderIcon:
            x = (self.width() - iw) / 2
            super()._drawIcon(self._icon, painter, QRectF(x, 9, iw, ih))
            rect = QRectF(0, ih + 13, self.width(), self.height() - ih - 13)
            painter.drawText(rect, Qt.AlignHCenter | Qt.AlignTop, self.text())

class CommandToolTipFilter(ToolTipFilter):
    """ Command tool tip filter """

    def _canShowToolTip(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return super()._canShowToolTip() and self.parent().isIconOnly()

class MoreActionsButton(CommandButton):
    """ More action button """

    def _postInit(self):
        if False:
            while True:
                i = 10
        super()._postInit()
        self.setIcon(FluentIcon.MORE)

    def sizeHint(self):
        if False:
            for i in range(10):
                print('nop')
        return QSize(40, 34)

    def clearState(self):
        if False:
            i = 10
            return i + 15
        self.setAttribute(Qt.WA_UnderMouse, False)
        e = QHoverEvent(QEvent.HoverLeave, QPoint(-1, -1), QPoint())
        QApplication.sendEvent(self, e)

class CommandSeparator(QWidget):
    """ Command separator """

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self.setFixedSize(9, 34)

    def paintEvent(self, e):
        if False:
            return 10
        painter = QPainter(self)
        painter.setPen(QColor(255, 255, 255, 21) if isDarkTheme() else QColor(0, 0, 0, 15))
        painter.drawLine(5, 2, 5, self.height() - 2)

class CommandMenu(RoundMenu):
    """ Command menu """

    def __init__(self, parent=None):
        if False:
            return 10
        super().__init__('', parent)
        self.setItemHeight(32)
        self.view.setIconSize(QSize(16, 16))

class CommandBar(QFrame):
    """ Command bar """

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent=parent)
        self._widgets = []
        self._hiddenWidgets = []
        self._hiddenActions = []
        self._menuAnimation = MenuAnimationType.DROP_DOWN
        self._toolButtonStyle = Qt.ToolButtonIconOnly
        self._iconSize = QSize(16, 16)
        self._isButtonTight = False
        self._spacing = 4
        self.moreButton = MoreActionsButton(self)
        self.moreButton.clicked.connect(self._showMoreActionsMenu)
        self.moreButton.hide()
        setFont(self, 12)
        self.setAttribute(Qt.WA_TranslucentBackground)

    def setSpaing(self, spacing: int):
        if False:
            return 10
        if spacing == self._spacing:
            return
        self._spacing = spacing
        self.updateGeometry()

    def spacing(self):
        if False:
            return 10
        return self._spacing

    def addAction(self, action: QAction):
        if False:
            print('Hello World!')
        ' add action\n\n        Parameters\n        ----------\n        action: QAction\n            the action to add\n        '
        if action in self.actions():
            return
        button = self._createButton(action)
        self._insertWidgetToLayout(-1, button)
        super().addAction(action)
        return button

    def addActions(self, actions: Iterable[QAction]):
        if False:
            while True:
                i = 10
        for action in actions:
            self.addAction(action)

    def addHiddenAction(self, action: QAction):
        if False:
            print('Hello World!')
        ' add hidden action '
        if action in self.actions():
            return
        self._hiddenActions.append(action)
        self.updateGeometry()
        super().addAction(action)

    def addHiddenActions(self, actions: List[QAction]):
        if False:
            while True:
                i = 10
        ' add hidden action '
        for action in actions:
            self.addHiddenAction(action)

    def insertAction(self, before: QAction, action: QAction):
        if False:
            return 10
        if before not in self.actions():
            return
        index = self.actions().index(before)
        button = self._createButton(action)
        self._insertWidgetToLayout(index, button)
        super().insertAction(before, action)
        return button

    def addSeparator(self):
        if False:
            while True:
                i = 10
        self.insertSeparator(-1)

    def insertSeparator(self, index: int):
        if False:
            while True:
                i = 10
        self._insertWidgetToLayout(index, CommandSeparator(self))

    def addWidget(self, widget: QWidget):
        if False:
            while True:
                i = 10
        ' add widget to command bar '
        self._insertWidgetToLayout(-1, widget)

    def removeAction(self, action: QAction):
        if False:
            return 10
        if action not in self.actions():
            return
        for w in self.commandButtons:
            if w.action() is action:
                self._widgets.remove(w)
                w.hide()
                w.deleteLater()
                break
        self.updateGeometry()

    def removeWidget(self, widget: QWidget):
        if False:
            return 10
        if widget not in self._widgets:
            return
        self._widgets.remove(widget)
        self.updateGeometry()

    def removeHiddenAction(self, action: QAction):
        if False:
            return 10
        if action in self._hiddenActions:
            self._hiddenActions.remove(action)

    def setToolButtonStyle(self, style: Qt.ToolButtonStyle):
        if False:
            return 10
        ' set the style of tool button '
        if self.toolButtonStyle() == style:
            return
        self._toolButtonStyle = style
        for w in self.commandButtons:
            w.setToolButtonStyle(style)

    def toolButtonStyle(self):
        if False:
            return 10
        return self._toolButtonStyle

    def setButtonTight(self, isTight: bool):
        if False:
            for i in range(10):
                print('nop')
        if self.isButtonTight() == isTight:
            return
        self._isButtonTight = isTight
        for w in self.commandButtons:
            w.setTight(isTight)
        self.updateGeometry()

    def isButtonTight(self):
        if False:
            print('Hello World!')
        return self._isButtonTight

    def setIconSize(self, size: QSize):
        if False:
            for i in range(10):
                print('nop')
        if size == self._iconSize:
            return
        self._iconSize = size
        for w in self.commandButtons:
            w.setIconSize(size)

    def iconSize(self):
        if False:
            i = 10
            return i + 15
        return self._iconSize

    def resizeEvent(self, e):
        if False:
            i = 10
            return i + 15
        self.updateGeometry()

    def _createButton(self, action: QAction):
        if False:
            i = 10
            return i + 15
        ' create command button '
        button = CommandButton(self)
        button.setAction(action)
        button.setToolButtonStyle(self.toolButtonStyle())
        button.setTight(self.isButtonTight())
        button.setIconSize(self.iconSize())
        button.setFont(self.font())
        return button

    def _insertWidgetToLayout(self, index: int, widget: QWidget):
        if False:
            i = 10
            return i + 15
        ' add widget to layout '
        widget.setParent(self)
        widget.show()
        if index < 0:
            self._widgets.append(widget)
        else:
            self._widgets.insert(index, widget)
        self.setFixedHeight(max((w.height() for w in self._widgets)))
        self.updateGeometry()

    def minimumSizeHint(self) -> QSize:
        if False:
            i = 10
            return i + 15
        return self.moreButton.size()

    def updateGeometry(self):
        if False:
            for i in range(10):
                print('nop')
        self._hiddenWidgets.clear()
        self.moreButton.hide()
        visibles = self._visibleWidgets()
        x = self.contentsMargins().left()
        h = self.height()
        for widget in visibles:
            widget.show()
            widget.move(x, (h - widget.height()) // 2)
            x += widget.width() + self.spacing()
        if self._hiddenActions or len(visibles) < len(self._widgets):
            self.moreButton.show()
            self.moreButton.move(x, (h - self.moreButton.height()) // 2)
        for widget in self._widgets[len(visibles):]:
            widget.hide()
            self._hiddenWidgets.append(widget)

    def _visibleWidgets(self) -> List[QWidget]:
        if False:
            i = 10
            return i + 15
        ' return the visible widgets in layout '
        if self.suitableWidth() <= self.width():
            return self._widgets
        w = self.moreButton.width()
        for (index, widget) in enumerate(self._widgets):
            w += widget.width()
            if index > 0:
                w += self.spacing()
            if w > self.width():
                break
        return self._widgets[:index]

    def suitableWidth(self):
        if False:
            print('Hello World!')
        widths = [w.width() for w in self._widgets]
        if self._hiddenActions:
            widths.append(self.moreButton.width())
        return sum(widths) + self.spacing() * max(len(widths) - 1, 0)

    def resizeToSuitableWidth(self):
        if False:
            while True:
                i = 10
        self.setFixedWidth(self.suitableWidth())

    def setFont(self, font: QFont):
        if False:
            for i in range(10):
                print('nop')
        super().setFont(font)
        for button in self.commandButtons:
            button.setFont(font)

    @property
    def commandButtons(self):
        if False:
            return 10
        return [w for w in self._widgets if isinstance(w, CommandButton)]

    def setMenuDropDown(self, down: bool):
        if False:
            return 10
        ' set the animation direction of more actions menu '
        if down:
            self._menuAnimation = MenuAnimationType.DROP_DOWN
        else:
            self._menuAnimation = MenuAnimationType.PULL_UP

    def isMenuDropDown(self):
        if False:
            for i in range(10):
                print('nop')
        return self._menuAnimation == MenuAnimationType.DROP_DOWN

    def _showMoreActionsMenu(self):
        if False:
            while True:
                i = 10
        ' show more actions menu '
        self.moreButton.clearState()
        actions = self._hiddenActions.copy()
        for w in reversed(self._hiddenWidgets):
            if isinstance(w, CommandButton):
                actions.insert(0, w.action())
        menu = CommandMenu(self)
        menu.addActions(actions)
        x = -menu.width() + menu.layout().contentsMargins().right() + self.moreButton.width() + 18
        if self._menuAnimation == MenuAnimationType.DROP_DOWN:
            y = self.moreButton.height()
        else:
            y = -5
        pos = self.moreButton.mapToGlobal(QPoint(x, y))
        menu.exec(pos, aniType=self._menuAnimation)

class CommandViewMenu(CommandMenu):
    """ Command view menu """

    def __init__(self, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self.view.setObjectName('commandListWidget')

    def setDropDown(self, down: bool, long=False):
        if False:
            while True:
                i = 10
        self.view.setProperty('dropDown', down)
        self.view.setProperty('long', long)
        self.view.setStyle(QApplication.style())
        self.view.update()

class CommandViewBar(CommandBar):
    """ Command view bar """

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self.setMenuDropDown(True)

    def setMenuDropDown(self, down: bool):
        if False:
            print('Hello World!')
        ' set the animation direction of more actions menu '
        if down:
            self._menuAnimation = MenuAnimationType.FADE_IN_DROP_DOWN
        else:
            self._menuAnimation = MenuAnimationType.FADE_IN_PULL_UP

    def isMenuDropDown(self):
        if False:
            print('Hello World!')
        return self._menuAnimation == MenuAnimationType.FADE_IN_DROP_DOWN

    def _showMoreActionsMenu(self):
        if False:
            for i in range(10):
                print('nop')
        self.moreButton.clearState()
        actions = self._hiddenActions.copy()
        for w in reversed(self._hiddenWidgets):
            if isinstance(w, CommandButton):
                actions.insert(0, w.action())
        menu = CommandViewMenu(self)
        menu.addActions(actions)
        view = self.parent()
        view.setMenuVisible(True)
        menu.closedSignal.connect(lambda : view.setMenuVisible(False))
        menu.setDropDown(self.isMenuDropDown(), menu.view.width() > view.width() + 5)
        if menu.view.width() < view.width():
            menu.view.setFixedWidth(view.width())
            menu.adjustSize()
        x = -menu.width() + menu.layout().contentsMargins().right() + self.moreButton.width() + 18
        if self.isMenuDropDown():
            y = self.moreButton.height()
        else:
            y = -13
            menu.setShadowEffect(0, (0, 0), QColor(0, 0, 0, 0))
            menu.layout().setContentsMargins(12, 20, 12, 8)
        pos = self.moreButton.mapToGlobal(QPoint(x, y))
        menu.exec(pos, aniType=self._menuAnimation)

class CommandBarView(FlyoutViewBase):
    """ Command bar view """

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent=parent)
        self.bar = CommandViewBar(self)
        self.hBoxLayout = QHBoxLayout(self)
        self.hBoxLayout.setContentsMargins(6, 6, 6, 6)
        self.hBoxLayout.addWidget(self.bar)
        self.hBoxLayout.setSizeConstraint(QHBoxLayout.SetMinAndMaxSize)
        self.setButtonTight(True)
        self.setIconSize(QSize(14, 14))
        self._isMenuVisible = False

    def setMenuVisible(self, isVisible):
        if False:
            while True:
                i = 10
        self._isMenuVisible = isVisible
        self.update()

    def addWidget(self, widget: QWidget):
        if False:
            print('Hello World!')
        self.bar.addWidget(widget)

    def setSpaing(self, spacing: int):
        if False:
            print('Hello World!')
        self.bar.setSpaing(spacing)

    def spacing(self):
        if False:
            i = 10
            return i + 15
        return self.bar.spacing()

    def addAction(self, action: QAction):
        if False:
            i = 10
            return i + 15
        return self.bar.addAction(action)

    def addActions(self, actions: Iterable[QAction]):
        if False:
            while True:
                i = 10
        self.bar.addActions(actions)

    def addHiddenAction(self, action: QAction):
        if False:
            while True:
                i = 10
        self.bar.addHiddenAction(action)

    def addHiddenActions(self, actions: List[QAction]):
        if False:
            for i in range(10):
                print('nop')
        self.bar.addHiddenActions(actions)

    def insertAction(self, before: QAction, action: QAction):
        if False:
            print('Hello World!')
        return self.bar.insertAction(before, action)

    def addSeparator(self):
        if False:
            while True:
                i = 10
        self.bar.addSeparator()

    def insertSeparator(self, index: int):
        if False:
            return 10
        self.bar.insertSeparator(index)

    def removeAction(self, action: QAction):
        if False:
            print('Hello World!')
        self.bar.removeAction(action)

    def removeWidget(self, widget: QWidget):
        if False:
            return 10
        self.bar.removeWidget(widget)

    def removeHiddenAction(self, action: QAction):
        if False:
            while True:
                i = 10
        self.bar.removeAction(action)

    def setToolButtonStyle(self, style: Qt.ToolButtonStyle):
        if False:
            while True:
                i = 10
        self.bar.setToolButtonStyle(style)

    def toolButtonStyle(self):
        if False:
            i = 10
            return i + 15
        return self.bar.toolButtonStyle()

    def setButtonTight(self, isTight: bool):
        if False:
            while True:
                i = 10
        self.bar.setButtonTight(isTight)

    def isButtonTight(self):
        if False:
            print('Hello World!')
        return self.bar.isButtonTight()

    def setIconSize(self, size: QSize):
        if False:
            for i in range(10):
                print('nop')
        self.bar.setIconSize(size)

    def iconSize(self):
        if False:
            i = 10
            return i + 15
        return self.bar.iconSize()

    def setFont(self, font: QFont):
        if False:
            for i in range(10):
                print('nop')
        self.bar.setFont(font)

    def setMenuDropDown(self, down: bool):
        if False:
            for i in range(10):
                print('nop')
        self.bar.setMenuDropDown(down)

    def suitableWidth(self):
        if False:
            return 10
        m = self.contentsMargins()
        return m.left() + m.right() + self.bar.suitableWidth()

    def resizeToSuitableWidth(self):
        if False:
            i = 10
            return i + 15
        self.bar.resizeToSuitableWidth()
        self.setFixedWidth(self.suitableWidth())

    def actions(self):
        if False:
            i = 10
            return i + 15
        return self.bar.actions()

    def paintEvent(self, e):
        if False:
            i = 10
            return i + 15
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        path = QPainterPath()
        path.setFillRule(Qt.WindingFill)
        path.addRoundedRect(QRectF(self.rect().adjusted(1, 1, -1, -1)), 8, 8)
        if self._isMenuVisible:
            y = self.height() - 10 if self.bar.isMenuDropDown() else 1
            path.addRect(1, y, self.width() - 2, 9)
        painter.setBrush(QColor(40, 40, 40) if isDarkTheme() else QColor(248, 248, 248))
        painter.setPen(QColor(56, 56, 56) if isDarkTheme() else QColor(233, 233, 233))
        painter.drawPath(path.simplified())