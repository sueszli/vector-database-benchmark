from qt.core import QPoint, QRect, QSize, QSizePolicy, QStyle, QStyleOption, QStylePainter, Qt, QToolBar, QToolButton, QWidget, pyqtSignal

class Separator(QWidget):

    def __init__(self, icon_size, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self.desired_height = icon_size.height() * 0.85

    def style_option(self):
        if False:
            i = 10
            return i + 15
        opt = QStyleOption()
        opt.initFrom(self)
        opt.state |= QStyle.StateFlag.State_Horizontal
        return opt

    def sizeHint(self):
        if False:
            i = 10
            return i + 15
        width = self.style().pixelMetric(QStyle.PixelMetric.PM_ToolBarSeparatorExtent, self.style_option(), self)
        return QSize(width, int(self.devicePixelRatioF() * self.desired_height))

    def paintEvent(self, ev):
        if False:
            for i in range(10):
                print('nop')
        p = QStylePainter(self)
        p.drawPrimitive(QStyle.PrimitiveElement.PE_IndicatorToolBarSeparator, self.style_option())

class Button(QToolButton):
    layout_needed = pyqtSignal()

    def __init__(self, action, parent=None):
        if False:
            return 10
        super().__init__(parent)
        self.action = action
        self.setAutoRaise(True)
        action.changed.connect(self.update_state)
        self.update_state()
        self.clicked.connect(self.action.trigger)

    def update_state(self):
        if False:
            while True:
                i = 10
        ac = self.action
        self.setIcon(ac.icon())
        self.setToolTip(ac.toolTip() or self.action.text())
        self.setEnabled(ac.isEnabled())
        self.setCheckable(ac.isCheckable())
        self.setChecked(ac.isChecked())
        self.setMenu(ac.menu())
        old = self.isVisible()
        self.setVisible(ac.isVisible())
        if self.isVisible() != old:
            self.layout_needed.emit()

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'Button({self.toolTip()})'

class SingleLineToolBar(QToolBar):

    def __init__(self, parent=None, icon_size=18):
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self.setIconSize(QSize(icon_size, icon_size))

    def add_action(self, ac, popup_mode=QToolButton.ToolButtonPopupMode.DelayedPopup):
        if False:
            for i in range(10):
                print('nop')
        self.addAction(ac)
        w = self.widgetForAction(ac)
        w.setPopupMode(popup_mode)

    def add_separator(self):
        if False:
            while True:
                i = 10
        self.addSeparator()

class LayoutItem:

    def __init__(self, w):
        if False:
            while True:
                i = 10
        self.widget = w
        self.sz = sz = w.sizeHint()
        self.width = sz.width()
        self.height = sz.height()

class Group:

    def __init__(self, parent=None, leading_separator=None):
        if False:
            for i in range(10):
                print('nop')
        self.items = []
        self.width = self.height = 0
        self.parent = parent
        self.leading_separator = leading_separator

    def __bool__(self):
        if False:
            print('Hello World!')
        return bool(self.items)

    def smart_spacing(self, horizontal=True):
        if False:
            i = 10
            return i + 15
        p = self.parent
        if p is None:
            return -1
        if p.isWidgetType():
            which = QStyle.PixelMetric.PM_LayoutHorizontalSpacing if horizontal else QStyle.PixelMetric.PM_LayoutVerticalSpacing
            return p.style().pixelMetric(which, None, p)
        return p.spacing()

    def layout_spacing(self, wid, horizontal=True):
        if False:
            return 10
        ans = self.smart_spacing(horizontal)
        if ans != -1:
            return ans
        return wid.style().layoutSpacing(QSizePolicy.ControlType.ToolButton, QSizePolicy.ControlType.ToolButton, Qt.Orientation.Horizontal if horizontal else Qt.Orientation.Vertical)

    def add_widget(self, w):
        if False:
            while True:
                i = 10
        item = LayoutItem(w)
        self.items.append(item)
        (hs, vs) = (self.layout_spacing(w), self.layout_spacing(w, False))
        if self.items:
            self.width += hs
        self.width += item.width
        self.height = max(vs + item.height, self.height)

class FlowToolBar(QWidget):

    def __init__(self, parent=None, icon_size=18):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.icon_size = QSize(icon_size, icon_size)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.items = []
        self.button_map = {}
        self.applied_geometry = QRect(0, 0, 0, 0)

    def add_action(self, ac, popup_mode=QToolButton.ToolButtonPopupMode.DelayedPopup):
        if False:
            return 10
        w = Button(ac, self)
        w.setPopupMode(popup_mode)
        w.setIconSize(self.icon_size)
        self.button_map[ac] = w
        self.items.append(w)
        w.layout_needed.connect(self.updateGeometry)
        self.updateGeometry()

    def add_separator(self):
        if False:
            for i in range(10):
                print('nop')
        self.items.append(Separator(self.icon_size, self))
        self.updateGeometry()

    def hasHeightForWidth(self):
        if False:
            i = 10
            return i + 15
        return True

    def heightForWidth(self, width):
        if False:
            for i in range(10):
                print('nop')
        return self.do_layout(QRect(0, 0, width, 0), apply_geometry=False)

    def minimumSize(self):
        if False:
            print('Hello World!')
        size = QSize()
        for item in self.items:
            size = size.expandedTo(item.minimumSize())
        return size
    sizeHint = minimumSize

    def paintEvent(self, ev):
        if False:
            i = 10
            return i + 15
        if self.applied_geometry != self.rect():
            self.do_layout(self.rect(), apply_geometry=True)
        super().paintEvent(ev)

    def do_layout(self, rect, apply_geometry=False):
        if False:
            i = 10
            return i + 15
        (x, y) = (rect.x(), rect.y())
        line_height = 0

        def layout_spacing(wid, horizontal=True):
            if False:
                return 10
            ans = self.smart_spacing(horizontal)
            if ans != -1:
                return ans
            return wid.style().layoutSpacing(QSizePolicy.ControlType.ToolButton, QSizePolicy.ControlType.ToolButton, Qt.Orientation.Horizontal if horizontal else Qt.Orientation.Vertical)
        (lines, current_line) = ([], [])
        gmap = {}
        if apply_geometry:
            for item in self.items:
                if isinstance(item, Separator):
                    item.setGeometry(0, 0, 0, 0)

        def commit_line():
            if False:
                while True:
                    i = 10
            while current_line and isinstance(current_line[-1], Separator):
                current_line.pop()
            if current_line:
                lines.append((line_height, current_line))
        groups = []
        current_group = Group(self.parent())
        for wid in self.items:
            if not wid.isVisible() or (not current_group and isinstance(wid, Separator)):
                continue
            if isinstance(wid, Separator):
                groups.append(current_group)
                current_group = Group(self.parent(), wid)
            else:
                current_group.add_widget(wid)
        if current_group:
            groups.append(current_group)
        x = rect.x()
        y = 0
        line_height = 0
        vs = 0
        for group in groups:
            if current_line and x + group.width >= rect.right():
                commit_line()
                current_line = []
                x = rect.x()
                y += group.height
                group.leading_separator = None
                line_height = 0
            if group.leading_separator:
                current_line.append(group.leading_separator)
                sz = group.leading_separator.sizeHint()
                gmap[group.leading_separator] = (x, y, sz)
                x += sz.width() + group.layout_spacing(group.leading_separator)
            for item in group.items:
                wid = item.widget
                if not vs:
                    vs = group.layout_spacing(wid, False)
                if apply_geometry:
                    gmap[wid] = (x, y, item.sz)
                x += item.width + group.layout_spacing(wid)
                current_line.append(wid)
            line_height = group.height
        commit_line()
        if apply_geometry:
            self.applied_geometry = rect
            for (line_height, items) in lines:
                for wid in items:
                    (x, wy, isz) = gmap[wid]
                    if isz.height() < line_height:
                        wy += (line_height - isz.height()) // 2
                    if wid.isVisible():
                        wid.setGeometry(QRect(QPoint(x, wy), isz))
        return y + line_height - rect.y()

def create_flow_toolbar(parent=None, icon_size=18, restrict_to_single_line=False):
    if False:
        print('Hello World!')
    if restrict_to_single_line:
        return SingleLineToolBar(parent, icon_size)
    return FlowToolBar(parent, icon_size)