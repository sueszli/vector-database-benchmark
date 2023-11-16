from PyQt6 import QtCore, QtGui, QtWidgets
from picard.collection import load_user_collections, user_collections
from picard.util import strxfrm

class CollectionMenu(QtWidgets.QMenu):

    def __init__(self, albums, *args):
        if False:
            while True:
                i = 10
        super().__init__(*args)
        self.ids = set((a.id for a in albums))
        self._ignore_update = False
        self.update_collections()

    def update_collections(self):
        if False:
            i = 10
            return i + 15
        self._ignore_update = True
        self.clear()
        self.actions = []
        for (id_, collection) in sorted(user_collections.items(), key=lambda k_v: (strxfrm(str(k_v[1])), k_v[0])):
            action = QtWidgets.QWidgetAction(self)
            action.setDefaultWidget(CollectionMenuItem(self, collection))
            self.addAction(action)
            self.actions.append(action)
        self._ignore_update = False
        self.addSeparator()
        self.refresh_action = self.addAction(_('Refresh List'))
        self.hovered.connect(self.update_highlight)

    def refresh_list(self):
        if False:
            i = 10
            return i + 15
        self.refresh_action.setEnabled(False)
        load_user_collections(self.update_collections)

    def mouseReleaseEvent(self, event):
        if False:
            i = 10
            return i + 15
        if self.actionAt(event.pos()) == self.refresh_action and self.refresh_action.isEnabled():
            self.refresh_list()

    def update_highlight(self, action):
        if False:
            while True:
                i = 10
        if self._ignore_update:
            return
        for a in self.actions:
            a.defaultWidget().set_active(a == action)

    def update_active_action_for_widget(self, widget):
        if False:
            while True:
                i = 10
        if self._ignore_update:
            return
        for action in self.actions:
            action_widget = action.defaultWidget()
            is_active = action_widget == widget
            if is_active:
                self._ignore_hover = True
                self.setActiveAction(action)
                self._ignore_hover = False
            action_widget.set_active(is_active)

class CollectionMenuItem(QtWidgets.QWidget):

    def __init__(self, menu, collection):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.menu = menu
        self.active = False
        self._setup_layout(menu, collection)
        self._setup_colors()

    def _setup_layout(self, menu, collection):
        if False:
            return 10
        layout = QtWidgets.QVBoxLayout(self)
        style = self.style()
        layout.setContentsMargins(style.pixelMetric(QtWidgets.QStyle.PixelMetric.PM_LayoutLeftMargin), style.pixelMetric(QtWidgets.QStyle.PixelMetric.PM_FocusFrameVMargin), style.pixelMetric(QtWidgets.QStyle.PixelMetric.PM_LayoutRightMargin), style.pixelMetric(QtWidgets.QStyle.PixelMetric.PM_FocusFrameVMargin))
        self.checkbox = CollectionCheckBox(self, menu, collection)
        layout.addWidget(self.checkbox)

    def _setup_colors(self):
        if False:
            return 10
        palette = self.palette()
        self.text_color = palette.text().color()
        self.highlight_color = palette.highlightedText().color()

    def set_active(self, active):
        if False:
            return 10
        self.active = active
        palette = self.palette()
        textcolor = self.highlight_color if active else self.text_color
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, textcolor)
        self.checkbox.setPalette(palette)

    def enterEvent(self, e):
        if False:
            i = 10
            return i + 15
        self.menu.update_active_action_for_widget(self)

    def leaveEvent(self, e):
        if False:
            print('Hello World!')
        self.set_active(False)

    def paintEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        painter = QtWidgets.QStylePainter(self)
        option = QtWidgets.QStyleOptionMenuItem()
        option.initFrom(self)
        option.state = QtWidgets.QStyle.StateFlag.State_None
        if self.isEnabled():
            option.state |= QtWidgets.QStyle.StateFlag.State_Enabled
        if self.active:
            option.state |= QtWidgets.QStyle.StateFlag.State_Selected
        painter.drawControl(QtWidgets.QStyle.ControlElement.CE_MenuItem, option)

class CollectionCheckBox(QtWidgets.QCheckBox):

    def __init__(self, parent, menu, collection):
        if False:
            return 10
        self.menu = menu
        self.collection = collection
        super().__init__(self.label(), parent)
        releases = collection.releases & menu.ids
        if len(releases) == len(menu.ids):
            self.setCheckState(QtCore.Qt.CheckState.Checked)
        elif not releases:
            self.setCheckState(QtCore.Qt.CheckState.Unchecked)
        else:
            self.setCheckState(QtCore.Qt.CheckState.PartiallyChecked)

    def nextCheckState(self):
        if False:
            return 10
        ids = self.menu.ids
        if ids & self.collection.pending:
            return
        diff = ids - self.collection.releases
        if diff:
            self.collection.add_releases(diff, self.updateText)
            self.setCheckState(QtCore.Qt.CheckState.Checked)
        else:
            self.collection.remove_releases(ids & self.collection.releases, self.updateText)
            self.setCheckState(QtCore.Qt.CheckState.Unchecked)

    def updateText(self):
        if False:
            i = 10
            return i + 15
        self.setText(self.label())

    def label(self):
        if False:
            while True:
                i = 10
        c = self.collection
        return ngettext('%(name)s (%(count)i release)', '%(name)s (%(count)i releases)', c.size) % {'name': c.name, 'count': c.size}