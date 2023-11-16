from ...Qt import QtCore, QtGui, QtWidgets
from ...WidgetGroup import WidgetGroup
from . import axisCtrlTemplate_generic as ui_template
import weakref
translate = QtCore.QCoreApplication.translate

class ViewBoxMenu(QtWidgets.QMenu):

    def __init__(self, view):
        if False:
            i = 10
            return i + 15
        QtWidgets.QMenu.__init__(self)
        self.view = weakref.ref(view)
        self.valid = False
        self.viewMap = weakref.WeakValueDictionary()
        self.setTitle(translate('ViewBox', 'ViewBox options'))
        self.viewAll = QtGui.QAction(translate('ViewBox', 'View All'), self)
        self.viewAll.triggered.connect(self.autoRange)
        self.addAction(self.viewAll)
        self.ctrl = []
        self.widgetGroups = []
        self.dv = QtGui.QDoubleValidator(self)
        for axis in 'XY':
            m = self.addMenu(f"{axis} {translate('ViewBox', 'axis')}")
            w = QtWidgets.QWidget()
            ui = ui_template.Ui_Form()
            ui.setupUi(w)
            a = QtWidgets.QWidgetAction(self)
            a.setDefaultWidget(w)
            m.addAction(a)
            self.ctrl.append(ui)
            wg = WidgetGroup(w)
            self.widgetGroups.append(wg)
            connects = [(ui.mouseCheck.toggled, 'MouseToggled'), (ui.manualRadio.clicked, 'ManualClicked'), (ui.minText.editingFinished, 'RangeTextChanged'), (ui.maxText.editingFinished, 'RangeTextChanged'), (ui.autoRadio.clicked, 'AutoClicked'), (ui.autoPercentSpin.valueChanged, 'AutoSpinChanged'), (ui.linkCombo.currentIndexChanged, 'LinkComboChanged'), (ui.autoPanCheck.toggled, 'AutoPanToggled'), (ui.visibleOnlyCheck.toggled, 'VisibleOnlyToggled')]
            for (sig, fn) in connects:
                sig.connect(getattr(self, axis.lower() + fn))
        self.ctrl[0].invertCheck.toggled.connect(self.xInvertToggled)
        self.ctrl[1].invertCheck.toggled.connect(self.yInvertToggled)
        leftMenu = self.addMenu(translate('ViewBox', 'Mouse Mode'))
        group = QtGui.QActionGroup(self)
        group.triggered.connect(self.setMouseMode)
        pan = QtGui.QAction(translate('ViewBox', '3 button'), group)
        zoom = QtGui.QAction(translate('ViewBox', '1 button'), group)
        pan.setCheckable(True)
        zoom.setCheckable(True)
        leftMenu.addActions(group.actions())
        self.mouseModes = [pan, zoom]
        self.view().sigStateChanged.connect(self.viewStateChanged)
        self.updateState()

    def viewStateChanged(self):
        if False:
            for i in range(10):
                print('nop')
        self.valid = False
        if self.ctrl[0].minText.isVisible() or self.ctrl[1].minText.isVisible():
            self.updateState()

    def updateState(self):
        if False:
            i = 10
            return i + 15
        state = self.view().getState(copy=False)
        if state['mouseMode'] == ViewBox.PanMode:
            self.mouseModes[0].setChecked(True)
        else:
            self.mouseModes[1].setChecked(True)
        for i in [0, 1]:
            tr = state['targetRange'][i]
            self.ctrl[i].minText.setText('%0.5g' % tr[0])
            self.ctrl[i].maxText.setText('%0.5g' % tr[1])
            if state['autoRange'][i] is not False:
                self.ctrl[i].autoRadio.setChecked(True)
                if state['autoRange'][i] is not True:
                    self.ctrl[i].autoPercentSpin.setValue(int(state['autoRange'][i] * 100))
            else:
                self.ctrl[i].manualRadio.setChecked(True)
            self.ctrl[i].mouseCheck.setChecked(state['mouseEnabled'][i])
            c = self.ctrl[i].linkCombo
            c.blockSignals(True)
            try:
                view = state['linkedViews'][i]
                if view is None:
                    view = ''
                ind = c.findText(view)
                if ind == -1:
                    ind = 0
                c.setCurrentIndex(ind)
            finally:
                c.blockSignals(False)
            self.ctrl[i].autoPanCheck.setChecked(state['autoPan'][i])
            self.ctrl[i].visibleOnlyCheck.setChecked(state['autoVisibleOnly'][i])
            xy = ['x', 'y'][i]
            self.ctrl[i].invertCheck.setChecked(state.get(xy + 'Inverted', False))
        self.valid = True

    def popup(self, *args):
        if False:
            while True:
                i = 10
        if not self.valid:
            self.updateState()
        QtWidgets.QMenu.popup(self, *args)

    def autoRange(self):
        if False:
            return 10
        self.view().autoRange()

    def xMouseToggled(self, b):
        if False:
            i = 10
            return i + 15
        self.view().setMouseEnabled(x=b)

    def xManualClicked(self):
        if False:
            return 10
        self.view().enableAutoRange(ViewBox.XAxis, False)

    def xRangeTextChanged(self):
        if False:
            print('Hello World!')
        self.ctrl[0].manualRadio.setChecked(True)
        self.view().setXRange(*self._validateRangeText(0), padding=0)

    def xAutoClicked(self):
        if False:
            while True:
                i = 10
        val = self.ctrl[0].autoPercentSpin.value() * 0.01
        self.view().enableAutoRange(ViewBox.XAxis, val)

    def xAutoSpinChanged(self, val):
        if False:
            print('Hello World!')
        self.ctrl[0].autoRadio.setChecked(True)
        self.view().enableAutoRange(ViewBox.XAxis, val * 0.01)

    def xLinkComboChanged(self, ind):
        if False:
            while True:
                i = 10
        self.view().setXLink(str(self.ctrl[0].linkCombo.currentText()))

    def xAutoPanToggled(self, b):
        if False:
            return 10
        self.view().setAutoPan(x=b)

    def xVisibleOnlyToggled(self, b):
        if False:
            i = 10
            return i + 15
        self.view().setAutoVisible(x=b)

    def yMouseToggled(self, b):
        if False:
            for i in range(10):
                print('nop')
        self.view().setMouseEnabled(y=b)

    def yManualClicked(self):
        if False:
            return 10
        self.view().enableAutoRange(ViewBox.YAxis, False)

    def yRangeTextChanged(self):
        if False:
            for i in range(10):
                print('nop')
        self.ctrl[1].manualRadio.setChecked(True)
        self.view().setYRange(*self._validateRangeText(1), padding=0)

    def yAutoClicked(self):
        if False:
            print('Hello World!')
        val = self.ctrl[1].autoPercentSpin.value() * 0.01
        self.view().enableAutoRange(ViewBox.YAxis, val)

    def yAutoSpinChanged(self, val):
        if False:
            print('Hello World!')
        self.ctrl[1].autoRadio.setChecked(True)
        self.view().enableAutoRange(ViewBox.YAxis, val * 0.01)

    def yLinkComboChanged(self, ind):
        if False:
            for i in range(10):
                print('nop')
        self.view().setYLink(str(self.ctrl[1].linkCombo.currentText()))

    def yAutoPanToggled(self, b):
        if False:
            for i in range(10):
                print('nop')
        self.view().setAutoPan(y=b)

    def yVisibleOnlyToggled(self, b):
        if False:
            print('Hello World!')
        self.view().setAutoVisible(y=b)

    def yInvertToggled(self, b):
        if False:
            return 10
        self.view().invertY(b)

    def xInvertToggled(self, b):
        if False:
            return 10
        self.view().invertX(b)

    def setMouseMode(self, action):
        if False:
            for i in range(10):
                print('nop')
        mode = None
        if action == self.mouseModes[0]:
            mode = 'pan'
        elif action == self.mouseModes[1]:
            mode = 'rect'
        if mode is not None:
            self.view().setLeftButtonAction(mode)

    def setViewList(self, views):
        if False:
            print('Hello World!')
        names = ['']
        self.viewMap.clear()
        for v in views:
            name = v.name
            if name is None:
                continue
            names.append(name)
            self.viewMap[name] = v
        for i in [0, 1]:
            c = self.ctrl[i].linkCombo
            current = c.currentText()
            c.blockSignals(True)
            changed = True
            try:
                c.clear()
                for name in names:
                    c.addItem(name)
                    if name == current:
                        changed = False
                        c.setCurrentIndex(c.count() - 1)
            finally:
                c.blockSignals(False)
            if changed:
                c.setCurrentIndex(0)
                c.currentIndexChanged.emit(c.currentIndex())

    def _validateRangeText(self, axis):
        if False:
            i = 10
            return i + 15
        'Validate range text inputs. Return current value(s) if invalid.'
        inputs = (self.ctrl[axis].minText.text(), self.ctrl[axis].maxText.text())
        vals = self.view().viewRange()[axis]
        for (i, text) in enumerate(inputs):
            try:
                vals[i] = float(text)
            except ValueError:
                pass
        return vals
from .ViewBox import ViewBox