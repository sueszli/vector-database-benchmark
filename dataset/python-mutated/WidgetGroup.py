"""
WidgetGroup.py -  WidgetGroup class for easily managing lots of Qt widgets
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more information.

This class addresses the problem of having to save and restore the state
of a large group of widgets. 
"""
import inspect
import weakref
from .Qt import QtCore, QtWidgets
__all__ = ['WidgetGroup']

def splitterState(w):
    if False:
        i = 10
        return i + 15
    s = w.saveState().toPercentEncoding().data().decode()
    return s

def restoreSplitter(w, s):
    if False:
        for i in range(10):
            print('nop')
    if type(s) is list:
        w.setSizes(s)
    elif type(s) is str:
        w.restoreState(QtCore.QByteArray.fromPercentEncoding(s.encode()))
    else:
        print("Can't configure QSplitter using object of type", type(s))
    if w.count() > 0:
        for i in w.sizes():
            if i > 0:
                return
        w.setSizes([50] * w.count())

def comboState(w):
    if False:
        while True:
            i = 10
    ind = w.currentIndex()
    data = w.itemData(ind)
    if data is not None:
        try:
            if not data.isValid():
                data = None
            else:
                data = data.toInt()[0]
        except AttributeError:
            pass
    if data is None:
        return str(w.itemText(ind))
    else:
        return data

def setComboState(w, v):
    if False:
        while True:
            i = 10
    if type(v) is int:
        ind = w.findData(v)
        if ind > -1:
            w.setCurrentIndex(ind)
            return
    w.setCurrentIndex(w.findText(str(v)))

class WidgetGroup(QtCore.QObject):
    """State manager for groups of widgets.

    WidgetGroup handles common problems that arise when dealing with groups of widgets like a control
    panel:
    - Provide a single place for saving / restoring the state of all widgets in the group
    - Provide a single signal for detecting when any of the widgets have changed
    """
    classes = {QtWidgets.QSpinBox: (lambda w: w.valueChanged, QtWidgets.QSpinBox.value, QtWidgets.QSpinBox.setValue), QtWidgets.QDoubleSpinBox: (lambda w: w.valueChanged, QtWidgets.QDoubleSpinBox.value, QtWidgets.QDoubleSpinBox.setValue), QtWidgets.QSplitter: (None, splitterState, restoreSplitter, True), QtWidgets.QCheckBox: (lambda w: w.stateChanged, QtWidgets.QCheckBox.isChecked, QtWidgets.QCheckBox.setChecked), QtWidgets.QComboBox: (lambda w: w.currentIndexChanged, comboState, setComboState), QtWidgets.QGroupBox: (lambda w: w.toggled, QtWidgets.QGroupBox.isChecked, QtWidgets.QGroupBox.setChecked, True), QtWidgets.QLineEdit: (lambda w: w.editingFinished, lambda w: str(w.text()), QtWidgets.QLineEdit.setText), QtWidgets.QRadioButton: (lambda w: w.toggled, QtWidgets.QRadioButton.isChecked, QtWidgets.QRadioButton.setChecked), QtWidgets.QSlider: (lambda w: w.valueChanged, QtWidgets.QSlider.value, QtWidgets.QSlider.setValue)}
    sigChanged = QtCore.Signal(str, object)

    def __init__(self, widgetList=None):
        if False:
            print('Hello World!')
        "Initialize WidgetGroup, adding specified widgets into this group.\n        widgetList can be: \n         - a list of widget specifications (widget, [name], [scale])\n         - a dict of name: widget pairs\n         - any QObject, and all compatible child widgets will be added recursively.\n        \n        The 'scale' parameter for each widget allows QSpinBox to display a different value than the value recorded\n        in the group state (for example, the program may set a spin box value to 100e-6 and have it displayed as 100 to the user)\n        "
        QtCore.QObject.__init__(self)
        self.widgetList = weakref.WeakKeyDictionary()
        self.scales = weakref.WeakKeyDictionary()
        self.cache = {}
        self.uncachedWidgets = weakref.WeakKeyDictionary()
        if isinstance(widgetList, QtCore.QObject):
            self.autoAdd(widgetList)
        elif isinstance(widgetList, list):
            for w in widgetList:
                self.addWidget(*w)
        elif isinstance(widgetList, dict):
            for (name, w) in widgetList.items():
                self.addWidget(w, name)
        elif widgetList is None:
            return
        else:
            raise Exception('Wrong argument type %s' % type(widgetList))

    def addWidget(self, w, name=None, scale=None):
        if False:
            i = 10
            return i + 15
        if not self.acceptsType(w):
            raise Exception('Widget type %s not supported by WidgetGroup' % type(w))
        if name is None:
            name = str(w.objectName())
        if name == '':
            raise Exception("Cannot add widget '%s' without a name." % str(w))
        self.widgetList[w] = name
        self.scales[w] = scale
        self.readWidget(w)
        if type(w) in WidgetGroup.classes:
            signal = WidgetGroup.classes[type(w)][0]
        else:
            signal = w.widgetGroupInterface()[0]
        if signal is not None:
            if inspect.isfunction(signal) or inspect.ismethod(signal):
                signal = signal(w)
            signal.connect(self.widgetChanged)
        else:
            self.uncachedWidgets[w] = None

    def findWidget(self, name):
        if False:
            print('Hello World!')
        for w in self.widgetList:
            if self.widgetList[w] == name:
                return w
        return None

    def interface(self, obj):
        if False:
            for i in range(10):
                print('nop')
        t = type(obj)
        if t in WidgetGroup.classes:
            return WidgetGroup.classes[t]
        else:
            return obj.widgetGroupInterface()

    def checkForChildren(self, obj):
        if False:
            for i in range(10):
                print('nop')
        'Return true if we should automatically search the children of this object for more.'
        iface = self.interface(obj)
        return len(iface) > 3 and iface[3]

    def autoAdd(self, obj):
        if False:
            i = 10
            return i + 15
        accepted = self.acceptsType(obj)
        if accepted:
            self.addWidget(obj)
        if not accepted or self.checkForChildren(obj):
            for c in obj.children():
                self.autoAdd(c)

    def acceptsType(self, obj):
        if False:
            for i in range(10):
                print('nop')
        for c in WidgetGroup.classes:
            if isinstance(obj, c):
                return True
        if hasattr(obj, 'widgetGroupInterface'):
            return True
        return False

    def setScale(self, widget, scale):
        if False:
            while True:
                i = 10
        val = self.readWidget(widget)
        self.scales[widget] = scale
        self.setWidget(widget, val)

    def widgetChanged(self, *args):
        if False:
            for i in range(10):
                print('nop')
        w = self.sender()
        n = self.widgetList[w]
        v1 = self.cache[n]
        v2 = self.readWidget(w)
        if v1 != v2:
            self.sigChanged.emit(self.widgetList[w], v2)

    def state(self):
        if False:
            return 10
        for w in self.uncachedWidgets:
            self.readWidget(w)
        return self.cache.copy()

    def setState(self, s):
        if False:
            for i in range(10):
                print('nop')
        for w in self.widgetList:
            n = self.widgetList[w]
            if n not in s:
                continue
            self.setWidget(w, s[n])

    def readWidget(self, w):
        if False:
            for i in range(10):
                print('nop')
        if type(w) in WidgetGroup.classes:
            getFunc = WidgetGroup.classes[type(w)][1]
        else:
            getFunc = w.widgetGroupInterface()[1]
        if getFunc is None:
            return None
        if inspect.ismethod(getFunc) and getFunc.__self__ is not None:
            val = getFunc()
        else:
            val = getFunc(w)
        if self.scales[w] is not None:
            val /= self.scales[w]
        n = self.widgetList[w]
        self.cache[n] = val
        return val

    def setWidget(self, w, v):
        if False:
            return 10
        if self.scales[w] is not None:
            v *= self.scales[w]
        if type(w) in WidgetGroup.classes:
            setFunc = WidgetGroup.classes[type(w)][2]
        else:
            setFunc = w.widgetGroupInterface()[2]
        if inspect.ismethod(setFunc) and setFunc.__self__ is not None:
            setFunc(v)
        else:
            setFunc(w, v)