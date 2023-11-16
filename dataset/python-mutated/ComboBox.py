import sys
from collections import OrderedDict
from ..Qt import QtWidgets
__all__ = ['ComboBox']

class ComboBox(QtWidgets.QComboBox):
    """Extends QComboBox to add extra functionality.

      * Handles dict mappings -- user selects a text key, and the ComboBox indicates
        the selected value.
      * Requires item strings to be unique
      * Remembers selected value if list is cleared and subsequently repopulated
      * setItems() replaces the items in the ComboBox and blocks signals if the
        value ultimately does not change.
    """

    def __init__(self, parent=None, items=None, default=None):
        if False:
            while True:
                i = 10
        QtWidgets.QComboBox.__init__(self, parent)
        self.currentIndexChanged.connect(self.indexChanged)
        self._ignoreIndexChange = False
        if 'darwin' in sys.platform:
            self.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._chosenText = None
        self._items = OrderedDict()
        if items is not None:
            self.setItems(items)
            if default is not None:
                self.setValue(default)

    def setValue(self, value):
        if False:
            while True:
                i = 10
        'Set the selected item to the first one having the given value.'
        text = None
        for (k, v) in self._items.items():
            if v == value:
                text = k
                break
        if text is None:
            raise ValueError(value)
        self.setText(text)

    def setText(self, text):
        if False:
            i = 10
            return i + 15
        'Set the selected item to the first one having the given text.'
        ind = self.findText(text)
        if ind == -1:
            raise ValueError(text)
        self.setCurrentIndex(ind)

    def value(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If items were given as a list of strings, then return the currently \n        selected text. If items were given as a dict, then return the value\n        corresponding to the currently selected key. If the combo list is empty,\n        return None.\n        '
        if self.count() == 0:
            return None
        text = self.currentText()
        return self._items[text]

    def ignoreIndexChange(func):
        if False:
            for i in range(10):
                print('nop')

        def fn(self, *args, **kwds):
            if False:
                for i in range(10):
                    print('nop')
            prev = self._ignoreIndexChange
            self._ignoreIndexChange = True
            try:
                ret = func(self, *args, **kwds)
            finally:
                self._ignoreIndexChange = prev
            return ret
        return fn

    def blockIfUnchanged(func):
        if False:
            for i in range(10):
                print('nop')

        def fn(self, *args, **kwds):
            if False:
                i = 10
                return i + 15
            prevVal = self.value()
            blocked = self.signalsBlocked()
            self.blockSignals(True)
            try:
                ret = func(self, *args, **kwds)
            finally:
                self.blockSignals(blocked)
            if self.value() != prevVal:
                self.currentIndexChanged.emit(self.currentIndex())
            return ret
        return fn

    @ignoreIndexChange
    @blockIfUnchanged
    def setItems(self, items):
        if False:
            return 10
        '\n        *items* may be a list, a tuple, or a dict. \n        If a dict is given, then the keys are used to populate the combo box\n        and the values will be used for both value() and setValue().\n        '
        self.clear()
        self.addItems(items)

    def items(self):
        if False:
            return 10
        return self.items.copy()

    def updateList(self, items):
        if False:
            return 10
        return self.setItems(items)

    def indexChanged(self, index):
        if False:
            i = 10
            return i + 15
        if self._ignoreIndexChange:
            return
        self._chosenText = self.currentText()

    def setCurrentIndex(self, index):
        if False:
            return 10
        QtWidgets.QComboBox.setCurrentIndex(self, index)

    def itemsChanged(self):
        if False:
            for i in range(10):
                print('nop')
        if self._chosenText is not None:
            try:
                self.setText(self._chosenText)
            except ValueError:
                pass

    @ignoreIndexChange
    def insertItem(self, *args):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    @ignoreIndexChange
    def insertItems(self, *args):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @ignoreIndexChange
    def addItem(self, *args, **kwds):
        if False:
            print('Hello World!')
        try:
            if isinstance(args[0], str):
                text = args[0]
                if len(args) == 2:
                    value = args[1]
                else:
                    value = kwds.get('value', text)
            else:
                text = args[1]
                if len(args) == 3:
                    value = args[2]
                else:
                    value = kwds.get('value', text)
        except IndexError:
            raise TypeError('First or second argument of addItem must be a string.')
        if text in self._items:
            raise Exception('ComboBox already has item named "%s".' % text)
        self._items[text] = value
        QtWidgets.QComboBox.addItem(self, *args)
        self.itemsChanged()

    def setItemValue(self, name, value):
        if False:
            return 10
        if name not in self._items:
            self.addItem(name, value)
        else:
            self._items[name] = value

    @ignoreIndexChange
    @blockIfUnchanged
    def addItems(self, items):
        if False:
            while True:
                i = 10
        if isinstance(items, list) or isinstance(items, tuple):
            texts = items
            items = dict([(x, x) for x in items])
        elif isinstance(items, dict):
            texts = list(items.keys())
        else:
            raise TypeError('items argument must be list or dict or tuple (got %s).' % type(items))
        for t in texts:
            if t in self._items:
                raise Exception('ComboBox already has item named "%s".' % t)
        for (k, v) in items.items():
            self._items[k] = v
        QtWidgets.QComboBox.addItems(self, list(texts))
        self.itemsChanged()

    @ignoreIndexChange
    def clear(self):
        if False:
            while True:
                i = 10
        self._items = OrderedDict()
        QtWidgets.QComboBox.clear(self)
        self.itemsChanged()

    def saveState(self):
        if False:
            while True:
                i = 10
        ind = self.currentIndex()
        data = self.itemData(ind)
        if data is not None:
            try:
                if not data.isValid():
                    data = None
                else:
                    data = data.toInt()[0]
            except AttributeError:
                pass
        if data is None:
            return self.itemText(ind)
        else:
            return data

    def restoreState(self, v):
        if False:
            for i in range(10):
                print('nop')
        if type(v) is int:
            ind = self.findData(v)
            if ind > -1:
                self.setCurrentIndex(ind)
                return
        self.setCurrentIndex(self.findText(str(v)))

    def widgetGroupInterface(self):
        if False:
            print('Hello World!')
        return (self.currentIndexChanged, self.saveState, self.restoreState)