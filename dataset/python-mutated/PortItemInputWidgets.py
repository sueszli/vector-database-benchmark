from qtpy.QtGui import QFontMetrics, QFont
from qtpy.QtWidgets import QSpinBox, QLineEdit, QCheckBox, QComboBox
from .WidgetBaseClasses import NodeInputWidget

class DType_IW_Base(NodeInputWidget):

    def __init__(self, params):
        if False:
            while True:
                i = 10
        super().__init__(params)
        self.dtype = self.input.dtype
        self.block = False

    def change_val(self, val):
        if False:
            print('Hello World!')
        if not self.block:
            self.dtype.val = val
            self.update_node_input(val)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.__class__.__name__

class Data_IW(DType_IW_Base, QLineEdit):
    base_width = None

    def __init__(self, params):
        if False:
            return 10
        DType_IW_Base.__init__(self, params)
        QLineEdit.__init__(self)
        self.setFont(QFont('source code pro', 10))
        self.val_update_event(self.dtype.val)
        if self.dtype.size == 's':
            self.base_width = 30
        elif self.dtype.size == 'm':
            self.base_width = 70
        elif self.dtype.size == 'l':
            self.base_width = 150
        self.max_width = self.base_width * 3
        self.setFixedWidth(self.base_width)
        self.fm = QFontMetrics(self.font())
        self.setToolTip(self.dtype.doc)
        self.textChanged.connect(self.text_changed)
        self.editingFinished.connect(self.editing_finished)

    def text_changed(self, new_text):
        if False:
            print('Hello World!')
        'manages resizing of widget to content'
        text_width = self.fm.width(new_text)
        new_width = text_width + 15
        if new_width < self.max_width:
            self.setFixedWidth(new_width if new_width > self.base_width else self.base_width)
        else:
            self.setFixedWidth(self.max_width)
        self.node.update_shape()

    def editing_finished(self):
        if False:
            i = 10
            return i + 15
        'updates the input'
        self.change_val(self.get_val())

    def get_val(self):
        if False:
            return 10
        try:
            return eval(self.text())
        except Exception as e:
            return self.text()

    def val_update_event(self, val):
        if False:
            i = 10
            return i + 15
        'triggered when input is connected and received new data;\n        displays the data in the widget (without updating)'
        self.block = True
        try:
            self.setText(str(val))
        except Exception as e:
            pass
        finally:
            self.block = False

    def get_state(self) -> dict:
        if False:
            return 10
        return {'text': self.text()}

    def set_state(self, data: dict):
        if False:
            return 10
        self.val_update_event(data['text'])

class Data_IW_S(Data_IW):
    base_width = 30

class Data_IW_M(Data_IW):
    base_width = 70

class Data_IW_L(Data_IW):
    base_width = 150

class String_IW(DType_IW_Base, QLineEdit):
    width = None

    def __init__(self, params):
        if False:
            for i in range(10):
                print('nop')
        DType_IW_Base.__init__(self, params)
        QLineEdit.__init__(self)
        self.setFont(QFont('source code pro', 10))
        self.setText(self.dtype.val)
        self.setFixedWidth(self.width)
        self.setToolTip(self.dtype.doc)
        self.editingFinished.connect(self.editing_finished)

    def editing_finished(self):
        if False:
            i = 10
            return i + 15
        'updates the input'
        self.change_val(self.get_val())

    def get_val(self):
        if False:
            i = 10
            return i + 15
        return self.text()

    def val_update_event(self, val):
        if False:
            print('Hello World!')
        'triggered when input is connected and received new data;\n        displays the data in the widget (without updating)'
        self.block = True
        self.setText(str(val))
        self.block = False

    def get_state(self) -> dict:
        if False:
            while True:
                i = 10
        return {'text': self.text()}

    def set_state(self, data: dict):
        if False:
            while True:
                i = 10
        self.val_update_event(data['text'])

class String_IW_S(String_IW):
    width = 30

class String_IW_M(String_IW):
    width = 70

class String_IW_L(String_IW):
    width = 150

class Integer_IW(DType_IW_Base, QSpinBox):

    def __init__(self, params):
        if False:
            print('Hello World!')
        DType_IW_Base.__init__(self, params)
        QSpinBox.__init__(self)
        if self.dtype.bounds:
            self.setRange(self.dtype.bounds[0], self.dtype.bounds[1])
        self.setValue(self.dtype.val)
        self.setToolTip(self.dtype.doc)
        self.valueChanged.connect(self.widget_val_changed)

    def widget_val_changed(self, val):
        if False:
            i = 10
            return i + 15
        self.change_val(val)

    def get_val(self):
        if False:
            return 10
        return self.value()

    def val_update_event(self, val):
        if False:
            while True:
                i = 10
        'triggered when input is connected and received new data;\n        displays the data in the widget (without updating)'
        self.block = True
        try:
            self.setValue(val)
        except Exception as e:
            pass
        finally:
            self.block = False

    def get_state(self) -> dict:
        if False:
            i = 10
            return i + 15
        return {'val': self.value()}

    def set_state(self, data: dict):
        if False:
            print('Hello World!')
        self.val_update_event(data['val'])

class Float_IW(DType_IW_Base, QLineEdit):

    def __init__(self, params):
        if False:
            return 10
        DType_IW_Base.__init__(self, params)
        QLineEdit.__init__(self)
        self.setFont(QFont('source code pro', 10))
        fm = QFontMetrics(self.font())
        self.setMaximumWidth(fm.width(' ') * self.dtype.decimals + 1)
        self.setText(str(self.dtype.val))
        self.setToolTip(self.dtype.doc)
        self.textChanged.connect(self.widget_text_changed)

    def widget_text_changed(self):
        if False:
            print('Hello World!')
        self.change_val(self.get_val())

    def get_val(self):
        if False:
            for i in range(10):
                print('nop')
        return float(self.text())

    def val_update_event(self, val):
        if False:
            print('Hello World!')
        'triggered when input is connected and received new data;\n        displays the data in the widget (without updating)'
        self.block = True
        try:
            self.setText(str(val))
        except Exception as e:
            pass
        finally:
            self.block = False

    def get_state(self) -> dict:
        if False:
            while True:
                i = 10
        return {'text': self.text()}

    def set_state(self, data: dict):
        if False:
            while True:
                i = 10
        self.val_update_event(data['text'])

class Boolean_IW(DType_IW_Base, QCheckBox):

    def __init__(self, params):
        if False:
            while True:
                i = 10
        DType_IW_Base.__init__(self, params)
        QCheckBox.__init__(self)
        self.setChecked(self.dtype.val)
        self.setToolTip(self.dtype.doc)
        self.stateChanged.connect(self.state_changed)

    def state_changed(self, state):
        if False:
            while True:
                i = 10
        self.change_val(self.get_val())

    def get_val(self):
        if False:
            while True:
                i = 10
        return self.isChecked()

    def val_update_event(self, val):
        if False:
            return 10
        'triggered when input is connected and received new data;\n        displays the data in the widget (without updating)'
        self.block = True
        try:
            self.setChecked(bool(val))
        except Exception as e:
            pass
        finally:
            self.block = False

    def get_state(self) -> dict:
        if False:
            i = 10
            return i + 15
        return {'checked': self.isChecked()}

    def set_state(self, data: dict):
        if False:
            while True:
                i = 10
        self.val_update_event(data['checked'])

class Choice_IW(DType_IW_Base, QComboBox):

    def __init__(self, params):
        if False:
            i = 10
            return i + 15
        DType_IW_Base.__init__(self, params)
        QComboBox.__init__(self)
        self.addItems(self.dtype.items)
        self.setCurrentText(self.dtype.val)
        self.setToolTip(self.dtype.doc)
        self.currentTextChanged.connect(self.widget_text_changed)

    def widget_text_changed(self):
        if False:
            while True:
                i = 10
        self.change_val(self.get_val())

    def get_val(self):
        if False:
            i = 10
            return i + 15
        return self.currentText()

    def val_update_event(self, val):
        if False:
            i = 10
            return i + 15
        'triggered when input is connected and received new data;\n        displays the data in the widget (without updating)'
        self.block = True
        try:
            self.setCurrentText(val)
        except Exception as e:
            pass
        finally:
            self.block = False

    def get_state(self) -> dict:
        if False:
            i = 10
            return i + 15
        return {'items': [self.itemText(i) for i in range(self.count())], 'active': self.currentText()}

    def set_state(self, data: dict):
        if False:
            while True:
                i = 10
        self.block = True
        self.clear()
        self.addItems(data['items'])
        self.setCurrentText(data['active'])
        self.block = False