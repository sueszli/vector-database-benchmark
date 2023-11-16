import FBD_view
import FBD_model
import remi
import remi.gui as gui
import time
import inspect
import types

class PRINT(FBD_view.FunctionBlockView):

    @FBD_model.FunctionBlock.decorate_process([])
    def do(self, IN, EN=True):
        if False:
            for i in range(10):
                print('nop')
        if not EN:
            return
        print(IN)

class STRING(FBD_view.FunctionBlockView):

    @property
    @gui.editor_attribute_decorator('WidgetSpecific', 'Defines the actual value', str, {})
    def value(self):
        if False:
            print('Hello World!')
        if len(self.outputs) < 1:
            return ''
        return self.outputs['OUT'].get_value()

    @value.setter
    def value(self, value):
        if False:
            return 10
        self.outputs['OUT'].set_value(value)

    def __init__(self, name, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        FBD_view.FunctionBlockView.__init__(self, name, *args, **kwargs)
        self.outputs['OUT'].set_value('A STRING VALUE')

    @FBD_model.FunctionBlock.decorate_process(['OUT'])
    def do(self):
        if False:
            while True:
                i = 10
        OUT = self.outputs['OUT'].get_value()
        return OUT

class STRING_SWAP_CASE(FBD_view.FunctionBlockView):

    @FBD_model.FunctionBlock.decorate_process(['OUT'])
    def do(self, IN, EN=True):
        if False:
            print('Hello World!')
        if not EN:
            return
        OUT = IN.swapcase()
        return OUT

class BOOL(FBD_view.FunctionBlockView):

    @property
    @gui.editor_attribute_decorator('WidgetSpecific', 'Defines the actual value', bool, {})
    def value(self):
        if False:
            return 10
        if len(self.outputs) < 1:
            return False
        return self.outputs['OUT'].get_value()

    @value.setter
    def value(self, value):
        if False:
            print('Hello World!')
        self.outputs['OUT'].set_value(value)

    def __init__(self, name, *args, **kwargs):
        if False:
            return 10
        FBD_view.FunctionBlockView.__init__(self, name, *args, **kwargs)
        self.outputs['OUT'].set_value(False)

    @FBD_model.FunctionBlock.decorate_process(['OUT'])
    def do(self):
        if False:
            for i in range(10):
                print('nop')
        OUT = self.outputs['OUT'].get_value()
        return OUT

class RISING_EDGE(FBD_view.FunctionBlockView):
    previous_value = None

    @FBD_model.FunctionBlock.decorate_process(['OUT'])
    def do(self, IN):
        if False:
            for i in range(10):
                print('nop')
        OUT = self.previous_value != IN and IN
        self.previous_value = IN
        return OUT

class NOT(FBD_view.FunctionBlockView):

    @FBD_model.FunctionBlock.decorate_process(['OUT'])
    def do(self, IN):
        if False:
            i = 10
            return i + 15
        OUT = not IN
        return OUT

class AND(FBD_view.FunctionBlockView):

    @FBD_model.FunctionBlock.decorate_process(['OUT'])
    def do(self, IN1, IN2):
        if False:
            print('Hello World!')
        OUT = IN1 and IN2
        return OUT

class OR(FBD_view.FunctionBlockView):

    @FBD_model.FunctionBlock.decorate_process(['OUT'])
    def do(self, IN1, IN2):
        if False:
            while True:
                i = 10
        OUT = IN1 or IN2
        return OUT

class XOR(FBD_view.FunctionBlockView):

    @FBD_model.FunctionBlock.decorate_process(['OUT'])
    def do(self, IN1, IN2):
        if False:
            return 10
        OUT = IN1 != IN2
        return OUT

class PULSAR(FBD_view.FunctionBlockView):
    _ton = 1000
    _toff = 1000

    @property
    @gui.editor_attribute_decorator('WidgetSpecific', 'Defines the actual TON value', int, {'possible_values': '', 'min': 0, 'max': 65535, 'default': 0, 'step': 1})
    def ton(self):
        if False:
            print('Hello World!')
        return self._ton

    @ton.setter
    def ton(self, value):
        if False:
            print('Hello World!')
        self._ton = value

    @property
    @gui.editor_attribute_decorator('WidgetSpecific', 'Defines the actual TOFF value', int, {'possible_values': '', 'min': 0, 'max': 65535, 'default': 0, 'step': 1})
    def toff(self):
        if False:
            print('Hello World!')
        return self._toff

    @toff.setter
    def toff(self, value):
        if False:
            return 10
        self._toff = value
    tstart = 0

    def __init__(self, name, *args, **kwargs):
        if False:
            print('Hello World!')
        FBD_view.FunctionBlockView.__init__(self, name, *args, **kwargs)
        self.outputs['OUT'].set_value(False)
        self.tstart = time.time()

    @FBD_model.FunctionBlock.decorate_process(['OUT'])
    def do(self):
        if False:
            return 10
        OUT = int((time.time() - self.tstart) * 1000) % (self.ton + self.toff) < self.ton
        return OUT