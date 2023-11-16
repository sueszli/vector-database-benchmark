from PyQt5.QtCore import Qt
from PyQt5.QtDesigner import QPyDesignerCustomWidgetPlugin
from qfluentwidgets import SpinBox, CompactSpinBox, DoubleSpinBox, CompactDoubleSpinBox, TextEdit, TimeEdit, CompactTimeEdit, DateTimeEdit, CompactDateTimeEdit, LineEdit, PlainTextEdit, DateEdit, CompactDateEdit, SearchLineEdit, PasswordLineEdit
from plugin_base import PluginBase

class TextPlugin(PluginBase):

    def group(self):
        if False:
            return 10
        return super().group() + ' (Text)'

class LineEditPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Line edit plugin """

    def createWidget(self, parent):
        if False:
            return 10
        return LineEdit(parent)

    def icon(self):
        if False:
            print('Hello World!')
        return super().icon('TextBox')

    def name(self):
        if False:
            return 10
        return 'LineEdit'

class SearchLineEditPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Search line edit plugin """

    def createWidget(self, parent):
        if False:
            i = 10
            return i + 15
        return SearchLineEdit(parent)

    def icon(self):
        if False:
            i = 10
            return i + 15
        return super().icon('TextBox')

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'SearchLineEdit'

class PasswordLineEditPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Password line edit plugin """

    def createWidget(self, parent):
        if False:
            print('Hello World!')
        return PasswordLineEdit(parent)

    def icon(self):
        if False:
            for i in range(10):
                print('nop')
        return super().icon('PasswordBox')

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'PasswordLineEdit'

class TextEditPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Text edit plugin """

    def createWidget(self, parent):
        if False:
            print('Hello World!')
        return TextEdit(parent)

    def icon(self):
        if False:
            while True:
                i = 10
        return super().icon('RichEditBox')

    def name(self):
        if False:
            return 10
        return 'TextEdit'

class PlainTextEditPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Plain text edit plugin """

    def createWidget(self, parent):
        if False:
            return 10
        return PlainTextEdit(parent)

    def icon(self):
        if False:
            i = 10
            return i + 15
        return super().icon('RichEditBox')

    def name(self):
        if False:
            i = 10
            return i + 15
        return 'PlainTextEdit'

class DateEditPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Date edit plugin """

    def createWidget(self, parent):
        if False:
            return 10
        return DateEdit(parent)

    def icon(self):
        if False:
            for i in range(10):
                print('nop')
        return super().icon('NumberBox')

    def name(self):
        if False:
            return 10
        return 'DateEdit'

class TimeEditPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Time edit plugin """

    def createWidget(self, parent):
        if False:
            while True:
                i = 10
        return TimeEdit(parent)

    def icon(self):
        if False:
            return 10
        return super().icon('NumberBox')

    def name(self):
        if False:
            while True:
                i = 10
        return 'TimeEdit'

class DateTimeEditPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Date time edit plugin """

    def createWidget(self, parent):
        if False:
            print('Hello World!')
        return DateTimeEdit(parent)

    def icon(self):
        if False:
            return 10
        return super().icon('NumberBox')

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'DateTimeEdit'

class SpinBoxPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Spin box plugin """

    def createWidget(self, parent):
        if False:
            i = 10
            return i + 15
        return SpinBox(parent)

    def icon(self):
        if False:
            print('Hello World!')
        return super().icon('NumberBox')

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'SpinBox'

class DoubleSpinBoxPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Double spin box plugin """

    def createWidget(self, parent):
        if False:
            while True:
                i = 10
        return DoubleSpinBox(parent)

    def icon(self):
        if False:
            i = 10
            return i + 15
        return super().icon('NumberBox')

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'DoubleSpinBox'

class CompactDateEditPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Compact date edit plugin """

    def createWidget(self, parent):
        if False:
            i = 10
            return i + 15
        return CompactDateEdit(parent)

    def icon(self):
        if False:
            print('Hello World!')
        return super().icon('NumberBox')

    def name(self):
        if False:
            return 10
        return 'CompactDateEdit'

class CompactTimeEditPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Compact time edit plugin """

    def createWidget(self, parent):
        if False:
            return 10
        return CompactTimeEdit(parent)

    def icon(self):
        if False:
            i = 10
            return i + 15
        return super().icon('NumberBox')

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'CompactTimeEdit'

class CompactDateTimeEditPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Date time edit plugin """

    def createWidget(self, parent):
        if False:
            print('Hello World!')
        return CompactDateTimeEdit(parent)

    def icon(self):
        if False:
            while True:
                i = 10
        return super().icon('NumberBox')

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'CompactDateTimeEdit'

class CompactSpinBoxPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Compact spin box plugin """

    def createWidget(self, parent):
        if False:
            print('Hello World!')
        return CompactSpinBox(parent)

    def icon(self):
        if False:
            return 10
        return super().icon('NumberBox')

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'CompactSpinBox'

class CompactDoubleSpinBoxPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Compact double spin box plugin """

    def createWidget(self, parent):
        if False:
            for i in range(10):
                print('nop')
        return CompactDoubleSpinBox(parent)

    def icon(self):
        if False:
            for i in range(10):
                print('nop')
        return super().icon('NumberBox')

    def name(self):
        if False:
            i = 10
            return i + 15
        return 'CompactDoubleSpinBox'