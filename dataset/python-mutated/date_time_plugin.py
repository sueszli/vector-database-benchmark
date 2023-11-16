from PyQt5.QtCore import Qt
from PyQt5.QtDesigner import QPyDesignerCustomWidgetPlugin
from qfluentwidgets import DatePicker, TimePicker, ZhDatePicker, AMTimePicker, CalendarPicker
from plugin_base import PluginBase

class DateTimePlugin(PluginBase):

    def group(self):
        if False:
            for i in range(10):
                print('nop')
        return super().group() + ' (Date Time)'

class CalendarPickerPlugin(DateTimePlugin, QPyDesignerCustomWidgetPlugin):
    """ Calendar picker plugin """

    def createWidget(self, parent):
        if False:
            i = 10
            return i + 15
        return CalendarPicker(parent)

    def icon(self):
        if False:
            return 10
        return super().icon('CalendarDatePicker')

    def name(self):
        if False:
            i = 10
            return i + 15
        return 'CalendarPicker'

class DatePickerPlugin(DateTimePlugin, QPyDesignerCustomWidgetPlugin):
    """ Date picker plugin """

    def createWidget(self, parent):
        if False:
            print('Hello World!')
        return DatePicker(parent)

    def icon(self):
        if False:
            return 10
        return super().icon('DatePicker')

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'DatePicker'

class ZhDatePickerPlugin(DateTimePlugin, QPyDesignerCustomWidgetPlugin):
    """ Chinese Date picker plugin """

    def createWidget(self, parent):
        if False:
            return 10
        return ZhDatePicker(parent)

    def icon(self):
        if False:
            print('Hello World!')
        return super().icon('DatePicker')

    def name(self):
        if False:
            print('Hello World!')
        return 'ZhDatePicker'

    def toolTip(self):
        if False:
            return 10
        return 'Chinese date picker'

class TimePickerPlugin(DateTimePlugin, QPyDesignerCustomWidgetPlugin):
    """ Time picker plugin """

    def createWidget(self, parent):
        if False:
            print('Hello World!')
        return TimePicker(parent)

    def icon(self):
        if False:
            for i in range(10):
                print('nop')
        return super().icon('TimePicker')

    def name(self):
        if False:
            while True:
                i = 10
        return 'TimePicker'

    def toolTip(self):
        if False:
            i = 10
            return i + 15
        return '24 hours time picker'

class AMTimePickerPlugin(DateTimePlugin, QPyDesignerCustomWidgetPlugin):
    """ AM/PM time picker plugin """

    def createWidget(self, parent):
        if False:
            i = 10
            return i + 15
        return AMTimePicker(parent)

    def icon(self):
        if False:
            for i in range(10):
                print('nop')
        return super().icon('TimePicker')

    def name(self):
        if False:
            while True:
                i = 10
        return 'AMTimePicker'

    def toolTip(self):
        if False:
            while True:
                i = 10
        return 'AM/PM time picker'