from PyQt5.QtCore import Qt
from qfluentwidgets import DatePicker, TimePicker, AMTimePicker, ZhDatePicker, CalendarPicker
from .gallery_interface import GalleryInterface
from ..common.translator import Translator

class DateTimeInterface(GalleryInterface):
    """ Date time interface """

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        t = Translator()
        super().__init__(title=t.dateTime, subtitle='qfluentwidgets.components.time_picker', parent=parent)
        self.setObjectName('dateTimeInterface')
        self.addExampleCard(title=self.tr('A simple CalendarPicker'), widget=CalendarPicker(self), sourcePath='https://github.com/zhiyiYo/PyQt-Fluent-Widgets/blob/master/examples/date_time/calendar_picker/demo.py')
        w = CalendarPicker(self)
        w.setDateFormat(Qt.TextDate)
        self.addExampleCard(title=self.tr('A CalendarPicker in another format'), widget=w, sourcePath='https://github.com/zhiyiYo/PyQt-Fluent-Widgets/blob/master/examples/date_time/calendar_picker/demo.py')
        self.addExampleCard(title=self.tr('A simple DatePicker'), widget=DatePicker(self), sourcePath='https://github.com/zhiyiYo/PyQt-Fluent-Widgets/blob/master/examples/date_time/time_picker/demo.py')
        self.addExampleCard(title=self.tr('A DatePicker in another format'), widget=ZhDatePicker(self), sourcePath='https://github.com/zhiyiYo/PyQt-Fluent-Widgets/blob/master/examples/date_time/time_picker/demo.py')
        self.addExampleCard(title=self.tr('A simple TimePicker'), widget=AMTimePicker(self), sourcePath='https://github.com/zhiyiYo/PyQt-Fluent-Widgets/blob/master/examples/date_time/time_picker/demo.py')
        self.addExampleCard(title=self.tr('A TimePicker using a 24-hour clock'), widget=TimePicker(self), sourcePath='https://github.com/zhiyiYo/PyQt-Fluent-Widgets/blob/master/examples/date_time/time_picker/demo.py')
        self.addExampleCard(title=self.tr('A TimePicker with seconds column'), widget=TimePicker(self, True), sourcePath='https://github.com/zhiyiYo/PyQt-Fluent-Widgets/blob/master/examples/date_time/time_picker/demo.py')