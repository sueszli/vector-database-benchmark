"""
Created on 2018年1月30日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: CalendarQssStyle
@description: 日历美化样式
"""
import sys
try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QTextCharFormat, QBrush, QColor
    from PyQt5.QtWidgets import QApplication, QCalendarWidget
except ImportError:
    from PySide2.QtCore import Qt
    from PySide2.QtGui import QTextCharFormat, QBrush, QColor
    from PySide2.QtWidgets import QApplication, QCalendarWidget
StyleSheet = '\n/*顶部导航区域*/\n#qt_calendar_navigationbar {\n    background-color: rgb(0, 188, 212);\n    min-height: 100px;\n}\n\n\n/*上一个月按钮和下一个月按钮(从源码里找到的objectName)*/\n#qt_calendar_prevmonth, #qt_calendar_nextmonth {\n    border: none; /*去掉边框*/\n    margin-top: 64px;\n    color: white;\n    min-width: 36px;\n    max-width: 36px;\n    min-height: 36px;\n    max-height: 36px;\n    border-radius: 18px; /*看来近似椭圆*/\n    font-weight: bold; /*字体加粗*/\n    qproperty-icon: none; /*去掉默认的方向键图片，当然也可以自定义*/\n    background-color: transparent;/*背景颜色透明*/\n}\n#qt_calendar_prevmonth {\n    qproperty-text: "<"; /*修改按钮的文字*/\n}\n#qt_calendar_nextmonth {\n    qproperty-text: ">";\n}\n#qt_calendar_prevmonth:hover, #qt_calendar_nextmonth:hover {\n    background-color: rgba(225, 225, 225, 100);\n}\n#qt_calendar_prevmonth:pressed, #qt_calendar_nextmonth:pressed {\n    background-color: rgba(235, 235, 235, 100);\n}\n\n\n/*年,月控件*/\n#qt_calendar_yearbutton, #qt_calendar_monthbutton {\n    color: white;\n    margin: 18px;\n    min-width: 60px;\n    border-radius: 30px;\n}\n#qt_calendar_yearbutton:hover, #qt_calendar_monthbutton:hover {\n    background-color: rgba(225, 225, 225, 100);\n}\n#qt_calendar_yearbutton:pressed, #qt_calendar_monthbutton:pressed {\n    background-color: rgba(235, 235, 235, 100);\n}\n\n\n/*年份输入框*/\n#qt_calendar_yearedit {\n    min-width: 50px;\n    color: white;\n    background: transparent;/*让输入框背景透明*/\n}\n#qt_calendar_yearedit::up-button { /*往上的按钮*/\n    width: 20px;\n    subcontrol-position: right;/*移动到右边*/\n}\n#qt_calendar_yearedit::down-button { /*往下的按钮*/\n    width: 20px;\n    subcontrol-position: left; /*移动到左边去*/\n}\n\n\n/*月份选择菜单*/\nCalendarWidget QToolButton QMenu {\n     background-color: white;\n}\nCalendarWidget QToolButton QMenu::item {\n    padding: 10px;\n}\nCalendarWidget QToolButton QMenu::item:selected:enabled {\n    background-color: rgb(230, 230, 230);\n}\nCalendarWidget QToolButton::menu-indicator {\n    /*image: none;去掉月份选择下面的小箭头*/\n    subcontrol-position: right center;/*右边居中*/\n}\n\n\n/*下方的日历表格*/\n#qt_calendar_calendarview {\n    outline: 0px;/*去掉选中后的虚线框*/\n    selection-background-color: rgb(0, 188, 212); /*选中背景颜色*/\n}\n'

class CalendarWidget(QCalendarWidget):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(CalendarWidget, self).__init__(*args, **kwargs)
        self.setVerticalHeaderFormat(self.NoVerticalHeader)
        fmtGreen = QTextCharFormat()
        fmtGreen.setForeground(QBrush(Qt.green))
        self.setWeekdayTextFormat(Qt.Saturday, fmtGreen)
        fmtOrange = QTextCharFormat()
        fmtOrange.setForeground(QBrush(QColor(252, 140, 28)))
        self.setWeekdayTextFormat(Qt.Sunday, fmtOrange)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(StyleSheet)
    w = CalendarWidget()
    w.show()
    sys.exit(app.exec_())