from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel
from PyQt5.QtGui import QPainter, QPixmap, QFont, QFontMetrics, QBrush, QColor
from PyQt5.QtCore import Qt, QSize
from PyQt5 import QtCore
from PyQt5.QtCore import Qt as Qtc
from PyQt5.QtCore import pyqtSignal
from gnuradio import gr
import pmt

class LabeledDigitalNumberControl(QFrame):

    def __init__(self, lbl='', min_freq_hz=0, max_freq_hz=6000000000, parent=None, thousands_separator=',', background_color='black', fontColor='white', click_callback=None):
        if False:
            for i in range(10):
                print('nop')
        QFrame.__init__(self, parent)
        self.numberControl = DigitalNumberControl(min_freq_hz, max_freq_hz, self, thousands_separator, background_color, fontColor, click_callback)
        layout = QVBoxLayout()
        self.lbl = QLabel(lbl, self)
        if len:
            self.hasLabel = True
            layout.addWidget(self.lbl)
        else:
            self.hasLabel = False
        layout.addWidget(self.numberControl)
        layout.setAlignment(Qtc.AlignCenter | Qtc.AlignVCenter)
        self.setLayout(layout)
        self.show()

    def minimumSizeHint(self):
        if False:
            while True:
                i = 10
        if self.hasLabel:
            return QSize(self.numberControl.minimumWidth() + 10, 100)
        else:
            return QSize(self.numberControl.minimumWidth() + 10, 50)

    def setReadOnly(self, b_read_only):
        if False:
            print('Hello World!')
        self.numberControl.setReadOnly(b_read_only)

    def setFrequency(self, new_freq):
        if False:
            while True:
                i = 10
        self.numberControl.setFrequency(new_freq)

    def getFrequency(self):
        if False:
            i = 10
            return i + 15
        return self.numberControl.getFrequency()

class DigitalNumberControl(QFrame):
    updateInt = pyqtSignal(int)
    updateFloat = pyqtSignal(float)

    def __init__(self, min_freq_hz=0, max_freq_hz=6000000000, parent=None, thousands_separator=',', background_color='black', fontColor='white', click_callback=None):
        if False:
            print('Hello World!')
        QFrame.__init__(self, parent)
        self.updateInt.connect(self.onUpdateInt)
        self.updateFloat.connect(self.onUpdateFloat)
        self.min_freq = int(min_freq_hz)
        self.max_freq = int(max_freq_hz)
        self.numDigitsInFreq = len(str(max_freq_hz))
        self.thousands_separator = thousands_separator
        self.click_callback = click_callback
        self.read_only = False
        self.setColors(QColor(background_color), QColor(fontColor))
        self.numberFont = QFont('Arial', 12, QFont.Normal)
        self.cur_freq = min_freq_hz
        teststr = ''
        for i in range(0, self.numDigitsInFreq):
            teststr += '0'
        fm = QFontMetrics(self.numberFont)
        if len(self.thousands_separator) > 0:
            numgroups = int(float(self.numDigitsInFreq - 1) / 3.0)
            if numgroups > 0:
                for i in range(0, numgroups):
                    teststr += self.thousands_separator
            textstr = teststr
        else:
            textstr = teststr
        width = fm.width(textstr)
        self.minwidth = width
        if self.minwidth < 410:
            self.minwidth = 410
        self.setMaximumHeight(70)
        self.setMinimumWidth(self.minwidth)
        self.show()

    def minimumSizeHint(self):
        if False:
            while True:
                i = 10
        return QSize(self.minwidth, 50)

    def setReadOnly(self, b_read_only):
        if False:
            for i in range(10):
                print('nop')
        self.read_only = b_read_only

    def mousePressEvent(self, event):
        if False:
            return 10
        super(DigitalNumberControl, self).mousePressEvent(event)
        self.offset = event.pos()
        if self.read_only:
            gr.log.trace('click received but read-only.  Not changing frequency.')
            return
        fm = QFontMetrics(self.numberFont)
        if len(self.thousands_separator) > 0:
            if self.thousands_separator != '.':
                textstr = format(self.getFrequency(), self.thousands_separator)
            else:
                textstr = format(self.getFrequency(), ',')
                textstr = textstr.replace(',', '.')
        else:
            textstr = str(self.getFrequency())
        width = fm.width(textstr)
        clickpos = self.size().width() - 2 - self.offset.x()
        found_number = False
        clicked_thousands = False
        for i in range(1, len(textstr) + 1):
            width = fm.width(textstr[-i:])
            charstr = textstr[-i:]
            widthchar = fm.width(charstr[0])
            if clickpos >= width - widthchar and clickpos <= width:
                clicked_char = i - 1
                clicked_num_index = clicked_char
                found_number = True
                if len(self.thousands_separator) > 0:
                    if charstr[0] != self.thousands_separator:
                        numSeps = charstr.count(self.thousands_separator)
                        clicked_num_index -= numSeps
                        gr.log.trace(f'clicked number: {clicked_num_index}')
                    else:
                        clicked_thousands = True
                        gr.log.trace('clicked thousands separator')
                else:
                    gr.log.trace('clicked number: ' + str(clicked_char))
                clicked_up = False
                if self.offset.y() > self.size().height() / 2:
                    gr.log.trace('clicked down')
                else:
                    gr.log.trace('clicked up')
                    clicked_up = True
                if not clicked_thousands:
                    cur_freq = self.getFrequency()
                    increment = pow(10, clicked_num_index)
                    if clicked_up:
                        cur_freq += increment
                    else:
                        cur_freq -= increment
                    self.setFrequencyNow(cur_freq)
                    if self.click_callback is not None:
                        self.click_callback(self.getFrequency())
                break
        if not found_number and (not clicked_thousands):
            clicked_up = False
            if self.offset.y() > self.size().height() / 2:
                gr.log.trace('clicked down in the high area')
            else:
                gr.log.trace('clicked up in the high area')
                clicked_up = True
            textstr = str(self.getFrequency())
            numNumbers = len(textstr)
            increment = pow(10, numNumbers)
            cur_freq = self.getFrequency()
            if clicked_up:
                cur_freq += increment
            else:
                cur_freq -= increment
            self.setFrequencyNow(cur_freq)
            if self.click_callback is not None:
                gr.log.trace('Calling self.click_callback')
                self.click_callback(self.getFrequency())
            else:
                gr.log.trace('self.click_callback is None.  Not calling callback.')

    def setColors(self, background, fontColor):
        if False:
            print('Hello World!')
        self.background_color = background
        self.fontColor = fontColor

    def reverseString(self, astring):
        if False:
            print('Hello World!')
        astring = astring[::-1]
        return astring

    def onUpdateInt(self, new_freq):
        if False:
            return 10
        if new_freq >= self.min_freq and new_freq <= self.max_freq:
            self.cur_freq = int(new_freq)
        self.update()

    def onUpdateFloat(self, new_freq):
        if False:
            return 10
        if new_freq >= self.min_freq and new_freq <= self.max_freq:
            self.cur_freq = int(new_freq)
        self.update()

    def setFrequencyNow(self, new_freq):
        if False:
            return 10
        if new_freq >= self.min_freq and new_freq <= self.max_freq:
            self.cur_freq = int(new_freq)
            self.update()

    def setFrequency(self, new_freq):
        if False:
            print('Hello World!')
        if type(new_freq) == int:
            self.updateInt.emit(new_freq)
        else:
            self.updateFloat.emit(new_freq)

    def getFrequency(self):
        if False:
            return 10
        return self.cur_freq

    def resizeEvent(self, event):
        if False:
            while True:
                i = 10
        self.pxMap = QPixmap(self.size())
        self.pxMap.fill(self.background_color)
        self.update()

    def paintEvent(self, event):
        if False:
            while True:
                i = 10
        super().paintEvent(event)
        painter = QPainter(self)
        size = self.size()
        brush = QBrush()
        brush.setColor(self.background_color)
        brush.setStyle(Qt.SolidPattern)
        rect = QtCore.QRect(2, 2, size.width() - 4, size.height() - 4)
        painter.fillRect(rect, brush)
        self.numberFont.setPixelSize(int(0.9 * size.height()))
        painter.setFont(self.numberFont)
        painter.setPen(self.fontColor)
        rect = event.rect()
        if len(self.thousands_separator) > 0:
            if self.thousands_separator != '.':
                textstr = format(self.getFrequency(), self.thousands_separator)
            else:
                textstr = format(self.getFrequency(), ',')
                textstr = textstr.replace(',', '.')
        else:
            textstr = str(self.getFrequency())
        rect = QtCore.QRect(0, 0, size.width() - 4, size.height())
        painter.drawText(rect, Qt.AlignRight + Qt.AlignVCenter, textstr)

class MsgDigitalNumberControl(gr.sync_block, LabeledDigitalNumberControl):
    """
    GNU Radio Block Class
    """

    def __init__(self, lbl='', min_freq_hz=0, max_freq_hz=6000000000, parent=None, thousands_separator=',', background_color='black', fontColor='white', var_callback=None, outputmsgname='freq'):
        if False:
            while True:
                i = 10
        gr.sync_block.__init__(self, name='MsgDigitalNumberControl', in_sig=None, out_sig=None)
        LabeledDigitalNumberControl.__init__(self, lbl, min_freq_hz, max_freq_hz, parent, thousands_separator, background_color, fontColor, self.click_callback)
        self.var_callback = var_callback
        self.outputmsgname = outputmsgname
        self.message_port_register_in(pmt.intern('valuein'))
        self.set_msg_handler(pmt.intern('valuein'), self.msgHandler)
        self.message_port_register_out(pmt.intern('valueout'))

    def msgHandler(self, msg):
        if False:
            return 10
        try:
            new_val = pmt.to_python(pmt.cdr(msg))
            if type(new_val) == float or type(new_val) == int:
                self.call_var_callback(new_val)
                self.setValue(new_val)
            else:
                gr.log.error('Value received was not an int or a float. %s' % str(type(new_val)))
        except Exception as e:
            gr.log.error('Error with message conversion: %s' % str(e))

    def call_var_callback(self, new_value):
        if False:
            print('Hello World!')
        if self.var_callback is not None:
            if type(self.var_callback) is float:
                self.var_callback = float(new_value)
            else:
                self.var_callback(float(new_value))

    def click_callback(self, new_value):
        if False:
            while True:
                i = 10
        self.call_var_callback(new_value)
        self.message_port_pub(pmt.intern('valueout'), pmt.cons(pmt.intern(self.outputmsgname), pmt.from_double(float(new_value))))

    def setValue(self, new_val):
        if False:
            i = 10
            return i + 15
        self.setFrequency(new_val)
        self.message_port_pub(pmt.intern('valueout'), pmt.cons(pmt.intern(self.outputmsgname), pmt.from_double(float(new_val))))

    def getValue(self):
        if False:
            for i in range(10):
                print('nop')
        self.getFrequency()

    def setReadOnly(self, b_read_only):
        if False:
            for i in range(10):
                print('nop')
        super().setReadOnly(b_read_only)