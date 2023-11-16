from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import operator
from MainWindow import Ui_MainWindow
READY = 0
INPUT = 1

class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        for n in range(0, 10):
            getattr(self, 'pushButton_n%s' % n).pressed.connect(lambda v=n: self.input_number(v))
        self.pushButton_add.pressed.connect(lambda : self.operation(operator.add))
        self.pushButton_sub.pressed.connect(lambda : self.operation(operator.sub))
        self.pushButton_mul.pressed.connect(lambda : self.operation(operator.mul))
        self.pushButton_div.pressed.connect(lambda : self.operation(operator.truediv))
        self.pushButton_pc.pressed.connect(self.operation_pc)
        self.pushButton_eq.pressed.connect(self.equals)
        self.actionReset.triggered.connect(self.reset)
        self.pushButton_ac.pressed.connect(self.reset)
        self.actionExit.triggered.connect(self.close)
        self.pushButton_m.pressed.connect(self.memory_store)
        self.pushButton_mr.pressed.connect(self.memory_recall)
        self.memory = 0
        self.reset()
        self.show()

    def display(self):
        if False:
            while True:
                i = 10
        self.lcdNumber.display(self.stack[-1])

    def reset(self):
        if False:
            print('Hello World!')
        self.state = READY
        self.stack = [0]
        self.last_operation = None
        self.current_op = None
        self.display()

    def memory_store(self):
        if False:
            return 10
        self.memory = self.lcdNumber.value()

    def memory_recall(self):
        if False:
            for i in range(10):
                print('nop')
        self.state = INPUT
        self.stack[-1] = self.memory
        self.display()

    def input_number(self, v):
        if False:
            print('Hello World!')
        if self.state == READY:
            self.state = INPUT
            self.stack[-1] = v
        else:
            self.stack[-1] = self.stack[-1] * 10 + v
        self.display()

    def operation(self, op):
        if False:
            i = 10
            return i + 15
        if self.current_op:
            self.equals()
        self.stack.append(0)
        self.state = INPUT
        self.current_op = op

    def operation_pc(self):
        if False:
            for i in range(10):
                print('nop')
        self.state = INPUT
        self.stack[-1] *= 0.01
        self.display()

    def equals(self):
        if False:
            i = 10
            return i + 15
        if self.state == READY and self.last_operation:
            (s, self.current_op) = self.last_operation
            self.stack.append(s)
        if self.current_op:
            self.last_operation = (self.stack[-1], self.current_op)
            try:
                self.stack = [self.current_op(*self.stack)]
            except Exception:
                self.lcdNumber.display('Err')
                self.stack = [0]
            else:
                self.current_op = None
                self.state = READY
                self.display()
if __name__ == '__main__':
    app = QApplication([])
    app.setApplicationName('Calculon')
    window = MainWindow()
    app.exec_()