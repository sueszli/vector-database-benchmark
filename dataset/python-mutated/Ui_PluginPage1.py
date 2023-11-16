from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):

    def setupUi(self, Form):
        if False:
            print('Hello World!')
        Form.setObjectName('Form')
        Form.resize(400, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName('verticalLayout')
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setObjectName('pushButton')
        self.verticalLayout.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setObjectName('pushButton_2')
        self.verticalLayout.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setObjectName('pushButton_3')
        self.verticalLayout.addWidget(self.pushButton_3)
        self.lineEdit = QtWidgets.QLineEdit(Form)
        self.lineEdit.setObjectName('lineEdit')
        self.verticalLayout.addWidget(self.lineEdit)
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        if False:
            return 10
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate('Form', 'Form'))
        self.pushButton.setText(_translate('Form', 'btn1'))
        self.pushButton_2.setText(_translate('Form', 'btn2'))
        self.pushButton_3.setText(_translate('Form', 'btn3'))
        self.lineEdit.setText(_translate('Form', '测试页面1'))
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())