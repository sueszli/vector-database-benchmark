# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\QGroup_432987409\WoHowLearn\0.M_I_pyqt\partner_625781186\5.hoverMenu\Main.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(908, 518)
        MainWindow.setStyleSheet("* #MainWindow {  border:none;}\n"
" #widget{   border:none;  }\n"
"/*#topWidget QWidget{  background-color: green;   }\n"
"#widget  QLabel{ background-color: rgb(85, 0, 255);   }*/\n"
"")
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setStyleSheet("#centralWidget {border:none;}")
        self.centralWidget.setObjectName("centralWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralWidget)
        self.verticalLayout.setContentsMargins(0, 2, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.toggleButton = QtWidgets.QPushButton(self.centralWidget)
        self.toggleButton.setStyleSheet("background-color:#1d4371;\n"
"\n"
"border-radius:0px;")
        self.toggleButton.setFlat(True)
        self.toggleButton.setObjectName("toggleButton")
        self.verticalLayout.addWidget(self.toggleButton)
        self.widget = QtWidgets.QWidget(self.centralWidget)
        self.widget.setStyleSheet("#widget {border:none;}")
        self.widget.setObjectName("widget")
        self.verticalLayout_14 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_14.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_14.setSpacing(0)
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.topWidget = QtWidgets.QWidget(self.widget)
        self.topWidget.setStyleSheet("")
        self.topWidget.setObjectName("topWidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.topWidget)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setSpacing(2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout_19 = QtWidgets.QVBoxLayout()
        self.verticalLayout_19.setContentsMargins(-1, -1, 6, -1)
        self.verticalLayout_19.setSpacing(2)
        self.verticalLayout_19.setObjectName("verticalLayout_19")
        self.label_17 = QtWidgets.QLabel(self.topWidget)
        self.label_17.setAlignment(QtCore.Qt.AlignCenter)
        self.label_17.setObjectName("label_17")
        self.verticalLayout_19.addWidget(self.label_17)
        self.label_18 = QtWidgets.QLabel(self.topWidget)
        self.label_18.setAlignment(QtCore.Qt.AlignCenter)
        self.label_18.setObjectName("label_18")
        self.verticalLayout_19.addWidget(self.label_18)
        self.gridLayout_2.addLayout(self.verticalLayout_19, 1, 4, 1, 1)
        self.W2 = SingeleWidget(self.topWidget)
        self.W2.setObjectName("W2")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.W2)
        self.verticalLayout_5.setContentsMargins(9, 9, 9, 9)
        self.verticalLayout_5.setSpacing(2)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setSpacing(2)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.filemenu_storeData_6 = B2(self.W2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.filemenu_storeData_6.sizePolicy().hasHeightForWidth())
        self.filemenu_storeData_6.setSizePolicy(sizePolicy)
        self.filemenu_storeData_6.setText("")
        self.filemenu_storeData_6.setObjectName("filemenu_storeData_6")
        self.horizontalLayout_5.addWidget(self.filemenu_storeData_6)
        self.verticalLayout_5.addLayout(self.horizontalLayout_5)
        self.label_22 = QtWidgets.QLabel(self.W2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_22.sizePolicy().hasHeightForWidth())
        self.label_22.setSizePolicy(sizePolicy)
        self.label_22.setAlignment(QtCore.Qt.AlignCenter)
        self.label_22.setObjectName("label_22")
        self.verticalLayout_5.addWidget(self.label_22)
        self.gridLayout_2.addWidget(self.W2, 1, 1, 1, 1)
        self.W4 = SingeleWidget(self.topWidget)
        self.W4.setObjectName("W4")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.W4)
        self.verticalLayout_7.setContentsMargins(9, 9, 9, 9)
        self.verticalLayout_7.setSpacing(2)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setSpacing(2)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.filemenu_storeData_8 = B4(self.W4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.filemenu_storeData_8.sizePolicy().hasHeightForWidth())
        self.filemenu_storeData_8.setSizePolicy(sizePolicy)
        self.filemenu_storeData_8.setText("")
        self.filemenu_storeData_8.setObjectName("filemenu_storeData_8")
        self.horizontalLayout_7.addWidget(self.filemenu_storeData_8)
        self.verticalLayout_7.addLayout(self.horizontalLayout_7)
        self.label_24 = QtWidgets.QLabel(self.W4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_24.sizePolicy().hasHeightForWidth())
        self.label_24.setSizePolicy(sizePolicy)
        self.label_24.setAlignment(QtCore.Qt.AlignCenter)
        self.label_24.setObjectName("label_24")
        self.verticalLayout_7.addWidget(self.label_24)
        self.gridLayout_2.addWidget(self.W4, 1, 3, 1, 1)
        self.W1 = SingeleWidget(self.topWidget)
        self.W1.setObjectName("W1")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.W1)
        self.verticalLayout_2.setContentsMargins(9, 9, 9, 9)
        self.verticalLayout_2.setSpacing(2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSpacing(2)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.filemenu_storeData_3 = B1(self.W1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.filemenu_storeData_3.sizePolicy().hasHeightForWidth())
        self.filemenu_storeData_3.setSizePolicy(sizePolicy)
        self.filemenu_storeData_3.setText("")
        self.filemenu_storeData_3.setObjectName("filemenu_storeData_3")
        self.horizontalLayout_2.addWidget(self.filemenu_storeData_3)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.label_15 = QtWidgets.QLabel(self.W1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_15.sizePolicy().hasHeightForWidth())
        self.label_15.setSizePolicy(sizePolicy)
        self.label_15.setAlignment(QtCore.Qt.AlignCenter)
        self.label_15.setObjectName("label_15")
        self.verticalLayout_2.addWidget(self.label_15)
        self.gridLayout_2.addWidget(self.W1, 1, 0, 1, 1)
        self.W3 = SingeleWidget(self.topWidget)
        self.W3.setObjectName("W3")
        self.gridLayout = QtWidgets.QGridLayout(self.W3)
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetMinAndMaxSize)
        self.gridLayout.setContentsMargins(9, 9, 9, 9)
        self.gridLayout.setSpacing(2)
        self.gridLayout.setObjectName("gridLayout")
        self.label_23 = QtWidgets.QLabel(self.W3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_23.sizePolicy().hasHeightForWidth())
        self.label_23.setSizePolicy(sizePolicy)
        self.label_23.setAlignment(QtCore.Qt.AlignCenter)
        self.label_23.setWordWrap(False)
        self.label_23.setObjectName("label_23")
        self.gridLayout.addWidget(self.label_23, 1, 0, 1, 1)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setSpacing(2)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.filemenu_storeData_7 = B3(self.W3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.filemenu_storeData_7.sizePolicy().hasHeightForWidth())
        self.filemenu_storeData_7.setSizePolicy(sizePolicy)
        self.filemenu_storeData_7.setText("")
        self.filemenu_storeData_7.setObjectName("filemenu_storeData_7")
        self.horizontalLayout_6.addWidget(self.filemenu_storeData_7)
        self.gridLayout.addLayout(self.horizontalLayout_6, 0, 0, 1, 1)
        self.gridLayout_2.addWidget(self.W3, 1, 2, 1, 1)
        self.verticalLayout_14.addWidget(self.topWidget)
        self.bottomWidget = QtWidgets.QWidget(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bottomWidget.sizePolicy().hasHeightForWidth())
        self.bottomWidget.setSizePolicy(sizePolicy)
        self.bottomWidget.setObjectName("bottomWidget")
        self.Bottom_Vbox = QtWidgets.QVBoxLayout(self.bottomWidget)
        self.Bottom_Vbox.setContentsMargins(0, 0, 0, 0)
        self.Bottom_Vbox.setSpacing(0)
        self.Bottom_Vbox.setObjectName("Bottom_Vbox")
        self.verticalLayout_14.addWidget(self.bottomWidget)
        self.verticalLayout.addWidget(self.widget)
        MainWindow.setCentralWidget(self.centralWidget)
        self.action1 = QtWidgets.QAction(MainWindow)
        self.action1.setObjectName("action1")
        self.action2 = QtWidgets.QAction(MainWindow)
        self.action2.setObjectName("action2")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.toggleButton.setText(_translate("MainWindow", "↑"))
        self.label_17.setText(_translate("MainWindow", "个人中心"))
        self.label_18.setText(_translate("MainWindow", "退出登录"))
        self.label_22.setText(_translate("MainWindow", "店铺数据"))
        self.label_24.setText(_translate("MainWindow", "店铺"))
        self.label_15.setText(_translate("MainWindow", "店铺数据"))
        self.label_23.setText(_translate("MainWindow", "店店店"))
        self.action1.setText(_translate("MainWindow", "1"))
        self.action2.setText(_translate("MainWindow", "2"))

from U_FuncWidget.BaseElement import SingeleWidget
from U_FuncWidget.Menu import B1, B2, B3, B4
try:
    import tbqrc_rc
except:
    from . import tbqrc_rc

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

