from PyQt6 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):

    def setupUi(self, Dialog):
        if False:
            for i in range(10):
                print('nop')
        Dialog.setObjectName('Dialog')
        Dialog.resize(800, 450)
        self.vboxlayout = QtWidgets.QVBoxLayout(Dialog)
        self.vboxlayout.setContentsMargins(9, 9, 9, 9)
        self.vboxlayout.setSpacing(6)
        self.vboxlayout.setObjectName('vboxlayout')
        self.dialog_splitter = QtWidgets.QSplitter(Dialog)
        self.dialog_splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.dialog_splitter.setObjectName('dialog_splitter')
        self.pages_tree = QtWidgets.QTreeWidget(self.dialog_splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pages_tree.sizePolicy().hasHeightForWidth())
        self.pages_tree.setSizePolicy(sizePolicy)
        self.pages_tree.setObjectName('pages_tree')
        self.pages_stack = QtWidgets.QStackedWidget(self.dialog_splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pages_stack.sizePolicy().hasHeightForWidth())
        self.pages_stack.setSizePolicy(sizePolicy)
        self.pages_stack.setObjectName('pages_stack')
        self.vboxlayout.addWidget(self.dialog_splitter)
        self.buttonbox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonbox.setMinimumSize(QtCore.QSize(0, 0))
        self.buttonbox.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.buttonbox.setObjectName('buttonbox')
        self.vboxlayout.addWidget(self.buttonbox)
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        if False:
            while True:
                i = 10
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_('Options'))