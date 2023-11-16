from PyQt6 import QtCore, QtGui, QtWidgets

class Ui_ScriptingDocumentationDialog(object):

    def setupUi(self, ScriptingDocumentationDialog):
        if False:
            print('Hello World!')
        ScriptingDocumentationDialog.setObjectName('ScriptingDocumentationDialog')
        ScriptingDocumentationDialog.resize(725, 457)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(ScriptingDocumentationDialog.sizePolicy().hasHeightForWidth())
        ScriptingDocumentationDialog.setSizePolicy(sizePolicy)
        ScriptingDocumentationDialog.setModal(False)
        self.verticalLayout = QtWidgets.QVBoxLayout(ScriptingDocumentationDialog)
        self.verticalLayout.setObjectName('verticalLayout')
        self.documentation_layout = QtWidgets.QVBoxLayout()
        self.documentation_layout.setObjectName('documentation_layout')
        self.verticalLayout.addLayout(self.documentation_layout)
        self.buttonBox = QtWidgets.QDialogButtonBox(ScriptingDocumentationDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Close)
        self.buttonBox.setObjectName('buttonBox')
        self.verticalLayout.addWidget(self.buttonBox)
        self.retranslateUi(ScriptingDocumentationDialog)
        QtCore.QMetaObject.connectSlotsByName(ScriptingDocumentationDialog)

    def retranslateUi(self, ScriptingDocumentationDialog):
        if False:
            for i in range(10):
                print('nop')
        _translate = QtCore.QCoreApplication.translate
        ScriptingDocumentationDialog.setWindowTitle(_('Scripting Documentation'))