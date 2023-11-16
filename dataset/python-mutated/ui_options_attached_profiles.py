from PyQt6 import QtCore, QtGui, QtWidgets

class Ui_AttachedProfilesDialog(object):

    def setupUi(self, AttachedProfilesDialog):
        if False:
            i = 10
            return i + 15
        AttachedProfilesDialog.setObjectName('AttachedProfilesDialog')
        AttachedProfilesDialog.resize(800, 450)
        self.vboxlayout = QtWidgets.QVBoxLayout(AttachedProfilesDialog)
        self.vboxlayout.setContentsMargins(9, 9, 9, 9)
        self.vboxlayout.setSpacing(6)
        self.vboxlayout.setObjectName('vboxlayout')
        self.options_list = QtWidgets.QTableView(AttachedProfilesDialog)
        self.options_list.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.options_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.options_list.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.options_list.setObjectName('options_list')
        self.vboxlayout.addWidget(self.options_list)
        self.buttonBox = QtWidgets.QDialogButtonBox(AttachedProfilesDialog)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.NoButton)
        self.buttonBox.setObjectName('buttonBox')
        self.vboxlayout.addWidget(self.buttonBox)
        self.retranslateUi(AttachedProfilesDialog)
        QtCore.QMetaObject.connectSlotsByName(AttachedProfilesDialog)

    def retranslateUi(self, AttachedProfilesDialog):
        if False:
            while True:
                i = 10
        _translate = QtCore.QCoreApplication.translate
        AttachedProfilesDialog.setWindowTitle(_('Profiles Attached to Options'))