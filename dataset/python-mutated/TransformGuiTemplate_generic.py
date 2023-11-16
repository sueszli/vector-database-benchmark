from ..Qt import QtCore, QtGui, QtWidgets

class Ui_Form(object):

    def setupUi(self, Form):
        if False:
            while True:
                i = 10
        Form.setObjectName('Form')
        Form.resize(224, 117)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(1)
        self.verticalLayout.setObjectName('verticalLayout')
        self.translateLabel = QtWidgets.QLabel(Form)
        self.translateLabel.setObjectName('translateLabel')
        self.verticalLayout.addWidget(self.translateLabel)
        self.rotateLabel = QtWidgets.QLabel(Form)
        self.rotateLabel.setObjectName('rotateLabel')
        self.verticalLayout.addWidget(self.rotateLabel)
        self.scaleLabel = QtWidgets.QLabel(Form)
        self.scaleLabel.setObjectName('scaleLabel')
        self.verticalLayout.addWidget(self.scaleLabel)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName('horizontalLayout')
        self.mirrorImageBtn = QtWidgets.QPushButton(Form)
        self.mirrorImageBtn.setToolTip('')
        self.mirrorImageBtn.setObjectName('mirrorImageBtn')
        self.horizontalLayout.addWidget(self.mirrorImageBtn)
        self.reflectImageBtn = QtWidgets.QPushButton(Form)
        self.reflectImageBtn.setObjectName('reflectImageBtn')
        self.horizontalLayout.addWidget(self.reflectImageBtn)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        if False:
            print('Hello World!')
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate('Form', 'PyQtGraph'))
        self.translateLabel.setText(_translate('Form', 'Translate:'))
        self.rotateLabel.setText(_translate('Form', 'Rotate:'))
        self.scaleLabel.setText(_translate('Form', 'Scale:'))
        self.mirrorImageBtn.setText(_translate('Form', 'Mirror'))
        self.reflectImageBtn.setText(_translate('Form', 'Reflect'))