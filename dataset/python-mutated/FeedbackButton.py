from ..Qt import QtCore, QtWidgets
__all__ = ['FeedbackButton']

class FeedbackButton(QtWidgets.QPushButton):
    """
    QPushButton which flashes success/failure indication for slow or asynchronous procedures.
    """
    sigCallSuccess = QtCore.Signal(object, object, object)
    sigCallFailure = QtCore.Signal(object, object, object)
    sigCallProcess = QtCore.Signal(object, object, object)
    sigReset = QtCore.Signal()

    def __init__(self, *args):
        if False:
            i = 10
            return i + 15
        QtWidgets.QPushButton.__init__(self, *args)
        self.origStyle = None
        self.origText = self.text()
        self.origStyle = self.styleSheet()
        self.origTip = self.toolTip()
        self.limitedTime = True
        self.sigCallSuccess.connect(self.success)
        self.sigCallFailure.connect(self.failure)
        self.sigCallProcess.connect(self.processing)
        self.sigReset.connect(self.reset)

    def feedback(self, success, message=None, tip='', limitedTime=True):
        if False:
            i = 10
            return i + 15
        'Calls success() or failure(). If you want the message to be displayed until the user takes an action, set limitedTime to False. Then call self.reset() after the desired action.Threadsafe.'
        if success:
            self.success(message, tip, limitedTime=limitedTime)
        else:
            self.failure(message, tip, limitedTime=limitedTime)

    def success(self, message=None, tip='', limitedTime=True):
        if False:
            return 10
        'Displays specified message on button and flashes button green to let user know action was successful. If you want the success to be displayed until the user takes an action, set limitedTime to False. Then call self.reset() after the desired action. Threadsafe.'
        isGuiThread = QtCore.QThread.currentThread() == QtCore.QCoreApplication.instance().thread()
        if isGuiThread:
            self.setEnabled(True)
            self.startBlink('#0F0', message, tip, limitedTime=limitedTime)
        else:
            self.sigCallSuccess.emit(message, tip, limitedTime)

    def failure(self, message=None, tip='', limitedTime=True):
        if False:
            i = 10
            return i + 15
        'Displays specified message on button and flashes button red to let user know there was an error. If you want the error to be displayed until the user takes an action, set limitedTime to False. Then call self.reset() after the desired action. Threadsafe. '
        isGuiThread = QtCore.QThread.currentThread() == QtCore.QCoreApplication.instance().thread()
        if isGuiThread:
            self.setEnabled(True)
            self.startBlink('#F00', message, tip, limitedTime=limitedTime)
        else:
            self.sigCallFailure.emit(message, tip, limitedTime)

    def processing(self, message='Processing..', tip='', processEvents=True):
        if False:
            while True:
                i = 10
        'Displays specified message on button to let user know the action is in progress. Threadsafe. '
        isGuiThread = QtCore.QThread.currentThread() == QtCore.QCoreApplication.instance().thread()
        if isGuiThread:
            self.setEnabled(False)
            self.setText(message, temporary=True)
            self.setToolTip(tip, temporary=True)
            if processEvents:
                QtWidgets.QApplication.processEvents()
        else:
            self.sigCallProcess.emit(message, tip, processEvents)

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        'Resets the button to its original text and style. Threadsafe.'
        isGuiThread = QtCore.QThread.currentThread() == QtCore.QCoreApplication.instance().thread()
        if isGuiThread:
            self.limitedTime = True
            self.setText()
            self.setToolTip()
            self.setStyleSheet()
        else:
            self.sigReset.emit()

    def startBlink(self, color, message=None, tip='', limitedTime=True):
        if False:
            while True:
                i = 10
        self.setFixedHeight(self.height())
        if message is not None:
            self.setText(message, temporary=True)
        self.setToolTip(tip, temporary=True)
        self.count = 0
        self.indStyle = 'QPushButton {background-color: %s}' % color
        self.limitedTime = limitedTime
        self.borderOn()
        if limitedTime:
            QtCore.QTimer.singleShot(2000, self.setText)
            QtCore.QTimer.singleShot(10000, self.setToolTip)

    def borderOn(self):
        if False:
            for i in range(10):
                print('nop')
        self.setStyleSheet(self.indStyle, temporary=True)
        if self.limitedTime or self.count <= 2:
            QtCore.QTimer.singleShot(100, self.borderOff)

    def borderOff(self):
        if False:
            print('Hello World!')
        self.setStyleSheet()
        self.count += 1
        if self.count >= 2:
            if self.limitedTime:
                return
        QtCore.QTimer.singleShot(30, self.borderOn)

    def setText(self, text=None, temporary=False):
        if False:
            i = 10
            return i + 15
        if text is None:
            text = self.origText
        QtWidgets.QPushButton.setText(self, text)
        if not temporary:
            self.origText = text

    def setToolTip(self, text=None, temporary=False):
        if False:
            print('Hello World!')
        if text is None:
            text = self.origTip
        QtWidgets.QPushButton.setToolTip(self, text)
        if not temporary:
            self.origTip = text

    def setStyleSheet(self, style=None, temporary=False):
        if False:
            while True:
                i = 10
        if style is None:
            style = self.origStyle
        QtWidgets.QPushButton.setStyleSheet(self, style)
        if not temporary:
            self.origStyle = style