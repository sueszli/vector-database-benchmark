from __future__ import print_function
from PyQt6.QtCore import QCoreApplication, QMetaObject, QObject, QSettings, pyqtSignal, pyqtSlot
app = QCoreApplication([])
app.setOrganizationName('BOGUS_NAME')
app.setOrganizationDomain('bogosity.com')
app.setApplicationName('BOGUS')
print('OK.')

class Communicate(QObject):
    speak = pyqtSignal(int)

    def __init__(self, name='', parent=None):
        if False:
            while True:
                i = 10
        QObject.__init__(self, parent)
        self.setObjectName(name)

class Speaker(QObject):

    @pyqtSlot(int)
    def on_communicator_speak(self, stuff):
        if False:
            while True:
                i = 10
        print(stuff)
speaker = Speaker()
someone = Communicate(name='communicator', parent=speaker)
QMetaObject.connectSlotsByName(speaker)
print('The answer is:', end='')
someone.speak.emit(42)
print('Slot should have made output by now.')