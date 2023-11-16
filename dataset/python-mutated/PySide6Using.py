""" This test is using signals and will only work if PySide properly accepts
compiled functions as callables.
"""
from __future__ import print_function
from PySide6.QtCore import QMetaObject, QObject, Signal, Slot

class Communicate(QObject):
    speak = Signal(int)

    def __init__(self, name='', parent=None):
        if False:
            print('Hello World!')
        QObject.__init__(self, parent)
        self.setObjectName(name)

class Speaker(QObject):

    @Slot(int)
    def on_communicator_speak(self, stuff):
        if False:
            print('Hello World!')
        print(stuff)
speaker = Speaker()
someone = Communicate(name='communicator', parent=speaker)
QMetaObject.connectSlotsByName(speaker)
print('The answer is:', end='')
someone.speak.emit(42)
print('Slot should have made output by now.')