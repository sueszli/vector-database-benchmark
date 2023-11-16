import os, datetime, time, re
import shutil
from send2trash import send2trash
from PyQt5.QtCore import QThread, pyqtSignal, QMutex
qmut = QMutex()

class deleteThread(QThread):
    delete_proess_signal = pyqtSignal(int)

    def __init__(self, fileList, dirList):
        if False:
            for i in range(10):
                print('nop')
        super(deleteThread, self).__init__()
        self.fileList = fileList
        self.dirList = dirList
        self.fileNum = len(fileList) + len(dirList)
        self.tempNum = 0

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        qmut.lock()
        try:
            for file_path in self.fileList:
                send2trash(file_path)
                self.tempNum = self.tempNum + 1
                proess = self.tempNum / int(self.fileNum) * 100
                self.delete_proess_signal.emit(int(proess))
            for file_path in self.dirList:
                send2trash(file_path)
                self.tempNum = self.tempNum + 1
                proess = self.tempNum / int(self.fileNum) * 100
                self.delete_proess_signal.emit(int(proess))
            qmut.unlock()
            self.exec_()
        except Exception as e:
            print(e)