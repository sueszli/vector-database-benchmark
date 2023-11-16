from ..Qt import QtCore, QtWidgets
from . import VerticalLabel
__all__ = ['CheckTable']

class CheckTable(QtWidgets.QWidget):
    sigStateChanged = QtCore.Signal(object, object, object)

    def __init__(self, columns):
        if False:
            return 10
        QtWidgets.QWidget.__init__(self)
        self.layout = QtWidgets.QGridLayout()
        self.layout.setSpacing(0)
        self.setLayout(self.layout)
        self.headers = []
        self.columns = columns
        col = 1
        for c in columns:
            label = VerticalLabel.VerticalLabel(c, orientation='vertical')
            self.headers.append(label)
            self.layout.addWidget(label, 0, col)
            col += 1
        self.rowNames = []
        self.rowWidgets = []
        self.oldRows = {}

    def updateRows(self, rows):
        if False:
            for i in range(10):
                print('nop')
        for r in self.rowNames[:]:
            if r not in rows:
                self.removeRow(r)
        for r in rows:
            if r not in self.rowNames:
                self.addRow(r)

    def addRow(self, name):
        if False:
            i = 10
            return i + 15
        label = QtWidgets.QLabel(name)
        row = len(self.rowNames) + 1
        self.layout.addWidget(label, row, 0)
        checks = []
        col = 1
        for c in self.columns:
            check = QtWidgets.QCheckBox('')
            check.col = c
            check.row = name
            self.layout.addWidget(check, row, col)
            checks.append(check)
            if name in self.oldRows:
                check.setChecked(self.oldRows[name][col])
            col += 1
            check.stateChanged.connect(self.checkChanged)
        self.rowNames.append(name)
        self.rowWidgets.append([label] + checks)

    def removeRow(self, name):
        if False:
            print('Hello World!')
        row = self.rowNames.index(name)
        self.oldRows[name] = self.saveState()['rows'][row]
        self.rowNames.pop(row)
        for w in self.rowWidgets[row]:
            w.setParent(None)
            if isinstance(w, QtWidgets.QCheckBox):
                w.stateChanged.disconnect(self.checkChanged)
        self.rowWidgets.pop(row)
        for i in range(row, len(self.rowNames)):
            widgets = self.rowWidgets[i]
            for j in range(len(widgets)):
                widgets[j].setParent(None)
                self.layout.addWidget(widgets[j], i + 1, j)

    def checkChanged(self, state):
        if False:
            for i in range(10):
                print('nop')
        check = QtCore.QObject.sender(self)
        self.sigStateChanged.emit(check.row, check.col, state)

    def saveState(self):
        if False:
            i = 10
            return i + 15
        rows = []
        for i in range(len(self.rowNames)):
            row = [self.rowNames[i]] + [c.isChecked() for c in self.rowWidgets[i][1:]]
            rows.append(row)
        return {'cols': self.columns, 'rows': rows}

    def restoreState(self, state):
        if False:
            for i in range(10):
                print('nop')
        rows = [r[0] for r in state['rows']]
        self.updateRows(rows)
        for r in state['rows']:
            rowNum = self.rowNames.index(r[0])
            for i in range(1, len(r)):
                self.rowWidgets[rowNum][i].setChecked(r[i])