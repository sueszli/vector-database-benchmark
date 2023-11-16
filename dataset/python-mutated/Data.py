import numpy as np
from ...graphicsItems.LinearRegionItem import LinearRegionItem
from ...Qt import QtCore, QtWidgets
from ...widgets.TreeWidget import TreeWidget
from ..Node import Node
from . import functions
from .common import CtrlNode

class ColumnSelectNode(Node):
    """Select named columns from a record array or MetaArray."""
    nodeName = 'ColumnSelect'

    def __init__(self, name):
        if False:
            while True:
                i = 10
        Node.__init__(self, name, terminals={'In': {'io': 'in'}})
        self.columns = set()
        self.columnList = QtWidgets.QListWidget()
        self.axis = 0
        self.columnList.itemChanged.connect(self.itemChanged)

    def process(self, In, display=True):
        if False:
            print('Hello World!')
        if display:
            self.updateList(In)
        out = {}
        if hasattr(In, 'implements') and In.implements('MetaArray'):
            for c in self.columns:
                out[c] = In[self.axis:c]
        elif isinstance(In, np.ndarray) and In.dtype.fields is not None:
            for c in self.columns:
                out[c] = In[c]
        else:
            self.In.setValueAcceptable(False)
            raise Exception('Input must be MetaArray or ndarray with named fields')
        return out

    def ctrlWidget(self):
        if False:
            print('Hello World!')
        return self.columnList

    def updateList(self, data):
        if False:
            print('Hello World!')
        if hasattr(data, 'implements') and data.implements('MetaArray'):
            cols = data.listColumns()
            for ax in cols:
                if len(cols[ax]) > 0:
                    self.axis = ax
                    cols = set(cols[ax])
                    break
        else:
            cols = list(data.dtype.fields.keys())
        rem = set()
        for c in self.columns:
            if c not in cols:
                self.removeTerminal(c)
                rem.add(c)
        self.columns -= rem
        self.columnList.blockSignals(True)
        self.columnList.clear()
        for c in cols:
            item = QtWidgets.QListWidgetItem(c)
            item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            if c in self.columns:
                item.setCheckState(QtCore.Qt.CheckState.Checked)
            else:
                item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.columnList.addItem(item)
        self.columnList.blockSignals(False)

    def itemChanged(self, item):
        if False:
            return 10
        col = str(item.text())
        if item.checkState() == QtCore.Qt.CheckState.Checked:
            if col not in self.columns:
                self.columns.add(col)
                self.addOutput(col)
        elif col in self.columns:
            self.columns.remove(col)
            self.removeTerminal(col)
        self.update()

    def saveState(self):
        if False:
            return 10
        state = Node.saveState(self)
        state['columns'] = list(self.columns)
        return state

    def restoreState(self, state):
        if False:
            print('Hello World!')
        Node.restoreState(self, state)
        self.columns = set(state.get('columns', []))
        for c in self.columns:
            self.addOutput(c)

class RegionSelectNode(CtrlNode):
    """Returns a slice from a 1-D array. Connect the 'widget' output to a plot to display a region-selection widget."""
    nodeName = 'RegionSelect'
    uiTemplate = [('start', 'spin', {'value': 0, 'step': 0.1}), ('stop', 'spin', {'value': 0.1, 'step': 0.1}), ('display', 'check', {'value': True}), ('movable', 'check', {'value': True})]

    def __init__(self, name):
        if False:
            while True:
                i = 10
        self.items = {}
        CtrlNode.__init__(self, name, terminals={'data': {'io': 'in'}, 'selected': {'io': 'out'}, 'region': {'io': 'out'}, 'widget': {'io': 'out', 'multi': True}})
        self.ctrls['display'].toggled.connect(self.displayToggled)
        self.ctrls['movable'].toggled.connect(self.movableToggled)

    def displayToggled(self, b):
        if False:
            print('Hello World!')
        for item in self.items.values():
            item.setVisible(b)

    def movableToggled(self, b):
        if False:
            print('Hello World!')
        for item in self.items.values():
            item.setMovable(b)

    def process(self, data=None, display=True):
        if False:
            i = 10
            return i + 15
        s = self.stateGroup.state()
        region = [s['start'], s['stop']]
        if display:
            conn = self['widget'].connections()
            for c in conn:
                plot = c.node().getPlot()
                if plot is None:
                    continue
                if c in self.items:
                    item = self.items[c]
                    item.setRegion(region)
                else:
                    item = LinearRegionItem(values=region)
                    self.items[c] = item
                    item.sigRegionChanged.connect(self.rgnChanged)
                    item.setVisible(s['display'])
                    item.setMovable(s['movable'])
        if self['selected'].isConnected():
            if data is None:
                sliced = None
            elif hasattr(data, 'implements') and data.implements('MetaArray'):
                sliced = data[0:s['start']:s['stop']]
            else:
                mask = (data['time'] >= s['start']) * (data['time'] < s['stop'])
                sliced = data[mask]
        else:
            sliced = None
        return {'selected': sliced, 'widget': self.items, 'region': region}

    def rgnChanged(self, item):
        if False:
            for i in range(10):
                print('nop')
        region = item.getRegion()
        self.stateGroup.setState({'start': region[0], 'stop': region[1]})
        self.update()

class TextEdit(QtWidgets.QTextEdit):

    def __init__(self, on_update):
        if False:
            return 10
        super().__init__()
        self.on_update = on_update
        self.lastText = None

    def focusOutEvent(self, ev):
        if False:
            print('Hello World!')
        text = self.toPlainText()
        if text != self.lastText:
            self.lastText = text
            self.on_update()
        super().focusOutEvent(ev)

class EvalNode(Node):
    """Return the output of a string evaluated/executed by the python interpreter.
    The string may be either an expression or a python script, and inputs are accessed as the name of the terminal. 
    For expressions, a single value may be evaluated for a single output, or a dict for multiple outputs.
    For a script, the text will be executed as the body of a function."""
    nodeName = 'PythonEval'

    def __init__(self, name):
        if False:
            while True:
                i = 10
        Node.__init__(self, name, terminals={'input': {'io': 'in', 'renamable': True, 'multiable': True}, 'output': {'io': 'out', 'renamable': True, 'multiable': True}}, allowAddInput=True, allowAddOutput=True)
        self.ui = QtWidgets.QWidget()
        self.layout = QtWidgets.QGridLayout()
        self.text = TextEdit(self.update)
        self.text.setTabStopWidth(30)
        self.text.setPlainText("# Access inputs as args['input_name']\nreturn {'output': None} ## one key per output terminal")
        self.layout.addWidget(self.text, 1, 0, 1, 2)
        self.ui.setLayout(self.layout)

    def ctrlWidget(self):
        if False:
            i = 10
            return i + 15
        return self.ui

    def setCode(self, code):
        if False:
            for i in range(10):
                print('nop')
        ind = []
        lines = code.split('\n')
        for line in lines:
            stripped = line.lstrip()
            if len(stripped) > 0:
                ind.append(len(line) - len(stripped))
        if len(ind) > 0:
            ind = min(ind)
            code = '\n'.join([line[ind:] for line in lines])
        self.text.clear()
        self.text.insertPlainText(code)

    def code(self):
        if False:
            return 10
        return self.text.toPlainText()

    def process(self, display=True, **args):
        if False:
            for i in range(10):
                print('nop')
        l = locals()
        l.update(args)
        try:
            text = self.text.toPlainText().replace('\n', ' ')
            output = eval(text, globals(), l)
        except SyntaxError:
            fn = 'def fn(**args):\n'
            run = '\noutput=fn(**args)\n'
            text = fn + '\n'.join(['    ' + l for l in self.text.toPlainText().split('\n')]) + run
            ldict = locals()
            exec(text, globals(), ldict)
            output = ldict['output']
        except:
            print(f'Error processing node: {self.name()}')
            raise
        return output

    def saveState(self):
        if False:
            i = 10
            return i + 15
        state = Node.saveState(self)
        state['text'] = self.text.toPlainText()
        return state

    def restoreState(self, state):
        if False:
            for i in range(10):
                print('nop')
        Node.restoreState(self, state)
        self.setCode(state['text'])
        self.restoreTerminals(state['terminals'])
        self.update()

class ColumnJoinNode(Node):
    """Concatenates record arrays and/or adds new columns"""
    nodeName = 'ColumnJoin'

    def __init__(self, name):
        if False:
            print('Hello World!')
        Node.__init__(self, name, terminals={'output': {'io': 'out'}})
        self.ui = QtWidgets.QWidget()
        self.layout = QtWidgets.QGridLayout()
        self.ui.setLayout(self.layout)
        self.tree = TreeWidget()
        self.addInBtn = QtWidgets.QPushButton('+ Input')
        self.remInBtn = QtWidgets.QPushButton('- Input')
        self.layout.addWidget(self.tree, 0, 0, 1, 2)
        self.layout.addWidget(self.addInBtn, 1, 0)
        self.layout.addWidget(self.remInBtn, 1, 1)
        self.addInBtn.clicked.connect(self.addInput)
        self.remInBtn.clicked.connect(self.remInput)
        self.tree.sigItemMoved.connect(self.update)

    def ctrlWidget(self):
        if False:
            print('Hello World!')
        return self.ui

    def addInput(self):
        if False:
            for i in range(10):
                print('nop')
        term = Node.addInput(self, 'input', renamable=True, removable=True, multiable=True)
        item = QtWidgets.QTreeWidgetItem([term.name()])
        item.term = term
        term.joinItem = item
        self.tree.addTopLevelItem(item)

    def remInput(self):
        if False:
            i = 10
            return i + 15
        sel = self.tree.currentItem()
        term = sel.term
        term.joinItem = None
        sel.term = None
        self.tree.removeTopLevelItem(sel)
        self.removeTerminal(term)
        self.update()

    def process(self, display=True, **args):
        if False:
            print('Hello World!')
        order = self.order()
        vals = []
        for name in order:
            if name not in args:
                continue
            val = args[name]
            if isinstance(val, np.ndarray) and len(val.dtype) > 0:
                vals.append(val)
            else:
                vals.append((name, None, val))
        return {'output': functions.concatenateColumns(vals)}

    def order(self):
        if False:
            return 10
        return [str(self.tree.topLevelItem(i).text(0)) for i in range(self.tree.topLevelItemCount())]

    def saveState(self):
        if False:
            for i in range(10):
                print('nop')
        state = Node.saveState(self)
        state['order'] = self.order()
        return state

    def restoreState(self, state):
        if False:
            print('Hello World!')
        Node.restoreState(self, state)
        inputs = self.inputs()
        for name in [n for n in state['order'] if n not in inputs]:
            Node.addInput(self, name, renamable=True, removable=True, multiable=True)
        inputs = self.inputs()
        order = [name for name in state['order'] if name in inputs]
        for name in inputs:
            if name not in order:
                order.append(name)
        self.tree.clear()
        for name in order:
            term = self[name]
            item = QtWidgets.QTreeWidgetItem([name])
            item.term = term
            term.joinItem = item
            self.tree.addTopLevelItem(item)

    def terminalRenamed(self, term, oldName):
        if False:
            for i in range(10):
                print('nop')
        Node.terminalRenamed(self, term, oldName)
        item = term.joinItem
        item.setText(0, term.name())
        self.update()

class Mean(CtrlNode):
    """Calculate the mean of an array across an axis.
    """
    nodeName = 'Mean'
    uiTemplate = [('axis', 'intSpin', {'value': 0, 'min': -1, 'max': 1000000})]

    def processData(self, data):
        if False:
            for i in range(10):
                print('nop')
        s = self.stateGroup.state()
        ax = None if s['axis'] == -1 else s['axis']
        return data.mean(axis=ax)

class Max(CtrlNode):
    """Calculate the maximum of an array across an axis.
    """
    nodeName = 'Max'
    uiTemplate = [('axis', 'intSpin', {'value': 0, 'min': -1, 'max': 1000000})]

    def processData(self, data):
        if False:
            for i in range(10):
                print('nop')
        s = self.stateGroup.state()
        ax = None if s['axis'] == -1 else s['axis']
        return data.max(axis=ax)

class Min(CtrlNode):
    """Calculate the minimum of an array across an axis.
    """
    nodeName = 'Min'
    uiTemplate = [('axis', 'intSpin', {'value': 0, 'min': -1, 'max': 1000000})]

    def processData(self, data):
        if False:
            for i in range(10):
                print('nop')
        s = self.stateGroup.state()
        ax = None if s['axis'] == -1 else s['axis']
        return data.min(axis=ax)

class Stdev(CtrlNode):
    """Calculate the standard deviation of an array across an axis.
    """
    nodeName = 'Stdev'
    uiTemplate = [('axis', 'intSpin', {'value': -0, 'min': -1, 'max': 1000000})]

    def processData(self, data):
        if False:
            return 10
        s = self.stateGroup.state()
        ax = None if s['axis'] == -1 else s['axis']
        return data.std(axis=ax)

class Index(CtrlNode):
    """Select an index from an array axis.
    """
    nodeName = 'Index'
    uiTemplate = [('axis', 'intSpin', {'value': 0, 'min': 0, 'max': 1000000}), ('index', 'intSpin', {'value': 0, 'min': 0, 'max': 1000000})]

    def processData(self, data):
        if False:
            while True:
                i = 10
        s = self.stateGroup.state()
        ax = s['axis']
        ind = s['index']
        if ax == 0:
            return data[ind]
        else:
            return data.take(ind, axis=ax)

class Slice(CtrlNode):
    """Select a slice from an array axis.
    """
    nodeName = 'Slice'
    uiTemplate = [('axis', 'intSpin', {'value': 0, 'min': 0, 'max': 1000000.0}), ('start', 'intSpin', {'value': 0, 'min': -1000000.0, 'max': 1000000.0}), ('stop', 'intSpin', {'value': -1, 'min': -1000000.0, 'max': 1000000.0}), ('step', 'intSpin', {'value': 1, 'min': -1000000.0, 'max': 1000000.0})]

    def processData(self, data):
        if False:
            for i in range(10):
                print('nop')
        s = self.stateGroup.state()
        ax = s['axis']
        start = s['start']
        stop = s['stop']
        step = s['step']
        if ax == 0:
            return data[start:stop:step]
        else:
            sl = [slice(None) for i in range(data.ndim)]
            sl[ax] = slice(start, stop, step)
            return data[sl]

class AsType(CtrlNode):
    """Convert an array to a different dtype.
    """
    nodeName = 'AsType'
    uiTemplate = [('dtype', 'combo', {'values': ['float', 'int', 'float32', 'float64', 'float128', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'], 'index': 0})]

    def processData(self, data):
        if False:
            return 10
        s = self.stateGroup.state()
        return data.astype(s['dtype'])