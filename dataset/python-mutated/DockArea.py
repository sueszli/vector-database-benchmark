import weakref
from ..Qt import QtWidgets
from .Container import Container, HContainer, TContainer, VContainer
from .Dock import Dock
from .DockDrop import DockDrop

class DockArea(Container, QtWidgets.QWidget):

    def __init__(self, parent=None, temporary=False, home=None):
        if False:
            print('Hello World!')
        Container.__init__(self, self)
        QtWidgets.QWidget.__init__(self, parent=parent)
        self.dockdrop = DockDrop(self)
        self.dockdrop.removeAllowedArea('center')
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.setLayout(self.layout)
        self.docks = weakref.WeakValueDictionary()
        self.topContainer = None
        self.dockdrop.raiseOverlay()
        self.temporary = temporary
        self.tempAreas = []
        self.home = home

    def type(self):
        if False:
            print('Hello World!')
        return 'top'

    def addDock(self, dock=None, position='bottom', relativeTo=None, **kwds):
        if False:
            for i in range(10):
                print('nop')
        "Adds a dock to this area.\n        \n        ============== =================================================================\n        **Arguments:**\n        dock           The new Dock object to add. If None, then a new Dock will be \n                       created.\n        position       'bottom', 'top', 'left', 'right', 'above', or 'below'\n        relativeTo     If relativeTo is None, then the new Dock is added to fill an \n                       entire edge of the window. If relativeTo is another Dock, then \n                       the new Dock is placed adjacent to it (or in a tabbed \n                       configuration for 'above' and 'below'). \n        ============== =================================================================\n        \n        All extra keyword arguments are passed to Dock.__init__() if *dock* is\n        None.        \n        "
        if dock is None:
            dock = Dock(**kwds)
        if not self.temporary:
            dock.orig_area = self
        if relativeTo is None or relativeTo is self:
            if self.topContainer is None:
                container = self
                neighbor = None
            else:
                container = self.topContainer
                neighbor = None
        else:
            if isinstance(relativeTo, str):
                relativeTo = self.docks[relativeTo]
            container = self.getContainer(relativeTo)
            if container is None:
                raise TypeError('Dock %s is not contained in a DockArea; cannot add another dock relative to it.' % relativeTo)
            neighbor = relativeTo
        neededContainer = {'bottom': 'vertical', 'top': 'vertical', 'left': 'horizontal', 'right': 'horizontal', 'above': 'tab', 'below': 'tab'}[position]
        if neededContainer != container.type() and container.type() == 'tab':
            neighbor = container
            container = container.container()
        if neededContainer != container.type():
            if neighbor is None:
                container = self.addContainer(neededContainer, self.topContainer)
            else:
                container = self.addContainer(neededContainer, neighbor)
        insertPos = {'bottom': 'after', 'top': 'before', 'left': 'before', 'right': 'after', 'above': 'before', 'below': 'after'}[position]
        old = dock.container()
        container.insert(dock, insertPos, neighbor)
        self.docks[dock.name()] = dock
        if old is not None:
            old.apoptose()
        return dock

    def moveDock(self, dock, position, neighbor):
        if False:
            i = 10
            return i + 15
        '\n        Move an existing Dock to a new location. \n        '
        if position in ['left', 'right', 'top', 'bottom'] and neighbor is not None and (neighbor.container() is not None) and (neighbor.container().type() == 'tab'):
            neighbor = neighbor.container()
        self.addDock(dock, position, neighbor)

    def getContainer(self, obj):
        if False:
            i = 10
            return i + 15
        if obj is None:
            return self
        return obj.container()

    def makeContainer(self, typ):
        if False:
            while True:
                i = 10
        if typ == 'vertical':
            new = VContainer(self)
        elif typ == 'horizontal':
            new = HContainer(self)
        elif typ == 'tab':
            new = TContainer(self)
        else:
            raise ValueError("typ must be one of 'vertical', 'horizontal', or 'tab'")
        return new

    def addContainer(self, typ, obj):
        if False:
            while True:
                i = 10
        'Add a new container around obj'
        new = self.makeContainer(typ)
        container = self.getContainer(obj)
        container.insert(new, 'before', obj)
        if obj is not None:
            new.insert(obj)
        self.dockdrop.raiseOverlay()
        return new

    def insert(self, new, pos=None, neighbor=None):
        if False:
            print('Hello World!')
        if self.topContainer is not None:
            self.topContainer.containerChanged(None)
        self.layout.addWidget(new)
        new.containerChanged(self)
        self.topContainer = new
        self.dockdrop.raiseOverlay()

    def count(self):
        if False:
            for i in range(10):
                print('nop')
        if self.topContainer is None:
            return 0
        return 1

    def resizeEvent(self, ev):
        if False:
            i = 10
            return i + 15
        self.dockdrop.resizeOverlay(self.size())

    def addTempArea(self):
        if False:
            i = 10
            return i + 15
        if self.home is None:
            area = DockArea(temporary=True, home=self)
            self.tempAreas.append(area)
            win = TempAreaWindow(area)
            area.win = win
            win.show()
        else:
            area = self.home.addTempArea()
        return area

    def floatDock(self, dock):
        if False:
            i = 10
            return i + 15
        'Removes *dock* from this DockArea and places it in a new window.'
        area = self.addTempArea()
        area.win.resize(dock.size())
        area.moveDock(dock, 'top', None)

    def removeTempArea(self, area):
        if False:
            for i in range(10):
                print('nop')
        self.tempAreas.remove(area)
        area.window().close()

    def saveState(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a serialized (storable) representation of the state of\n        all Docks in this DockArea.'
        if self.topContainer is None:
            main = None
        else:
            main = self.childState(self.topContainer)
        state = {'main': main, 'float': []}
        for a in self.tempAreas:
            geo = a.win.geometry()
            geo = (geo.x(), geo.y(), geo.width(), geo.height())
            state['float'].append((a.saveState(), geo))
        return state

    def childState(self, obj):
        if False:
            i = 10
            return i + 15
        if isinstance(obj, Dock):
            return ('dock', obj.name(), {})
        else:
            childs = []
            for i in range(obj.count()):
                childs.append(self.childState(obj.widget(i)))
            return (obj.type(), childs, obj.saveState())

    def restoreState(self, state, missing='error', extra='bottom'):
        if False:
            i = 10
            return i + 15
        "\n        Restore Dock configuration as generated by saveState.\n        \n        This function does not create any Docks--it will only \n        restore the arrangement of an existing set of Docks.\n        \n        By default, docks that are described in *state* but do not exist\n        in the dock area will cause an exception to be raised. This behavior\n        can be changed by setting *missing* to 'ignore' or 'create'.\n        \n        Extra docks that are in the dockarea but that are not mentioned in\n        *state* will be added to the bottom of the dockarea, unless otherwise\n        specified by the *extra* argument.\n        "
        (containers, docks) = self.findAll()
        oldTemps = self.tempAreas[:]
        if state['main'] is not None:
            self.buildFromState(state['main'], docks, self, missing=missing)
        for s in state['float']:
            a = self.addTempArea()
            a.buildFromState(s[0]['main'], docks, a, missing=missing)
            a.win.setGeometry(*s[1])
            a.apoptose()
        for d in docks.values():
            if extra == 'float':
                a = self.addTempArea()
                a.addDock(d, 'below')
            else:
                self.moveDock(d, extra, None)
        for c in containers:
            c.close()
        for a in oldTemps:
            a.apoptose()

    def buildFromState(self, state, docks, root, depth=0, missing='error'):
        if False:
            while True:
                i = 10
        (typ, contents, state) = state
        if typ == 'dock':
            try:
                obj = docks[contents]
                del docks[contents]
            except KeyError:
                if missing == 'error':
                    raise Exception('Cannot restore dock state; no dock with name "%s"' % contents)
                elif missing == 'create':
                    obj = Dock(name=contents)
                elif missing == 'ignore':
                    return
                else:
                    raise ValueError('"missing" argument must be one of "error", "create", or "ignore".')
        else:
            obj = self.makeContainer(typ)
        root.insert(obj, 'after')
        if typ != 'dock':
            for o in contents:
                self.buildFromState(o, docks, obj, depth + 1, missing=missing)
            obj.apoptose(propagate=False)
            obj.restoreState(state)

    def findAll(self, obj=None, c=None, d=None):
        if False:
            for i in range(10):
                print('nop')
        if obj is None:
            obj = self.topContainer
        if c is None:
            c = []
            d = {}
            for a in self.tempAreas:
                (c1, d1) = a.findAll()
                c.extend(c1)
                d.update(d1)
        if isinstance(obj, Dock):
            d[obj.name()] = obj
        elif obj is not None:
            c.append(obj)
            for i in range(obj.count()):
                o2 = obj.widget(i)
                (c2, d2) = self.findAll(o2)
                c.extend(c2)
                d.update(d2)
        return (c, d)

    def apoptose(self, propagate=True):
        if False:
            while True:
                i = 10
        if self.topContainer is None or self.topContainer.count() == 0:
            self.topContainer = None
            if self.temporary and self.home is not None:
                self.home.removeTempArea(self)

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        docks = self.findAll()[1]
        for dock in docks.values():
            dock.close()

    def dragEnterEvent(self, *args):
        if False:
            i = 10
            return i + 15
        self.dockdrop.dragEnterEvent(*args)

    def dragMoveEvent(self, *args):
        if False:
            print('Hello World!')
        self.dockdrop.dragMoveEvent(*args)

    def dragLeaveEvent(self, *args):
        if False:
            i = 10
            return i + 15
        self.dockdrop.dragLeaveEvent(*args)

    def dropEvent(self, *args):
        if False:
            i = 10
            return i + 15
        self.dockdrop.dropEvent(*args)

    def printState(self, state=None, name='Main'):
        if False:
            while True:
                i = 10
        if state is None:
            state = self.saveState()
        print('=== %s dock area ===' % name)
        if state['main'] is None:
            print('   (empty)')
        else:
            self._printAreaState(state['main'])
        for (i, float) in enumerate(state['float']):
            self.printState(float[0], name='float %d' % i)

    def _printAreaState(self, area, indent=0):
        if False:
            for i in range(10):
                print('nop')
        if area[0] == 'dock':
            print('  ' * indent + area[0] + ' ' + str(area[1:]))
            return
        else:
            print('  ' * indent + area[0])
            for ch in area[1]:
                self._printAreaState(ch, indent + 1)

class TempAreaWindow(QtWidgets.QWidget):

    def __init__(self, area, **kwargs):
        if False:
            print('Hello World!')
        QtWidgets.QWidget.__init__(self, **kwargs)
        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.dockarea = area
        self.layout.addWidget(area)

    def closeEvent(self, *args):
        if False:
            print('Hello World!')
        docks = self.dockarea.findAll()[1]
        for dock in docks.values():
            if hasattr(dock, 'orig_area'):
                dock.orig_area.addDock(dock)
        self.dockarea.clear()
        super().closeEvent(*args)