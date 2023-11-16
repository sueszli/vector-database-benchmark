import copy
from PyQt5.QtCore import Qt
from urh.signalprocessing.Encoding import Encoding
from urh.signalprocessing.Message import Message
from urh.signalprocessing.ProtocolAnalyzer import ProtocolAnalyzer
from urh.signalprocessing.ProtocolGroup import ProtocolGroup

class ProtocolTreeItem(object):

    def __init__(self, data: ProtocolAnalyzer or ProtocolGroup, parent):
        if False:
            print('Hello World!')
        '\n\n        :param data: ProtocolGroup for Folder or ProtoAnalyzer for ProtoFrame\n        :type parent: ProtocolTreeItem\n        :return:\n        '
        self.__itemData = data
        self.__parentItem = parent
        self.__childItems = data.items if type(data) == ProtocolGroup else []
        self.copy_data = False
        self.__data_copy = None

    @property
    def protocol(self):
        if False:
            while True:
                i = 10
        if isinstance(self.__itemData, ProtocolAnalyzer):
            if self.copy_data:
                if self.__data_copy is None:
                    self.__data_copy = copy.deepcopy(self.__itemData)
                    self.__data_copy.message_types = self.__itemData.message_types
                    nrz = Encoding([''])
                    for (i, message) in enumerate(self.__data_copy.messages):
                        decoded_bits = message.decoded_bits
                        message.decoder = nrz
                        message.plain_bits = decoded_bits
                        message.message_type = self.__itemData.messages[i].message_type
                    self.__data_copy.qt_signals.show_state_changed.connect(self.__itemData.qt_signals.show_state_changed.emit)
                return self.__data_copy
            else:
                return self.__itemData
        else:
            return None

    @property
    def is_group(self):
        if False:
            for i in range(10):
                print('nop')
        return type(self.__itemData) == ProtocolGroup

    @property
    def group(self):
        if False:
            i = 10
            return i + 15
        if type(self.__itemData) == ProtocolGroup:
            return self.__itemData
        else:
            return None

    @property
    def show(self):
        if False:
            print('Hello World!')
        if self.is_group:
            return self.group_check_state
        else:
            return self.protocol.show

    @show.setter
    def show(self, value: bool):
        if False:
            print('Hello World!')
        value = Qt.Checked if value else Qt.Unchecked
        if not self.is_group:
            self.protocol.show = value
            self.protocol.qt_signals.show_state_changed.emit()
        else:
            for child in self.__childItems:
                child.__itemData.show = value
            if self.childCount() > 0:
                self.__childItems[0].__itemData.qt_signals.show_state_changed.emit()

    @property
    def group_check_state(self):
        if False:
            while True:
                i = 10
        if not self.is_group:
            return None
        if self.childCount() == 0:
            return Qt.Unchecked
        if all((child.show for child in self.children)):
            return Qt.Checked
        elif any((child.show for child in self.children)):
            return Qt.PartiallyChecked
        else:
            return Qt.Unchecked

    @property
    def children(self):
        if False:
            i = 10
            return i + 15
        return self.__childItems

    def parent(self):
        if False:
            return 10
        '\n        :rtype: ProtocolTreeItem\n        '
        return self.__parentItem

    def child(self, number):
        if False:
            while True:
                i = 10
        '\n        :type number: int\n        :rtype: ProtocolTreeItem\n        '
        if number < self.childCount():
            return self.__childItems[number]
        else:
            return False

    def childCount(self) -> int:
        if False:
            print('Hello World!')
        return len(self.__childItems)

    def indexInParent(self):
        if False:
            print('Hello World!')
        if self.__parentItem is not None:
            return self.__parentItem.__childItems.index(self)
        return 0

    def columnCount(self) -> int:
        if False:
            i = 10
            return i + 15
        return 1

    def data(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__itemData.name

    def setData(self, value):
        if False:
            while True:
                i = 10
        self.__itemData.name = value
        return True

    def addGroup(self, name='New Group'):
        if False:
            print('Hello World!')
        self.__childItems.append(ProtocolTreeItem(ProtocolGroup(name), self))

    def appendChild(self, child):
        if False:
            for i in range(10):
                print('nop')
        child.setParent(self)
        self.__childItems.append(child)

    def addProtocol(self, proto):
        if False:
            i = 10
            return i + 15
        try:
            assert isinstance(proto, ProtocolAnalyzer)
            self.__childItems.append(ProtocolTreeItem(proto, self))
        except AssertionError:
            return

    def insertChild(self, pos, child):
        if False:
            i = 10
            return i + 15
        self.__childItems.insert(pos, child)

    def removeAtIndex(self, index: int):
        if False:
            for i in range(10):
                print('nop')
        child = self.__childItems[index]
        child.__parentItem = None
        self.__childItems.remove(child)

    def removeProtocol(self, protocol: ProtocolAnalyzer):
        if False:
            while True:
                i = 10
        assert self.is_group
        if protocol is None:
            return False
        for child in self.children:
            if child.protocol == protocol:
                child.setParent(None)
                return True
        return False

    def setParent(self, parent):
        if False:
            print('Hello World!')
        if self.parent() is not None:
            self.parent().__childItems.remove(self)
        self.__parentItem = parent

    def index_of(self, child):
        if False:
            while True:
                i = 10
        return self.__childItems.index(child)

    def swapChildren(self, child1, child2):
        if False:
            for i in range(10):
                print('nop')
        i1 = self.__childItems.index(child1)
        i2 = self.__childItems.index(child2)
        (self.__childItems[i1], self.__childItems[i2]) = (self.__childItems[i2], self.__childItems[i1])

    def bringChildsToFront(self, childs):
        if False:
            i = 10
            return i + 15
        for child in childs:
            self.__childItems.insert(0, self.__childItems.pop(self.__childItems.index(child)))

    def bringChildsToIndex(self, index, childs):
        if False:
            while True:
                i = 10
        for child in reversed(childs):
            self.__childItems.insert(index, self.__childItems.pop(self.__childItems.index(child)))

    def containsChilds(self, childs):
        if False:
            i = 10
            return i + 15
        for child in childs:
            if child not in self.__childItems:
                return False
        return True

    def sortChilds(self):
        if False:
            for i in range(10):
                print('nop')
        self.__childItems.sort()

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        return self.data() < other.data()

    def clearChilds(self):
        if False:
            print('Hello World!')
        self.__childItems[:] = []

    def __str__(self):
        if False:
            return 10
        return str(self.__itemData)