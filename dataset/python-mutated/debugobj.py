from reprlib import Repr
from idlelib.tree import TreeItem, TreeNode, ScrolledCanvas
myrepr = Repr()
myrepr.maxstring = 100
myrepr.maxother = 100

class ObjectTreeItem(TreeItem):

    def __init__(self, labeltext, object, setfunction=None):
        if False:
            i = 10
            return i + 15
        self.labeltext = labeltext
        self.object = object
        self.setfunction = setfunction

    def GetLabelText(self):
        if False:
            while True:
                i = 10
        return self.labeltext

    def GetText(self):
        if False:
            for i in range(10):
                print('nop')
        return myrepr.repr(self.object)

    def GetIconName(self):
        if False:
            i = 10
            return i + 15
        if not self.IsExpandable():
            return 'python'

    def IsEditable(self):
        if False:
            while True:
                i = 10
        return self.setfunction is not None

    def SetText(self, text):
        if False:
            while True:
                i = 10
        try:
            value = eval(text)
            self.setfunction(value)
        except:
            pass
        else:
            self.object = value

    def IsExpandable(self):
        if False:
            i = 10
            return i + 15
        return not not dir(self.object)

    def GetSubList(self):
        if False:
            i = 10
            return i + 15
        keys = dir(self.object)
        sublist = []
        for key in keys:
            try:
                value = getattr(self.object, key)
            except AttributeError:
                continue
            item = make_objecttreeitem(str(key) + ' =', value, lambda value, key=key, object=self.object: setattr(object, key, value))
            sublist.append(item)
        return sublist

class ClassTreeItem(ObjectTreeItem):

    def IsExpandable(self):
        if False:
            print('Hello World!')
        return True

    def GetSubList(self):
        if False:
            return 10
        sublist = ObjectTreeItem.GetSubList(self)
        if len(self.object.__bases__) == 1:
            item = make_objecttreeitem('__bases__[0] =', self.object.__bases__[0])
        else:
            item = make_objecttreeitem('__bases__ =', self.object.__bases__)
        sublist.insert(0, item)
        return sublist

class AtomicObjectTreeItem(ObjectTreeItem):

    def IsExpandable(self):
        if False:
            print('Hello World!')
        return False

class SequenceTreeItem(ObjectTreeItem):

    def IsExpandable(self):
        if False:
            print('Hello World!')
        return len(self.object) > 0

    def keys(self):
        if False:
            print('Hello World!')
        return range(len(self.object))

    def GetSubList(self):
        if False:
            while True:
                i = 10
        sublist = []
        for key in self.keys():
            try:
                value = self.object[key]
            except KeyError:
                continue

            def setfunction(value, key=key, object=self.object):
                if False:
                    print('Hello World!')
                object[key] = value
            item = make_objecttreeitem('%r:' % (key,), value, setfunction)
            sublist.append(item)
        return sublist

class DictTreeItem(SequenceTreeItem):

    def keys(self):
        if False:
            return 10
        keys = list(self.object.keys())
        try:
            keys.sort()
        except:
            pass
        return keys
dispatch = {int: AtomicObjectTreeItem, float: AtomicObjectTreeItem, str: AtomicObjectTreeItem, tuple: SequenceTreeItem, list: SequenceTreeItem, dict: DictTreeItem, type: ClassTreeItem}

def make_objecttreeitem(labeltext, object, setfunction=None):
    if False:
        for i in range(10):
            print('nop')
    t = type(object)
    if t in dispatch:
        c = dispatch[t]
    else:
        c = ObjectTreeItem
    return c(labeltext, object, setfunction)

def _object_browser(parent):
    if False:
        return 10
    import sys
    from tkinter import Toplevel
    top = Toplevel(parent)
    top.title('Test debug object browser')
    (x, y) = map(int, parent.geometry().split('+')[1:])
    top.geometry('+%d+%d' % (x + 100, y + 175))
    top.configure(bd=0, bg='yellow')
    top.focus_set()
    sc = ScrolledCanvas(top, bg='white', highlightthickness=0, takefocus=1)
    sc.frame.pack(expand=1, fill='both')
    item = make_objecttreeitem('sys', sys)
    node = TreeNode(sc.canvas, None, item)
    node.update()
if __name__ == '__main__':
    from unittest import main
    main('idlelib.idle_test.test_debugobj', verbosity=2, exit=False)
    from idlelib.idle_test.htest import run
    run(_object_browser)