"""Contains the DirectScrolledList class.

See the :ref:`directscrolledlist` page in the programming manual for a more
in-depth explanation and an example of how to use this class.
"""
__all__ = ['DirectScrolledListItem', 'DirectScrolledList']
from panda3d.core import TextNode
from direct.showbase import ShowBaseGlobal
from direct.showbase.MessengerGlobal import messenger
from . import DirectGuiGlobals as DGG
from direct.directnotify import DirectNotifyGlobal
from direct.task.Task import Task
from direct.task.TaskManagerGlobal import taskMgr
from .DirectFrame import DirectFrame
from .DirectButton import DirectButton

class DirectScrolledListItem(DirectButton):
    """
    While you are not required to use a DirectScrolledListItem for a
    DirectScrolledList, doing so takes care of the highlighting and
    unhighlighting of the list items.
    """
    notify = DirectNotifyGlobal.directNotify.newCategory('DirectScrolledListItem')

    def __init__(self, parent=None, **kw):
        if False:
            for i in range(10):
                print('nop')
        assert self.notify.debugStateCall(self)
        self._parent = parent
        if 'command' in kw:
            self.nextCommand = kw.get('command')
            del kw['command']
        if 'extraArgs' in kw:
            self.nextCommandExtraArgs = kw.get('extraArgs')
            del kw['extraArgs']
        optiondefs = (('parent', self._parent, None), ('command', self.select, None))
        self.defineoptions(kw, optiondefs)
        DirectButton.__init__(self)
        self.initialiseoptions(DirectScrolledListItem)

    def select(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.notify.debugStateCall(self)
        self.nextCommand(*self.nextCommandExtraArgs)
        self._parent.selectListItem(self)

class DirectScrolledList(DirectFrame):
    notify = DirectNotifyGlobal.directNotify.newCategory('DirectScrolledList')

    def __init__(self, parent=None, **kw):
        if False:
            for i in range(10):
                print('nop')
        assert self.notify.debugStateCall(self)
        self.index = 0
        self.__forceHeight = None
        " If one were to want a scrolledList that makes and adds its items\n           as needed, simply pass in an items list of strings (type 'str')\n           and when that item is needed, itemMakeFunction will be called\n           with the text, the index, and itemMakeExtraArgs.  If itemMakeFunction\n           is not specified, it will create a DirectFrame with the text."
        if 'items' in kw:
            for item in kw['items']:
                if not isinstance(item, str):
                    break
            else:
                kw['items'] = kw['items'][:]
        self.nextItemID = 10
        optiondefs = (('items', [], None), ('itemsAlign', TextNode.ACenter, DGG.INITOPT), ('itemsWordwrap', None, DGG.INITOPT), ('command', None, None), ('extraArgs', [], None), ('itemMakeFunction', None, None), ('itemMakeExtraArgs', [], None), ('numItemsVisible', 1, self.setNumItemsVisible), ('scrollSpeed', 8, self.setScrollSpeed), ('forceHeight', None, self.setForceHeight), ('incButtonCallback', None, self.setIncButtonCallback), ('decButtonCallback', None, self.setDecButtonCallback))
        self.defineoptions(kw, optiondefs)
        DirectFrame.__init__(self, parent)
        self.incButton = self.createcomponent('incButton', (), None, DirectButton, (self,))
        self.incButton.bind(DGG.B1PRESS, self.__incButtonDown)
        self.incButton.bind(DGG.B1RELEASE, self.__buttonUp)
        self.decButton = self.createcomponent('decButton', (), None, DirectButton, (self,))
        self.decButton.bind(DGG.B1PRESS, self.__decButtonDown)
        self.decButton.bind(DGG.B1RELEASE, self.__buttonUp)
        self.itemFrame = self.createcomponent('itemFrame', (), None, DirectFrame, (self,))
        for item in self['items']:
            if not isinstance(item, str):
                item.reparentTo(self.itemFrame)
        self.initialiseoptions(DirectScrolledList)
        self.recordMaxHeight()
        self.scrollTo(0)

    def setForceHeight(self):
        if False:
            return 10
        assert self.notify.debugStateCall(self)
        self.__forceHeight = self['forceHeight']

    def recordMaxHeight(self):
        if False:
            print('Hello World!')
        assert self.notify.debugStateCall(self)
        if self.__forceHeight is not None:
            self.maxHeight = self.__forceHeight
        else:
            self.maxHeight = 0.0
            for item in self['items']:
                if not isinstance(item, str):
                    self.maxHeight = max(self.maxHeight, item.getHeight())

    def setScrollSpeed(self):
        if False:
            return 10
        assert self.notify.debugStateCall(self)
        self.__scrollSpeed = self['scrollSpeed']
        if self.__scrollSpeed <= 0:
            self.__scrollSpeed = 1

    def setNumItemsVisible(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.notify.debugStateCall(self)
        self.__numItemsVisible = self['numItemsVisible']

    def destroy(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.notify.debugStateCall(self)
        taskMgr.remove(self.taskName('scroll'))
        if hasattr(self, 'currentSelected'):
            del self.currentSelected
        if self.__incButtonCallback:
            self.__incButtonCallback = None
        if self.__decButtonCallback:
            self.__decButtonCallback = None
        self.incButton.destroy()
        self.decButton.destroy()
        DirectFrame.destroy(self)

    def selectListItem(self, item):
        if False:
            return 10
        assert self.notify.debugStateCall(self)
        if hasattr(self, 'currentSelected'):
            self.currentSelected['state'] = DGG.NORMAL
        item['state'] = DGG.DISABLED
        self.currentSelected = item

    def scrollBy(self, delta):
        if False:
            return 10
        assert self.notify.debugStateCall(self)
        return self.scrollTo(self.index + delta)

    def getItemIndexForItemID(self, itemID):
        if False:
            return 10
        assert self.notify.debugStateCall(self)
        if len(self['items']) == 0:
            return 0
        if isinstance(self['items'][0], str):
            self.notify.warning('getItemIndexForItemID: cant find itemID for non-class list items!')
            return 0
        for i in range(len(self['items'])):
            if self['items'][i].itemID == itemID:
                return i
        self.notify.warning('getItemIndexForItemID: item not found!')
        return 0

    def scrollToItemID(self, itemID, centered=0):
        if False:
            print('Hello World!')
        assert self.notify.debugStateCall(self)
        self.scrollTo(self.getItemIndexForItemID(itemID), centered)

    def scrollTo(self, index, centered=0):
        if False:
            for i in range(10):
                print('nop')
        ' scrolls list so selected index is at top, or centered in box'
        assert self.notify.debugStateCall(self)
        try:
            self['numItemsVisible']
        except Exception:
            self.notify.info('crash 27633 fixed!')
            return
        numItemsVisible = self['numItemsVisible']
        numItemsTotal = len(self['items'])
        if centered:
            self.index = index - numItemsVisible // 2
        else:
            self.index = index
        if len(self['items']) <= numItemsVisible:
            self.incButton['state'] = DGG.DISABLED
            self.decButton['state'] = DGG.DISABLED
            self.index = 0
            ret = 0
        elif self.index <= 0:
            self.index = 0
            self.decButton['state'] = DGG.DISABLED
            self.incButton['state'] = DGG.NORMAL
            ret = 0
        elif self.index >= numItemsTotal - numItemsVisible:
            self.index = numItemsTotal - numItemsVisible
            self.incButton['state'] = DGG.DISABLED
            self.decButton['state'] = DGG.NORMAL
            ret = 0
        else:
            if self.incButton['state'] == DGG.DISABLED or self.decButton['state'] == DGG.DISABLED:
                self.__buttonUp(0)
            self.incButton['state'] = DGG.NORMAL
            self.decButton['state'] = DGG.NORMAL
            ret = 1
        for item in self['items']:
            if not isinstance(item, str):
                item.hide()
        upperRange = min(numItemsTotal, numItemsVisible)
        for i in range(self.index, self.index + upperRange):
            item = self['items'][i]
            if isinstance(item, str):
                if self['itemMakeFunction']:
                    item = self['itemMakeFunction'](item, i, self['itemMakeExtraArgs'])
                else:
                    item = DirectFrame(text=item, text_align=self['itemsAlign'], text_wordwrap=self['itemsWordwrap'], relief=None)
                self['items'][i] = item
                item.reparentTo(self.itemFrame)
                self.recordMaxHeight()
            item.show()
            item.setPos(0, 0, -(i - self.index) * self.maxHeight)
        if self['command']:
            self['command'](*self['extraArgs'])
        return ret

    def makeAllItems(self):
        if False:
            while True:
                i = 10
        assert self.notify.debugStateCall(self)
        for i in range(len(self['items'])):
            item = self['items'][i]
            if isinstance(item, str):
                if self['itemMakeFunction']:
                    item = self['itemMakeFunction'](item, i, self['itemMakeExtraArgs'])
                else:
                    item = DirectFrame(text=item, text_align=self['itemsAlign'], text_wordwrap=self['itemsWordwrap'], relief=None)
                self['items'][i] = item
                item.reparentTo(self.itemFrame)
        self.recordMaxHeight()

    def __scrollByTask(self, task):
        if False:
            while True:
                i = 10
        assert self.notify.debugStateCall(self)
        if task.time - task.prevTime < task.delayTime:
            return Task.cont
        else:
            ret = self.scrollBy(task.delta)
            task.prevTime = task.time
            if ret:
                return Task.cont
            else:
                return Task.done

    def __incButtonDown(self, event):
        if False:
            return 10
        assert self.notify.debugStateCall(self)
        task = Task(self.__scrollByTask)
        task.setDelay(1.0 / self.__scrollSpeed)
        task.prevTime = 0.0
        task.delta = 1
        taskName = self.taskName('scroll')
        taskMgr.add(task, taskName)
        self.scrollBy(task.delta)
        messenger.send('wakeup')
        if self.__incButtonCallback:
            self.__incButtonCallback()

    def __decButtonDown(self, event):
        if False:
            return 10
        assert self.notify.debugStateCall(self)
        task = Task(self.__scrollByTask)
        task.setDelay(1.0 / self.__scrollSpeed)
        task.prevTime = 0.0
        task.delta = -1
        taskName = self.taskName('scroll')
        taskMgr.add(task, taskName)
        self.scrollBy(task.delta)
        messenger.send('wakeup')
        if self.__decButtonCallback:
            self.__decButtonCallback()

    def __buttonUp(self, event):
        if False:
            i = 10
            return i + 15
        assert self.notify.debugStateCall(self)
        taskName = self.taskName('scroll')
        taskMgr.remove(taskName)

    def addItem(self, item, refresh=1):
        if False:
            return 10
        '\n        Add this string and extraArg to the list\n        '
        assert self.notify.debugStateCall(self)
        if not isinstance(item, str):
            item.itemID = self.nextItemID
            self.nextItemID += 1
        self['items'].append(item)
        if not isinstance(item, str):
            item.reparentTo(self.itemFrame)
        if refresh:
            self.refresh()
        if not isinstance(item, str):
            return item.itemID

    def removeItem(self, item, refresh=1):
        if False:
            i = 10
            return i + 15
        '\n        Remove this item from the panel\n        '
        assert self.notify.debugStateCall(self)
        if item in self['items']:
            if hasattr(self, 'currentSelected') and self.currentSelected is item:
                del self.currentSelected
            self['items'].remove(item)
            if not isinstance(item, str):
                item.reparentTo(ShowBaseGlobal.hidden)
            self.refresh()
            return 1
        else:
            return 0

    def removeAndDestroyItem(self, item, refresh=1):
        if False:
            for i in range(10):
                print('nop')
        '\n        Remove and destroy this item from the panel.\n        '
        assert self.notify.debugStateCall(self)
        if item in self['items']:
            if hasattr(self, 'currentSelected') and self.currentSelected is item:
                del self.currentSelected
            if hasattr(item, 'destroy') and hasattr(item.destroy, '__call__'):
                item.destroy()
            self['items'].remove(item)
            if not isinstance(item, str):
                item.reparentTo(ShowBaseGlobal.hidden)
            self.refresh()
            return 1
        else:
            return 0

    def removeAllItems(self, refresh=1):
        if False:
            i = 10
            return i + 15
        '\n        Remove this item from the panel\n        Warning 2006_10_19 tested only in the trolley metagame\n        '
        assert self.notify.debugStateCall(self)
        retval = 0
        while len(self['items']) > 0:
            item = self['items'][0]
            if hasattr(self, 'currentSelected') and self.currentSelected is item:
                del self.currentSelected
            self['items'].remove(item)
            if not isinstance(item, str):
                item.removeNode()
            retval = 1
        if refresh:
            self.refresh()
        return retval

    def removeAndDestroyAllItems(self, refresh=1):
        if False:
            print('Hello World!')
        '\n        Remove and destroy all items from the panel.\n        Warning 2006_10_19 tested only in the trolley metagame\n        '
        assert self.notify.debugStateCall(self)
        retval = 0
        while len(self['items']) > 0:
            item = self['items'][0]
            if hasattr(self, 'currentSelected') and self.currentSelected is item:
                del self.currentSelected
            if hasattr(item, 'destroy') and hasattr(item.destroy, '__call__'):
                item.destroy()
            self['items'].remove(item)
            if not isinstance(item, str):
                item.removeNode()
            retval = 1
        if refresh:
            self.refresh()
        return retval

    def refresh(self):
        if False:
            while True:
                i = 10
        '\n        Update the list - useful when adding or deleting items\n        or changing properties that would affect the scrolling\n        '
        assert self.notify.debugStateCall(self)
        self.recordMaxHeight()
        self.scrollTo(self.index)

    def getSelectedIndex(self):
        if False:
            return 10
        assert self.notify.debugStateCall(self)
        return self.index

    def getSelectedText(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.notify.debugStateCall(self)
        if isinstance(self['items'][self.index], str):
            return self['items'][self.index]
        else:
            return self['items'][self.index]['text']

    def setIncButtonCallback(self):
        if False:
            return 10
        assert self.notify.debugStateCall(self)
        self.__incButtonCallback = self['incButtonCallback']

    def setDecButtonCallback(self):
        if False:
            print('Hello World!')
        assert self.notify.debugStateCall(self)
        self.__decButtonCallback = self['decButtonCallback']