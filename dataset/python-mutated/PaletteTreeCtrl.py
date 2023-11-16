"""
Defines Palette tree control
"""
import wx
from .ObjectPaletteBase import ObjectBase, ObjectGen

class PaletteTreeCtrl(wx.TreeCtrl):

    def __init__(self, parent, treeStyle, rootName):
        if False:
            for i in range(10):
                print('nop')
        wx.TreeCtrl.__init__(self, parent, style=treeStyle)
        self.rootName = rootName
        self.root = self.AddRoot(self.rootName)
        self.paletteUI = parent
        self.opSortAlpha = 'Sort Alphabetical Order'
        self.opSortOrig = 'Sort Original Order'
        self.opSort = self.opSortOrig
        self.Bind(wx.EVT_TREE_BEGIN_DRAG, self.onBeginDrag)

    def OnCompareItems(self, item1, item2):
        if False:
            i = 10
            return i + 15
        return self.paletteUI.compareItems(item1, item2)

    def SortTreeNodes(self, parent):
        if False:
            while True:
                i = 10
        self.SortChildren(parent)
        (item, cookie) = self.GetFirstChild(parent)
        while item:
            if self.ItemHasChildren(item):
                self.SortTreeNodes(item)
            (item, cookie) = self.GetNextChild(parent, cookie)

    def addTreeNodes(self, parentItem, parentItemName, items, itemKeys):
        if False:
            print('Hello World!')
        roots = []
        rootItems = []
        for key in itemKeys:
            if parentItemName == items[key]:
                roots.append(key)
        for root in roots:
            newItem = self.AppendItem(parentItem, root)
            self.SetItemData(newItem, root)
            rootItems.append(newItem)
            itemKeys.remove(root)
        for rootItem in rootItems:
            self.addTreeNodes(rootItem, self.GetItemText(rootItem), items, itemKeys)

    def traverse(self, parent, itemText):
        if False:
            i = 10
            return i + 15
        if itemText == self.GetItemText(parent):
            return parent
        (item, cookie) = self.GetFirstChild(parent)
        while item:
            if itemText == self.GetItemText(item):
                return item
            if self.ItemHasChildren(item):
                child = self.traverse(item, itemText)
                if child is not None:
                    return child
            (item, cookie) = self.GetNextChild(parent, cookie)
        return None

    def AddGroup(self):
        if False:
            i = 10
            return i + 15
        parent = self.GetSelection()
        if parent is None:
            parent = self.GetRootItem()
        i = 1
        namestr = f'Group{i}'
        found = self.traverse(self.GetRootItem(), namestr)
        while found:
            i = i + 1
            namestr = f'Group{i}'
            found = self.traverse(self.GetRootItem(), namestr)
        newItem = self.AppendItem(parent, namestr)
        itemData = ObjectGen(name=namestr)
        parentName = self.GetItemText(parent)
        if parentName == self.rootName:
            self.paletteUI.palette.add(itemData)
        else:
            self.paletteUI.palette.add(itemData, parentName)
        self.SetItemPyData(newItem, itemData)
        self.Expand(self.GetRootItem())
        self.ScrollTo(newItem)

    def DeleteItem(self, item):
        if False:
            return 10
        itemText = self.GetItemText(item)
        if item and itemText != self.rootName:
            self.Delete(item)
            self.paletteUI.palette.delete(itemText)

    def DeleteSelected(self):
        if False:
            for i in range(10):
                print('nop')
        item = self.GetSelection()
        self.DeleteItem(item)

    def ReParent(self, parent, newParent):
        if False:
            print('Hello World!')
        (item, cookie) = self.GetFirstChild(parent)
        while item:
            itemName = self.GetItemText(item)
            itemData = self.GetItemData(item)
            newItem = self.AppendItem(newParent, itemName)
            self.SetItemPyData(newItem, itemData)
            if self.ItemHasChildren(item):
                self.ReParent(item, newItem)
            (item, cookie) = self.GetNextChild(parent, cookie)

    def ChangeHierarchy(self, itemName, x, y):
        if False:
            i = 10
            return i + 15
        parent = self.GetRootItem()
        item = self.traverse(parent, itemName)
        if item is None:
            return
        (dragToItem, flags) = self.HitTest(wx.Point(x, y))
        if dragToItem.IsOk():
            if dragToItem == item:
                return
            dragToItemName = self.GetItemText(dragToItem)
            if isinstance(self.paletteUI.palette.findItem(dragToItemName), ObjectBase):
                return
            newItem = self.AppendItem(dragToItem, itemName)
            itemObj = self.paletteUI.palette.findItem(itemName)
            if itemObj is not None:
                if dragToItemName == self.rootName:
                    self.paletteUI.palette.add(itemObj)
                else:
                    self.paletteUI.palette.add(itemObj, dragToItemName)
            self.ReParent(item, newItem)
            self.Delete(item)

    def onBeginDrag(self, event):
        if False:
            return 10
        item = event.GetItem()
        if item != self.GetRootItem():
            text = self.GetItemText(item)
            print("Starting drag'n'drop with %s..." % repr(text))
            tdo = wx.TextDataObject(text)
            tds = wx.DropSource(self)
            tds.SetData(tdo)
            tds.DoDragDrop(True)