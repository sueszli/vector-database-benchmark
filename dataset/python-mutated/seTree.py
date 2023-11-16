import os, Pmw
from direct.showbase.DirectObject import DirectObject
from panda3d.core import *
import tkinter
from tkinter import IntVar, Menu, PhotoImage, Label, Frame, Entry
ICONDIR = getModelPath().findFile(Filename('icons')).toOsSpecific()
if not os.path.isdir(ICONDIR):
    raise RuntimeError("can't find DIRECT icon directory (%r)" % ICONDIR)

class TreeNode:

    def __init__(self, canvas, parent, item, menuList=[]):
        if False:
            print('Hello World!')
        self.canvas = canvas
        self.parent = parent
        self.item = item
        self.state = 'collapsed'
        self.selected = 0
        self.children = {}
        self.kidKeys = []
        self.x = self.y = None
        self.iconimages = {}
        self.menuList = menuList
        self.menuVar = IntVar()
        self.menuVar.set(0)
        self._popupMenu = None
        self.image_id = None
        if self.menuList:
            if self.menuList[-1] == 'Separator':
                self.menuList = self.menuList[:-1]
            self._popupMenu = Menu(self.canvas, tearoff=0)
            for i in range(len(self.menuList)):
                item = self.menuList[i]
                if item == 'Separator':
                    self._popupMenu.add_separator()
                else:
                    self._popupMenu.add_radiobutton(label=item, variable=self.menuVar, value=i, indicatoron=0, command=self.popupMenuCommand)

    def destroy(self):
        if False:
            return 10
        for key in self.kidKeys:
            c = self.children[key]
            del self.children[key]
            c.destroy()
        self.parent = None

    def geticonimage(self, name):
        if False:
            i = 10
            return i + 15
        try:
            return self.iconimages[name]
        except KeyError:
            pass
        (file, ext) = os.path.splitext(name)
        ext = ext or '.gif'
        fullname = os.path.join(ICONDIR, file + ext)
        image = PhotoImage(master=self.canvas, file=fullname)
        self.iconimages[name] = image
        return image

    def select(self, event=None):
        if False:
            print('Hello World!')
        if self.selected:
            return
        self.deselectall()
        self.selected = 1
        if self.parent != None:
            if self.parent.state == 'expanded':
                self.canvas.delete(self.image_id)
                self.drawicon()
                self.drawtext()
        self.item.OnSelect(event)

    def deselect(self, event=None):
        if False:
            i = 10
            return i + 15
        if not self.selected:
            return
        self.selected = 0
        if self.parent != None:
            if self.parent.state == 'expanded':
                self.canvas.delete(self.image_id)
                self.drawicon()
                self.drawtext()

    def deselectall(self):
        if False:
            for i in range(10):
                print('nop')
        if self.parent:
            self.parent.deselectall()
        else:
            self.deselecttree()

    def deselecttree(self):
        if False:
            print('Hello World!')
        if self.selected:
            self.deselect()
        for key in self.kidKeys:
            child = self.children[key]
            child.deselecttree()

    def flip(self, event=None):
        if False:
            for i in range(10):
                print('nop')
        if self.state == 'expanded':
            self.collapse()
        else:
            self.expand()
        self.item.OnDoubleClick()
        return 'break'

    def popupMenu(self, event=None):
        if False:
            return 10
        if self._popupMenu:
            self._popupMenu.post(event.widget.winfo_pointerx(), event.widget.winfo_pointery())
            return 'break'

    def popupMenuCommand(self):
        if False:
            i = 10
            return i + 15
        command = self.menuList[self.menuVar.get()]
        self.item.MenuCommand(command)
        if self.parent and command != 'Update Explorer':
            self.parent.update()

    def expand(self, event=None):
        if False:
            i = 10
            return i + 15
        if not self.item.IsExpandable():
            return
        if self.state != 'expanded':
            self.state = 'expanded'
            self.update()
            self.view()

    def collapse(self, event=None):
        if False:
            for i in range(10):
                print('nop')
        if self.state != 'collapsed':
            self.state = 'collapsed'
            self.update()

    def view(self):
        if False:
            i = 10
            return i + 15
        top = self.y - 2
        bottom = self.lastvisiblechild().y + 17
        height = bottom - top
        visible_top = self.canvas.canvasy(0)
        visible_height = self.canvas.winfo_height()
        visible_bottom = self.canvas.canvasy(visible_height)
        if visible_top <= top and bottom <= visible_bottom:
            return
        (x0, y0, x1, y1) = self.canvas._getints(self.canvas['scrollregion'])
        if top >= visible_top and height <= visible_height:
            fraction = top + height - visible_height
        else:
            fraction = top
        fraction = float(fraction) / y1
        self.canvas.yview_moveto(fraction)

    def reveal(self):
        if False:
            return 10
        parent = self.parent
        while parent:
            if parent.state == 'collapsed':
                parent.state = 'expanded'
                parent = parent.parent
            else:
                break
        self.update()
        self.view()

    def lastvisiblechild(self):
        if False:
            i = 10
            return i + 15
        if self.kidKeys and self.state == 'expanded':
            return self.children[self.kidKeys[-1]].lastvisiblechild()
        else:
            return self

    def update(self):
        if False:
            for i in range(10):
                print('nop')
        if self.parent:
            self.parent.update()
        else:
            oldcursor = self.canvas['cursor']
            self.canvas['cursor'] = 'watch'
            self.canvas.update()
            self.canvas.delete(tkinter.ALL)
            self.draw(7, 2)
            (x0, y0, x1, y1) = self.canvas.bbox(tkinter.ALL)
            self.canvas.configure(scrollregion=(0, 0, x1, y1))
            self.canvas['cursor'] = oldcursor

    def draw(self, x, y):
        if False:
            while True:
                i = 10
        (self.x, self.y) = (x, y)
        self.drawicon()
        self.drawtext()
        if self.state != 'expanded':
            return y + 17
        sublist = self.item._GetSubList()
        if not sublist:
            return y + 17
        self.kidKeys = []
        for item in sublist:
            key = item.GetKey()
            if key in self.children:
                child = self.children[key]
            else:
                child = TreeNode(self.canvas, self, item, self.menuList)
            self.children[key] = child
            self.kidKeys.append(key)
        for key in list(self.children.keys()):
            if key not in self.kidKeys:
                del self.children[key]
        cx = x + 20
        cy = y + 17
        cylast = 0
        for key in self.kidKeys:
            child = self.children[key]
            cylast = cy
            self.canvas.create_line(x + 9, cy + 7, cx, cy + 7, fill='gray50')
            cy = child.draw(cx, cy)
            if child.item.IsExpandable():
                if child.state == 'expanded':
                    iconname = 'minusnode'
                    callback = child.collapse
                else:
                    iconname = 'plusnode'
                    callback = child.expand
                image = self.geticonimage(iconname)
                id = self.canvas.create_image(x + 9, cylast + 7, image=image)
                self.canvas.tag_bind(id, '<1>', callback)
                self.canvas.tag_bind(id, '<Double-1>', lambda x: None)
        id = self.canvas.create_line(x + 9, y + 10, x + 9, cylast + 7, fill='gray50')
        self.canvas.tag_lower(id)
        return cy

    def drawicon(self):
        if False:
            print('Hello World!')
        if self.selected:
            imagename = self.item.GetSelectedIconName() or self.item.GetIconName() or 'openfolder'
        else:
            imagename = self.item.GetIconName() or 'folder'
        image = self.geticonimage(imagename)
        id = self.canvas.create_image(self.x, self.y, anchor='nw', image=image)
        self.image_id = id
        self.canvas.tag_bind(id, '<1>', self.select)
        self.canvas.tag_bind(id, '<Double-1>', self.flip)
        self.canvas.tag_bind(id, '<3>', self.popupMenu)

    def drawtext(self, text=None):
        if False:
            while True:
                i = 10
        textx = self.x + 20 - 1
        texty = self.y - 1
        labeltext = self.item.GetLabelText()
        if labeltext:
            id = self.canvas.create_text(textx, texty, anchor='nw', text=labeltext)
            self.canvas.tag_bind(id, '<1>', self.select)
            self.canvas.tag_bind(id, '<Double-1>', self.flip)
            (x0, y0, x1, y1) = self.canvas.bbox(id)
            textx = max(x1, 200) + 10
        if text == None:
            text = self.item.GetText() or '<no text>'
        try:
            self.entry
        except AttributeError:
            pass
        else:
            self.edit_finish()
        try:
            label = self.label
        except AttributeError:
            self.label = Label(self.canvas, text=text, bd=0, padx=2, pady=2)
        if self.selected:
            self.label.configure(fg='white', bg='darkblue')
        else:
            fg = self.item.GetTextFg()
            self.label.configure(fg=fg, bg='white')
        id = self.canvas.create_window(textx, texty, anchor='nw', window=self.label)
        self.label.bind('<1>', self.select_or_edit)
        self.label.bind('<Double-1>', self.flip)
        self.label.bind('<3>', self.popupMenu)
        if text != self.label['text']:
            self.label['text'] = text
        self.text_id = id

    def select_or_edit(self, event=None):
        if False:
            print('Hello World!')
        if self.selected and self.item.IsEditable():
            text = self.item.GetTextForEdit()
            self.label['text'] = text
            self.drawtext(text)
            self.edit(event)
        else:
            self.select(event)

    def edit(self, event=None):
        if False:
            i = 10
            return i + 15
        self.entry = Entry(self.label, bd=0, highlightthickness=1, width=0)
        self.entry.insert(0, self.label['text'])
        self.entry.selection_range(0, tkinter.END)
        self.entry.pack(ipadx=5)
        self.entry.focus_set()
        self.entry.bind('<Return>', self.edit_finish)
        self.entry.bind('<Escape>', self.edit_cancel)

    def edit_finish(self, event=None):
        if False:
            return 10
        try:
            entry = self.entry
            del self.entry
        except AttributeError:
            return
        text = entry.get()
        entry.destroy()
        if text and text != self.item.GetText():
            self.item.SetText(text)
        text = self.item.GetText()
        self.label['text'] = text
        self.drawtext()
        self.canvas.focus_set()

    def edit_cancel(self, event=None):
        if False:
            return 10
        self.drawtext()
        self.canvas.focus_set()

    def find(self, searchKey):
        if False:
            return 10
        if searchKey == self.item.GetKey():
            return self
        sublist = self.item._GetSubList()
        for item in sublist:
            key = item.GetKey()
            if key in self.children:
                child = self.children[key]
            else:
                child = TreeNode(self.canvas, self, item, self.menuList)
                self.children[key] = child
                self.kidKeys.append(key)
            retVal = child.find(searchKey)
            if retVal:
                return retVal
        return None

class TreeItem:
    """Abstract class representing tree items.

    Methods should typically be overridden, otherwise a default action
    is used.

    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        'Constructor.  Do whatever you need to do.'

    def GetText(self):
        if False:
            i = 10
            return i + 15
        'Return text string to display.'

    def GetTextFg(self):
        if False:
            for i in range(10):
                print('nop')
        return 'black'

    def GetLabelText(self):
        if False:
            return 10
        'Return label text string to display in front of text (if any).'

    def IsExpandable(self):
        if False:
            return 10
        'Return whether there are subitems.'
        return 1

    def _GetSubList(self):
        if False:
            return 10
        'Do not override!  Called by TreeNode.'
        if not self.IsExpandable():
            return []
        sublist = self.GetSubList()
        return sublist

    def IsEditable(self):
        if False:
            while True:
                i = 10
        "Return whether the item's text may be edited."

    def SetText(self, text):
        if False:
            while True:
                i = 10
        "Change the item's text (if it is editable)."

    def GetIconName(self):
        if False:
            return 10
        'Return name of icon to be displayed normally.'

    def GetSelectedIconName(self):
        if False:
            for i in range(10):
                print('nop')
        'Return name of icon to be displayed when selected.'

    def GetSubList(self):
        if False:
            for i in range(10):
                print('nop')
        'Return list of items forming sublist.'

    def OnDoubleClick(self):
        if False:
            while True:
                i = 10
        'Called on a double-click on the item.'

    def OnSelect(self):
        if False:
            while True:
                i = 10
        'Called when item selected.'

    def GetTextForEdit(self):
        if False:
            while True:
                i = 10
        'Called before editting the item.'