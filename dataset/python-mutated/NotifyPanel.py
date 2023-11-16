"""Contains the NotifyPanel class."""
__all__ = ['NotifyPanel']

class NotifyPanel:
    """NotifyPanel class: this class contains methods for creating
    a panel to control direct/panda notify categories."""

    def __init__(self, directNotify, tl=None):
        if False:
            while True:
                i = 10
        '\n        NotifyPanel class pops up a control panel to view/set\n        notify levels for all available DIRECT and PANDA notify categories\n        '
        from direct.showbase.TkGlobal import Pmw
        from tkinter import Toplevel, Frame, Label, Radiobutton, IntVar
        from tkinter import HORIZONTAL, X, W, NW, BOTH, LEFT, RIGHT
        from panda3d.core import NSFatal, NSError, NSWarning, NSInfo, NSDebug, NSSpam
        if tl is None:
            tl = Toplevel()
            tl.title('Notify Controls')
            tl.geometry('300x400')
        self.activeCategory = None
        mainFrame = Frame(tl)
        framePane = Pmw.PanedWidget(mainFrame, orient=HORIZONTAL)
        categoryFrame = framePane.add('categories', size=200)
        severityFrame = framePane.add('severities', size=50)
        categories = self.getPandaCategoriesAsList()
        self.__categories = {}
        categoryNames = []
        for category in categories:
            name = category.getBasename()
            self.__categories[name] = category
            categoryNames.append(name)
        for name in directNotify.getCategories():
            category = directNotify.getCategory(name)
            self.__categories[name] = category
            categoryNames.append(name)
        categoryNames.sort()
        self.categoryList = Pmw.ScrolledListBox(categoryFrame, labelpos=NW, label_text='Categories:', label_font=('MSSansSerif', 10, 'bold'), listbox_takefocus=1, items=categoryNames, selectioncommand=self.setActivePandaCategory)
        self.categoryList.pack(expand=1, fill=BOTH)
        Label(severityFrame, text='Severity:', font=('MSSansSerif', 10, 'bold'), justify=RIGHT, anchor=W).pack(fill=X, padx=5)
        self.severity = IntVar()
        self.severity.set(0)
        self.fatalSeverity = Radiobutton(severityFrame, text='Fatal', justify=LEFT, anchor=W, value=NSFatal, variable=self.severity, command=self.setActiveSeverity)
        self.fatalSeverity.pack(fill=X)
        self.errorSeverity = Radiobutton(severityFrame, text='Error', justify=LEFT, anchor=W, value=NSError, variable=self.severity, command=self.setActiveSeverity)
        self.errorSeverity.pack(fill=X)
        self.warningSeverity = Radiobutton(severityFrame, text='Warning', justify=LEFT, anchor=W, value=NSWarning, variable=self.severity, command=self.setActiveSeverity)
        self.warningSeverity.pack(fill=X)
        self.infoSeverity = Radiobutton(severityFrame, text='Info', justify=LEFT, anchor=W, value=NSInfo, variable=self.severity, command=self.setActiveSeverity)
        self.infoSeverity.pack(fill=X)
        self.debugSeverity = Radiobutton(severityFrame, text='Debug', justify=LEFT, anchor=W, value=NSDebug, variable=self.severity, command=self.setActiveSeverity)
        self.debugSeverity.pack(fill=X)
        self.spamSeverity = Radiobutton(severityFrame, text='Spam', justify=LEFT, anchor=W, value=NSSpam, variable=self.severity, command=self.setActiveSeverity)
        self.spamSeverity.pack(fill=X)
        framePane.pack(expand=1, fill=BOTH)
        mainFrame.pack(expand=1, fill=BOTH)
        listbox = self.categoryList.component('listbox')
        listbox.bind('<KeyRelease-Up>', self.setActivePandaCategory)
        listbox.bind('<KeyRelease-Down>', self.setActivePandaCategory)
        listbox.focus_set()
        listbox.activate(0)
        self.categoryList.select_set(0)
        self.setActivePandaCategory()

    def _getPandaCategories(self, category):
        if False:
            while True:
                i = 10
        categories = [category]
        for i in range(category.getNumChildren()):
            child = category.getChild(i)
            categories.append(self._getPandaCategories(child))
        return categories

    def getPandaCategories(self):
        if False:
            while True:
                i = 10
        from panda3d.core import Notify
        topCategory = Notify.ptr().getTopCategory()
        return self._getPandaCategories(topCategory)

    def _getPandaCategoriesAsList(self, pc, catList):
        if False:
            print('Hello World!')
        for item in pc:
            if isinstance(item, list):
                self._getPandaCategoriesAsList(item, catList)
            else:
                catList.append(item)

    def getPandaCategoriesAsList(self):
        if False:
            i = 10
            return i + 15
        pc = self.getPandaCategories()
        pcList = []
        self._getPandaCategoriesAsList(pc, pcList)
        return pcList[1:]

    def setActivePandaCategory(self, event=None):
        if False:
            return 10
        categoryName = self.categoryList.getcurselection()[0]
        self.activeCategory = self.__categories.get(categoryName, None)
        if self.activeCategory:
            self.severity.set(self.activeCategory.getSeverity())

    def setActiveSeverity(self):
        if False:
            i = 10
            return i + 15
        if self.activeCategory:
            self.activeCategory.setSeverity(self.severity.get())