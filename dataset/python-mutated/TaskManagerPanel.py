"""Undocumented Module"""
__all__ = ['TaskManagerPanel', 'TaskManagerWidget']
from direct.tkwidgets.AppShell import AppShell
from direct.showbase.DirectObject import DirectObject
import Pmw
import tkinter as tk
from tkinter.messagebox import askokcancel

class TaskManagerPanel(AppShell):
    appname = 'TaskManager Panel'
    frameWidth = 300
    frameHeight = 400
    usecommandarea = 0
    usestatusarea = 0

    def __init__(self, taskMgr, parent=None, **kw):
        if False:
            i = 10
            return i + 15
        INITOPT = Pmw.INITOPT
        optiondefs = (('title', self.appname, None),)
        self.defineoptions(kw, optiondefs)
        self.taskMgr = taskMgr
        AppShell.__init__(self, parent=parent)
        self.initialiseoptions(TaskManagerPanel)

    def createInterface(self):
        if False:
            print('Hello World!')
        self.taskMgrWidget = TaskManagerWidget(self.interior(), self.taskMgr)

    def onDestroy(self, event):
        if False:
            while True:
                i = 10
        self.taskMgrWidget.onDestroy()

class TaskManagerWidget(DirectObject):
    """
    TaskManagerWidget class: this class contains methods for creating
    a panel to control taskManager tasks.
    """

    def __init__(self, parent, taskMgr):
        if False:
            for i in range(10):
                print('nop')
        '\n        TaskManagerWidget class pops up a control panel to view/delete\n        tasks managed by the taskManager.\n        '
        self.parent = parent
        self.taskMgr = taskMgr
        self.currentTask = None
        self.__taskDict = {}
        self.taskListBox = Pmw.ScrolledListBox(parent, labelpos=tk.NW, label_text='Tasks:', label_font=('MSSansSerif', 10, 'bold'), listbox_takefocus=1, items=[], selectioncommand=self.setCurrentTask)
        self.taskListBox.pack(expand=1, fill=tk.BOTH)
        self._popupMenu = tk.Menu(self.taskListBox.component('listbox'), tearoff=0)
        self._popupMenu.add_command(label='Remove Task', command=self.removeCurrentTask)
        self._popupMenu.add_command(label='Remove Matching Tasks', command=self.removeMatchingTasks)
        controlsFrame = tk.Frame(parent)
        self.removeButton = tk.Button(controlsFrame, text='Remove Task', command=self.removeCurrentTask)
        self.removeButton.grid(row=0, column=0, sticky=tk.EW)
        self.removeMatchingButton = tk.Button(controlsFrame, text='Remove Matching Tasks', command=self.removeMatchingTasks)
        self.removeMatchingButton.grid(row=0, column=1, sticky=tk.EW)
        self.taskMgrVerbose = tk.IntVar()
        self.taskMgrVerbose.set(0)
        self.update = tk.Button(controlsFrame, text='Update', command=self.updateTaskListBox)
        self.update.grid(row=1, column=0, sticky=tk.EW)
        self.dynamicUpdate = tk.Checkbutton(controlsFrame, text='Dynamic Update', variable=self.taskMgrVerbose, command=self.toggleTaskMgrVerbose)
        self.dynamicUpdate.grid(row=1, column=1, sticky=tk.EW)
        controlsFrame.pack(fill=tk.X)
        controlsFrame.grid_columnconfigure(0, weight=1)
        controlsFrame.grid_columnconfigure(1, weight=1)
        self.accept('TaskManager-spawnTask', self.spawnTaskHook)
        self.accept('TaskManager-removeTask', self.removeTaskHook)
        listbox = self.taskListBox.component('listbox')
        listbox.bind('<KeyRelease-Up>', self.setCurrentTask)
        listbox.bind('<KeyRelease-Down>', self.setCurrentTask)
        listbox.bind('<ButtonPress-3>', self.popupMenu)
        listbox.focus_set()
        self.updateTaskListBox()

    def popupMenu(self, event):
        if False:
            while True:
                i = 10
        "\n        listbox = self.taskListBox.component('listbox')\n        index = listbox.nearest(event.y)\n        listbox.selection_clear(0)\n        listbox.activate(index)\n        self.taskListBox.select_set(index)\n        self.setCurrentTask()\n        "
        self._popupMenu.post(event.widget.winfo_pointerx(), event.widget.winfo_pointery())
        return 'break'

    def setCurrentTask(self, event=None):
        if False:
            print('Hello World!')
        if len(self.taskListBox.curselection()) > 0:
            index = int(self.taskListBox.curselection()[0])
            self.currentTask = self.__taskDict[index]
        else:
            self.currentTask = None

    def updateTaskListBox(self):
        if False:
            print('Hello World!')
        taskNames = []
        self.__taskDict = {}
        count = 0
        for task in sorted(self.taskMgr.getTasks(), key=lambda t: t.getName()):
            taskNames.append(task.getName())
            self.__taskDict[count] = task
            count += 1
        if taskNames:
            self.taskListBox.setlist(taskNames)
            self.taskListBox.component('listbox').activate(0)
            self.setCurrentTask()

    def toggleTaskMgrVerbose(self):
        if False:
            for i in range(10):
                print('nop')
        if self.taskMgrVerbose.get():
            self.updateTaskListBox()

    def spawnTaskHook(self, task):
        if False:
            while True:
                i = 10
        if self.taskMgrVerbose.get():
            self.updateTaskListBox()

    def removeTaskHook(self, task):
        if False:
            print('Hello World!')
        if self.taskMgrVerbose.get():
            self.updateTaskListBox()

    def removeCurrentTask(self):
        if False:
            for i in range(10):
                print('nop')
        if self.currentTask:
            name = self.currentTask.name
            ok = 1
            if name == 'dataLoop' or name == 'resetPrevTransform' or name == 'tkLoop' or (name == 'eventManager') or (name == 'igLoop'):
                ok = askokcancel('TaskManagerControls', 'Remove: %s?' % name, parent=self.parent, default='cancel')
            if ok:
                self.taskMgr.remove(self.currentTask)
                self.updateTaskListBox()

    def removeMatchingTasks(self):
        if False:
            return 10
        name = self.taskListBox.getcurselection()[0]
        ok = 1
        if name == 'dataLoop' or name == 'resetPrevTransform' or name == 'tkLoop' or (name == 'eventManager') or (name == 'igLoop'):
            ok = askokcancel('TaskManagerControls', 'Remove tasks named: %s?' % name, parent=self.parent, default='cancel')
        if ok:
            self.taskMgr.remove(name)
            self.updateTaskListBox()

    def onDestroy(self):
        if False:
            while True:
                i = 10
        self.ignore('TaskManager-spawnTask')
        self.ignore('TaskManager-removeTask')