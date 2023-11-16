from direct.tkwidgets.AppShell import *
from direct.showbase.TkGlobal import *
from direct.task import Task
from tkinter.simpledialog import askfloat
FRAMES = 0
SECONDS = 1

class AnimPanel(AppShell):
    appname = 'Anim Panel'
    frameWidth = 575
    frameHeight = 250
    usecommandarea = 0
    usestatusarea = 0
    index = 0
    dragMode = False
    rateList = ['1/24.0', '0.1', '0.5', '1.0', '2.0', '5.0', '10.0']

    def __init__(self, aNode=None, parent=None, **kw):
        if False:
            while True:
                i = 10
        INITOPT = Pmw.INITOPT
        self.id = 'AnimPanel ' + aNode.getName()
        self.appname = self.id
        optiondefs = (('title', self.appname, None), ('actor', aNode, None), ('animList', [], None))
        self.defineoptions(kw, optiondefs)
        self.frameHeight = 300
        self.id = 'AnimPanel ' + aNode.getName()
        self.nodeName = aNode.getName()
        AppShell.__init__(self)
        self.initialiseoptions(AnimPanel)
        self.currTime = 0.0
        self.animName = None
        self.parent.resizable(False, False)

    def createInterface(self):
        if False:
            return 10
        interior = self.interior()
        menuBar = self.menuBar
        menuBar.addmenu('Anim', 'Anim Panel Operations')
        menuBar.addmenuitem('File', 'command', 'Load Animation', label='Load Animation', command=self.loadAnimation)
        menuBar.addmenuitem('Anim', 'command', 'Set actor controls to t = 0.0', label='Jump all to zero', command=self.resetAllToZero)
        menuBar.addmenuitem('Anim', 'command', 'Set Actor controls to end time', label='Jump all to end time', command=self.resetAllToEnd)
        menuBar.addmenuitem('Anim', 'separator')
        menuBar.addmenuitem('Anim', 'command', 'Play Current Animation', label='Play', command=self.play)
        menuBar.addmenuitem('Anim', 'command', 'Stop Current Animation', label='stop', command=self.stop)
        actorFrame = Frame(interior)
        name_label = Label(actorFrame, text=self.nodeName, font=('MSSansSerif', 16), relief=SUNKEN, borderwidth=3)
        name_label.place(x=5, y=5, anchor=NW)
        Label(actorFrame, text='Animation:', font=('MSSansSerif', 12)).place(x=140, y=5, anchor=NW)
        Label(actorFrame, text='Play Rate:', font=('MSSansSerif', 12)).place(x=140, y=35, anchor=NW)
        self['animList'] = self['actor'].getAnimNames()
        self.AnimEntry = self.createcomponent('AnimationMenu', (), None, Pmw.ComboBox, (actorFrame,), labelpos=W, entry_width=20, selectioncommand=self.setAnimation, scrolledlist_items=self['animList'])
        self.AnimEntry.place(x=240, y=10, anchor=NW)
        self.playRateEntry = self.createcomponent('playRateMenu', (), None, Pmw.ComboBox, (actorFrame,), labelpos=W, entry_width=20, selectioncommand=self.setPlayRate, scrolledlist_items=self.rateList)
        self.playRateEntry.place(x=240, y=40, anchor=NW)
        self.playRateEntry.selectitem('1.0')
        Label(actorFrame, text='Loop:', font=('MSSansSerif', 12)).place(x=420, y=5, anchor=NW)
        self.loopVar = IntVar()
        self.loopVar.set(0)
        self.loopButton = self.createcomponent('loopButton', (), None, Checkbutton, (actorFrame,), variable=self.loopVar)
        self.loopButton.place(x=470, y=7, anchor=NW)
        Label(actorFrame, text='Frame/Second:', font=('MSSansSerif', 11)).place(x=5, y=75, anchor=NW)
        self.unitsVar = IntVar()
        self.unitsVar.set(FRAMES)
        self.displayButton = self.createcomponent('displayButton', (), None, Checkbutton, (actorFrame,), command=self.updateDisplay, variable=self.unitsVar)
        self.displayButton.place(x=120, y=77, anchor=NW)
        frameFrame = Frame(actorFrame, relief=SUNKEN, bd=1)
        self.minLabel = self.createcomponent('minLabel', (), 'sLabel', Label, (frameFrame,), text=0)
        self.minLabel.pack(side=LEFT)
        self.frameControl = self.createcomponent('scale', (), None, Scale, (frameFrame,), from_=0, to=24, resolution=1.0, command=self.goTo, length=500, orient=HORIZONTAL, showvalue=1)
        self.frameControl.pack(side=LEFT, expand=1)
        self.frameControl.bind('<Button-1>', self.onPress)
        self.frameControl.bind('<ButtonRelease-1>', self.onRelease)
        self.maxLabel = self.createcomponent('maxLabel', (), 'sLabel', Label, (frameFrame,), text=24)
        self.maxLabel.pack(side=LEFT)
        frameFrame.pack(side=LEFT, expand=1, fill=X)
        ButtomFrame = Frame(actorFrame, relief=SUNKEN, bd=1, borderwidth=5)
        self.toStartButton = self.createcomponent('toStart', (), None, Button, (ButtomFrame,), text='<<', width=8, command=self.resetAllToZero)
        self.toStartButton.pack(side=LEFT, expand=1, fill=X)
        self.playButton = self.createcomponent('playButton', (), None, Button, (ButtomFrame,), text='Play', width=8, command=self.play)
        self.playButton.pack(side=LEFT, expand=1, fill=X)
        self.stopButton = self.createcomponent('stopButton', (), None, Button, (ButtomFrame,), text='Stop', width=8, state=DISABLED, command=self.stop)
        self.stopButton.pack(side=LEFT, expand=1, fill=X)
        self.toEndButton = self.createcomponent('toEnd', (), None, Button, (ButtomFrame,), text='>>', width=8, command=self.resetAllToEnd)
        self.toEndButton.pack(side=LEFT, expand=1, fill=X)
        ButtomFrame.place(anchor=NW, x=5, y=165)
        self.removeButton = self.createcomponent('Remove Animation', (), None, Button, (actorFrame,), text='Remove This Animation', width=20, command=self.removeAnim)
        self.removeButton.place(anchor=NW, x=5, y=220)
        self.loadButton = self.createcomponent('Load Animation', (), None, Button, (actorFrame,), text='Load New Animation', width=20, command=self.loadAnimation)
        self.loadButton.place(anchor=NW, x=180, y=220)
        actorFrame.pack(expand=1, fill=BOTH)

    def updateList(self):
        if False:
            print('Hello World!')
        self.ignore('DataH_loadFinish' + self.nodeName)
        del self.loaderWindow
        self['animList'] = self['actor'].getAnimNames()
        animL = self['actor'].getAnimNames()
        self.AnimEntry.setlist(animL)

    def removeAnim(self):
        if False:
            while True:
                i = 10
        name = self.AnimEntry.get()
        if taskMgr.hasTaskNamed(self.id + '_UpdateTask'):
            self.stop()
        self.accept('DataH_removeAnimFinish' + self.nodeName, self.afterRemove)
        messenger.send('AW_removeAnim', [self['actor'], name])
        return

    def afterRemove(self):
        if False:
            return 10
        self.ignore('DataH_removeAnimFinish' + self.nodeName)
        self['animList'] = self['actor'].getAnimNames()
        animL = self['actor'].getAnimNames()
        self.AnimEntry.setlist(animL)
        print('-----', animL)
        return

    def loadAnimation(self):
        if False:
            i = 10
            return i + 15
        self.loaderWindow = LoadAnimPanel(aNode=self['actor'])
        self.accept('DataH_loadFinish' + self.nodeName, self.updateList)
        return

    def play(self):
        if False:
            i = 10
            return i + 15
        self.animName = self.AnimEntry.get()
        if self.animName in self['animList']:
            animName = self.AnimEntry.get()
            self.playButton.config(state=DISABLED)
            self.lastT = globalClock.getFrameTime()
            taskMgr.add(self.playTask, self.id + '_UpdateTask')
            self.stopButton.config(state=NORMAL)
        else:
            print('----Illegal Animaion name!!', self.animName)
        return

    def playTask(self, task):
        if False:
            for i in range(10):
                print('nop')
        fLoop = self.loopVar.get()
        currT = globalClock.getFrameTime()
        deltaT = currT - self.lastT
        self.lastT = currT
        if self.dragMode:
            return Task.cont
        self.currTime = self.currTime + deltaT
        if self.currTime > self.maxSeconds:
            if fLoop:
                self.currTime = self.currTime % self.duration
                self.gotoT(self.currTime)
            else:
                self.currTime = 0.0
                self.gotoT(0.0)
                self.playButton.config(state=NORMAL)
                self.stopButton.config(state=DISABLED)
                return Task.done
        else:
            self.gotoT(self.currTime)
        return Task.cont

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        taskMgr.remove(self.id + '_UpdateTask')
        self.playButton.config(state=NORMAL)
        self.stopButton.config(state=DISABLED)
        return

    def setAnimation(self, animation):
        if False:
            print('Hello World!')
        self.animName = self.AnimEntry.get()
        playRate = '%0.1f' % self['actor'].getPlayRate(self.animName)
        if playRate not in self.rateList:

            def strCmp(a, b):
                if False:
                    return 10
                return cmp(eval(a), eval(b))
            self.rateList.append(playRate)
            self.rateList.sort(strCmp)
            self.playRateEntry.reset(self.rateList)
            self.playRateEntry.selectitem(playRate)
        self.currTime = 0.0
        self.frameControl.set(0)
        self.updateDisplay()
        return

    def setPlayRate(self, rate):
        if False:
            print('Hello World!')
        self.animName = self.AnimEntry.get()
        if self.animName in self['animList']:
            self['actor'].setPlayRate(eval(rate), self.animName)
            self.updateDisplay()
        return

    def updateDisplay(self):
        if False:
            while True:
                i = 10
        self.fps = self['actor'].getFrameRate(self.animName)
        self.duration = self['actor'].getDuration(self.animName)
        self.maxFrame = self['actor'].getNumFrames(self.animName) - 1
        self.maxSeconds = self.duration
        if self.unitsVar.get() == FRAMES:
            fromFrame = 0
            toFrame = self.maxFrame
            self.minLabel['text'] = fromFrame
            self.maxLabel['text'] = toFrame
            self.frameControl.configure(from_=fromFrame, to=toFrame, resolution=1.0)
        else:
            self.minLabel['text'] = '0.0'
            self.maxLabel['text'] = '%.2f' % self.duration
            self.frameControl.configure(from_=0.0, to=self.duration, resolution=0.01)

    def gotoT(self, time):
        if False:
            return 10
        if self.unitsVar.get() == FRAMES:
            self.frameControl.set(time * self.fps)
        else:
            self.frameControl.set(time)
        return

    def goTo(self, frame):
        if False:
            print('Hello World!')
        if self.animName in self['animList']:
            frame = float(frame)
            if self.unitsVar.get() == FRAMES:
                frame = frame / self.fps
            if self.dragMode:
                self.currTime = frame
            self['actor'].pose(self.animName, min(self.maxFrame, int(frame * self.fps)))
        return

    def onRelease(self, frame):
        if False:
            while True:
                i = 10
        self.dragMode = False
        return

    def onPress(self, frame):
        if False:
            print('Hello World!')
        self.dragMode = True
        return

    def resetAllToZero(self):
        if False:
            return 10
        self.currTime = 0.0
        self.gotoT(0)
        return

    def resetAllToEnd(self):
        if False:
            print('Hello World!')
        self.currTime = self.maxSeconds
        self.gotoT(self.duration)
        return

    def onDestroy(self, event):
        if False:
            i = 10
            return i + 15
        if taskMgr.hasTaskNamed(self.id + '_UpdateTask'):
            taskMgr.remove(self.id + '_UpdateTask')
        self.ignore('DataH_loadFinish')
        messenger.send('AW_close', [self.nodeName])
        '\n        If you have open any thing, please rewrite here!\n        '
        pass

class LoadAnimPanel(AppShell):
    appname = 'Load Animation'
    frameWidth = 575
    frameHeight = 200
    usecommandarea = 0
    usestatusarea = 0
    index = 0

    def __init__(self, aNode=None, parent=None, **kw):
        if False:
            print('Hello World!')
        INITOPT = Pmw.INITOPT
        self.id = 'Load Animation ' + aNode.getName()
        self.appname = self.id
        self.animDic = {}
        self.animList = []
        optiondefs = (('title', self.appname, None),)
        self.defineoptions(kw, optiondefs)
        self.frameHeight = 300
        self.nodeName = aNode.getName()
        self.Actor = aNode
        AppShell.__init__(self)
        self.initialiseoptions(LoadAnimPanel)

    def createInterface(self):
        if False:
            for i in range(10):
                print('nop')
        self.menuBar.destroy()
        interior = self.interior()
        mainFrame = Frame(interior)
        self.inputZone = Pmw.Group(mainFrame, tag_text='File Setting')
        self.inputZone.pack(fill='both', expand=1)
        settingFrame = self.inputZone.interior()
        Label(settingFrame, text='Anim Name').place(anchor=NW, x=60, y=5)
        Label(settingFrame, text='File Path').place(anchor=NW, x=205, y=5)
        self.AnimName_1 = self.createcomponent('Anim Name List', (), None, Pmw.ComboBox, (settingFrame,), label_text='Anim   :', labelpos=W, entry_width=10, selectioncommand=self.selectAnim, scrolledlist_items=self.animList)
        self.AnimFile_1 = Pmw.EntryField(settingFrame, value='')
        self.AnimFile_1.component('entry').config(width=20)
        self.AnimName_1.place(anchor=NW, x=10, y=25)
        self.AnimFile_1.place(anchor=NW, x=140, y=25)
        self.Browse_1 = self.createcomponent('File Browser1', (), None, Button, (mainFrame,), text='Browse...', command=self.Browse_1)
        self.Browse_1.place(anchor=NW, x=270, y=38)
        self.addIntoButton = self.createcomponent('Load Add', (), None, Button, (mainFrame,), text='Add to Load', command=self.addIntoList)
        self.addIntoButton.place(anchor=NW, x=345, y=38)
        att_label = Label(mainFrame, font=('MSSansSerif', 10), text="Attention! Animations won't be loaded in before you press the 'OK' button below!")
        att_label.place(anchor=NW, x=10, y=80)
        self.button_ok = Button(mainFrame, text='OK', command=self.ok_press, width=10)
        self.button_ok.pack(fill=BOTH, expand=0, side=RIGHT)
        mainFrame.pack(expand=1, fill=BOTH)

    def onDestroy(self, event):
        if False:
            i = 10
            return i + 15
        messenger.send('AWL_close', [self.nodeName])
        '\n        If you have open any thing, please rewrite here!\n        '
        pass

    def selectAnim(self, name):
        if False:
            for i in range(10):
                print('nop')
        if name in self.animDic:
            self.AnimFile_1.setvalue = self.animDic[name]
        return

    def Browse_1(self):
        if False:
            while True:
                i = 10
        AnimFilename = askopenfilename(defaultextension='.egg', filetypes=(('Egg Files', '*.egg'), ('Bam Files', '*.bam'), ('All files', '*')), initialdir='.', title='File Path for Anim 1', parent=self.parent)
        if AnimFilename:
            self.AnimFile_1.setvalue(AnimFilename)
        return

    def addIntoList(self):
        if False:
            print('Hello World!')
        name = self.AnimName_1.get()
        self.animDic[name] = Filename.fromOsSpecific(self.AnimFile_1.getvalue()).getFullpath()
        if name in self.animList:
            pass
        else:
            self.animList.append(name)
        self.AnimName_1.setlist(self.animList)
        print(self.animDic)
        return

    def ok_press(self):
        if False:
            print('Hello World!')
        messenger.send('AW_AnimationLoad', [self.Actor, self.animDic])
        self.quit()
        return