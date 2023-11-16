from direct.tkwidgets.AppShell import *
from direct.showbase.TkGlobal import *
from direct.task import Task
from tkinter.simpledialog import askfloat
FRAMES = 0
SECONDS = 1

class BlendAnimPanel(AppShell):
    appname = 'Blend Anim Panel'
    frameWidth = 575
    frameHeight = 450
    usecommandarea = 0
    usestatusarea = 0
    index = 0
    dragMode = False
    blendRatio = 0
    rateList = ['1/24.0', '0.1', '0.5', '1.0', '2.0', '5.0', '10.0']
    enableBlend = False
    currentBlendName = None

    def __init__(self, aNode=None, blendDict={}, parent=None, **kw):
        if False:
            print('Hello World!')
        INITOPT = Pmw.INITOPT
        self.id = 'BlendAnimPanel ' + aNode.getName()
        self.appname = self.id
        self.actorNode = aNode
        self.blendDict = blendDict.copy()
        if len(blendDict) > 0:
            self.blendList = blendDict.keys()
        else:
            self.blendList = []
        optiondefs = (('title', self.appname, None), ('actor', aNode, None), ('animList', [], None), ('blendAnimList', self.blendList, None))
        self.defineoptions(kw, optiondefs)
        self.id = 'Blend AnimPanel ' + aNode.getName()
        self.nodeName = aNode.getName()
        AppShell.__init__(self)
        self.initialiseoptions(BlendAnimPanel)
        self.currTime = 0.0
        self.animNameA = None
        self.animNameB = None
        self.parent.resizable(False, False)

    def createInterface(self):
        if False:
            while True:
                i = 10
        interior = self.interior()
        self.menuBar.destroy()
        actorFrame = Frame(interior)
        name_label = Label(actorFrame, text=self.nodeName, font=('MSSansSerif', 14), relief=SUNKEN, borderwidth=3)
        name_label.pack(side=TOP, expand=False)
        actorFrame.pack(side=TOP, expand=False, fill=X)
        group = Pmw.Group(interior, tag_pyclass=None)
        actorFrame = group.interior()
        group.pack(side=TOP, expand=False, fill=X)
        Label(actorFrame, text='Blended:', font=('MSSansSerif', 10)).pack(side=LEFT)
        self.blendAnimEntry = self.createcomponent('Blended Animation', (), None, Pmw.ComboBox, (actorFrame,), labelpos=W, entry_width=20, selectioncommand=self.setBlendAnim, scrolledlist_items=self['blendAnimList'])
        self.blendAnimEntry.pack(side=LEFT)
        Label(actorFrame, text='   ', font=('MSSansSerif', 10)).pack(side=LEFT)
        button = Button(actorFrame, text='Save', font=('MSSansSerif', 10), width=12, command=self.saveButtonPushed).pack(side=LEFT)
        button = Button(actorFrame, text='Remove', font=('MSSansSerif', 10), width=12, command=self.removeButtonPushed).pack(side=LEFT)
        button = Button(actorFrame, text='Rename', font=('MSSansSerif', 10), width=12, command=self.renameButtonPushed).pack(side=LEFT)
        actorFrame.pack(side=TOP, expand=False, fill=X)
        group = Pmw.Group(interior, tag_pyclass=None)
        actorFrame = group.interior()
        group.pack(side=TOP, expand=False, fill=X)
        Label(actorFrame, text='Animation A:', font=('MSSansSerif', 10)).pack(side=LEFT)
        self['animList'] = self['actor'].getAnimNames()
        self.AnimEntryA = self.createcomponent('AnimationMenuA', (), None, Pmw.ComboBox, (actorFrame,), labelpos=W, entry_width=20, entry_state=DISABLED, selectioncommand=lambda name, a='a': self.setAnimation(name, AB=a), scrolledlist_items=self['animList'])
        self.AnimEntryA.pack(side=LEFT)
        Label(actorFrame, text='   ', font=('MSSansSerif', 10)).pack(side=LEFT)
        Label(actorFrame, text='Animation B:', font=('MSSansSerif', 10)).pack(side=LEFT)
        self['animList'] = self['actor'].getAnimNames()
        self.AnimEntryB = self.createcomponent('AnimationMenuB', (), None, Pmw.ComboBox, (actorFrame,), labelpos=W, entry_width=20, entry_state=DISABLED, selectioncommand=lambda name, a='b': self.setAnimation(name, AB=a), scrolledlist_items=self['animList'])
        self.AnimEntryB.pack(side=LEFT)
        actorFrame.pack(side=TOP, expand=False, fill=X)
        actorFrame = Frame(interior, relief=SUNKEN, bd=1)
        Label(actorFrame, text='Enable Blending:', font=('MSSansSerif', 10)).pack(side=LEFT)
        self.blendVar = IntVar()
        self.blendVar.set(0)
        self.blendButton = self.createcomponent('blendButton', (), None, Checkbutton, (actorFrame,), variable=self.blendVar, command=self.toggleBlend)
        self.blendButton.pack(side=LEFT)
        actorFrame.pack(side=TOP, expand=False, fill=X)
        actorFrame = Frame(interior)
        frameFrame = Frame(actorFrame, relief=SUNKEN, bd=1)
        minRatioLabel = self.createcomponent('minRatioLabel', (), 'sLabel', Label, (frameFrame,), text=0.0)
        minRatioLabel.pack(side=LEFT)
        self.ratioControl = self.createcomponent('ratio', (), None, Scale, (frameFrame,), from_=0.0, to=1.0, resolution=0.01, command=self.setRatio, length=500, orient=HORIZONTAL, showvalue=1)
        self.ratioControl.pack(side=LEFT, expand=1)
        self.ratioControl.set(1.0)
        self.maxRatioLabel = self.createcomponent('maxRatioLabel', (), 'sLabel', Label, (frameFrame,), text=1.0)
        self.maxRatioLabel.pack(side=LEFT)
        frameFrame.pack(side=LEFT, expand=1, fill=X)
        actorFrame.pack(side=TOP, expand=True, fill=X)
        actorFrame = Frame(interior)
        Label(actorFrame, text='Play Rate:', font=('MSSansSerif', 10)).pack(side=LEFT)
        self.playRateEntry = self.createcomponent('playRateMenu', (), None, Pmw.ComboBox, (actorFrame,), labelpos=W, entry_width=20, selectioncommand=self.setPlayRate, scrolledlist_items=self.rateList)
        self.playRateEntry.pack(side=LEFT)
        self.playRateEntry.selectitem('1.0')
        Label(actorFrame, text='   ', font=('MSSansSerif', 10)).pack(side=LEFT)
        Label(actorFrame, text='Loop:', font=('MSSansSerif', 10)).pack(side=LEFT)
        self.loopVar = IntVar()
        self.loopVar.set(0)
        self.loopButton = self.createcomponent('loopButton', (), None, Checkbutton, (actorFrame,), variable=self.loopVar)
        self.loopButton.pack(side=LEFT)
        actorFrame.pack(side=TOP, expand=True, fill=X)
        actorFrame = Frame(interior)
        Label(actorFrame, text='Frame/Second:', font=('MSSansSerif', 10)).pack(side=LEFT)
        self.unitsVar = IntVar()
        self.unitsVar.set(FRAMES)
        self.displayButton = self.createcomponent('displayButton', (), None, Checkbutton, (actorFrame,), command=self.updateDisplay, variable=self.unitsVar)
        self.displayButton.pack(side=LEFT)
        actorFrame.pack(side=TOP, expand=True, fill=X)
        actorFrame = Frame(interior)
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
        actorFrame.pack(side=TOP, expand=True, fill=X)
        actorFrame = Frame(interior)
        ButtomFrame = Frame(actorFrame, relief=SUNKEN, bd=1, borderwidth=5)
        self.toStartButton = self.createcomponent('toStart', (), None, Button, (ButtomFrame,), text='<<', width=8, command=self.resetAllToZero)
        self.toStartButton.pack(side=LEFT, expand=1, fill=X)
        self.playButton = self.createcomponent('playButton', (), None, Button, (ButtomFrame,), text='Play', width=8, command=self.play)
        self.playButton.pack(side=LEFT, expand=1, fill=X)
        self.stopButton = self.createcomponent('stopButton', (), None, Button, (ButtomFrame,), text='Stop', width=8, state=DISABLED, command=self.stop)
        self.stopButton.pack(side=LEFT, expand=1, fill=X)
        self.toEndButton = self.createcomponent('toEnd', (), None, Button, (ButtomFrame,), text='>>', width=8, command=self.resetAllToEnd)
        self.toEndButton.pack(side=LEFT, expand=1, fill=X)
        ButtomFrame.pack(side=TOP, expand=True, fill=X)
        actorFrame.pack(expand=1, fill=BOTH)

    def updateList(self):
        if False:
            while True:
                i = 10
        self['animList'] = self['actor'].getAnimNames()
        animL = self['actor'].getAnimNames()
        self.AnimEntryA.setlist(animL)
        self.AnimEntryB.setlist(animL)

    def play(self):
        if False:
            for i in range(10):
                print('nop')
        self.animNameA = self.AnimEntryA.get()
        self.animNameB = self.AnimEntryB.get()
        if self.animNameA in self['animList'] and self.animNameB in self['animList']:
            self.playButton.config(state=DISABLED)
            self.lastT = globalClock.getFrameTime()
            taskMgr.add(self.playTask, self.id + '_UpdateTask')
            self.stopButton.config(state=NORMAL)
        else:
            print('----Illegal Animaion name!!', self.animNameA + ',  ' + self.animNameB)
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
            return 10
        taskMgr.remove(self.id + '_UpdateTask')
        self.playButton.config(state=NORMAL)
        self.stopButton.config(state=DISABLED)
        return

    def setAnimation(self, animation, AB='a'):
        if False:
            while True:
                i = 10
        print('OK!!!')
        if AB == 'a':
            if self.animNameA != None:
                self['actor'].setControlEffect(self.animNameA, 1.0, 'modelRoot', 'lodRoot')
            self.animNameA = self.AnimEntryA.get()
        else:
            if self.animNameB != None:
                self['actor'].setControlEffect(self.animNameB, 1.0, 'modelRoot', 'lodRoot')
            self.animNameB = self.AnimEntryB.get()
        self.currTime = 0.0
        self.frameControl.set(0)
        self.updateDisplay()
        self.setRatio(self.blendRatio)
        return

    def setPlayRate(self, rate):
        if False:
            for i in range(10):
                print('nop')
        self.animNameA = self.AnimEntryA.get()
        if self.animNameA in self['animList']:
            self['actor'].setPlayRate(eval(rate), self.animNameA)
            self.updateDisplay()
        if self.animNameB in self['animList']:
            self['actor'].setPlayRate(eval(rate), self.animNameB)
            self.updateDisplay()
        return

    def updateDisplay(self):
        if False:
            print('Hello World!')
        if not self.animNameA in self['animList']:
            return
        self.fps = self['actor'].getFrameRate(self.animNameA)
        self.duration = self['actor'].getDuration(self.animNameA)
        self.maxFrame = self['actor'].getNumFrames(self.animNameA) - 1
        if not self.animNameB in self['animList']:
            return
        if self.duration > self['actor'].getDuration(self.animNameB):
            self.duration = self['actor'].getDuration(self.animNameB)
        if self.maxFrame > self['actor'].getNumFrames(self.animNameB) - 1:
            self.maxFrame = self['actor'].getNumFrames(self.animNameB) - 1
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
            i = 10
            return i + 15
        if self.unitsVar.get() == FRAMES:
            self.frameControl.set(time * self.fps)
        else:
            self.frameControl.set(time)
        return

    def goTo(self, frame):
        if False:
            return 10
        if self.animNameA in self['animList'] and self.animNameB in self['animList']:
            frame = float(frame)
            if self.unitsVar.get() == FRAMES:
                frame = frame / self.fps
            if self.dragMode:
                self.currTime = frame
            self['actor'].pose(self.animNameA, min(self.maxFrame, int(frame * self.fps)))
            self['actor'].pose(self.animNameB, min(self.maxFrame, int(frame * self.fps)))
        return

    def onRelease(self, frame):
        if False:
            i = 10
            return i + 15
        self.dragMode = False
        return

    def onPress(self, frame):
        if False:
            for i in range(10):
                print('nop')
        self.dragMode = True
        return

    def resetAllToZero(self):
        if False:
            print('Hello World!')
        self.currTime = 0.0
        self.gotoT(0)
        return

    def resetAllToEnd(self):
        if False:
            for i in range(10):
                print('nop')
        self.currTime = self.maxSeconds
        self.gotoT(self.duration)
        return

    def toggleBlend(self):
        if False:
            while True:
                i = 10
        if self.blendVar.get():
            self.enableBlend = True
            self['actor'].enableBlend()
            self.setRatio(self.blendRatio)
        else:
            self.enableBlend = False
            self['actor'].disableBlend()
        return

    def setRatio(self, ratio):
        if False:
            i = 10
            return i + 15
        self.blendRatio = float(ratio)
        if self.enableBlend:
            if self.animNameA in self['animList']:
                self['actor'].setControlEffect(self.animNameA, self.blendRatio, 'modelRoot', 'lodRoot')
            if self.animNameB in self['animList']:
                self['actor'].setControlEffect(self.animNameB, 1 - self.blendRatio, 'modelRoot', 'lodRoot')
            return

    def setBlendAnim(self, name):
        if False:
            return 10
        if name in self.blendDict:
            self.currentBlendName = name
            animA = self.blendDict[name][0]
            animB = self.blendDict[name][1]
            ratio = self.blendDict[name][2]
            self.AnimEntryA.selectitem(animA)
            self.AnimEntryB.selectitem(animB)
            self.setAnimation(animA, AB='a')
            self.setAnimation(animB, AB='b')
            self.ratioControl.set(ratio)
        return

    def setBlendAnimList(self, dict, select=False):
        if False:
            print('Hello World!')
        self.blendDict.clear()
        del self.blendDict
        self.blendDict = dict.copy()
        print(self.blendDict)
        if len(self.blendDict) > 0:
            self.blendList = self.blendDict.keys()
        else:
            self.blendList = []
        self.blendAnimEntry.setlist(self.blendList)
        if select:
            if len(self.blendList) > 0:
                self.blendAnimEntry.selectitem(self.blendList[0])
                self.setBlendAnim(self.blendList[0])
                self.currentBlendName = self.blendList[0]
            else:
                self.blendAnimEntry.clear()
                self.currentBlendName = None
        return

    def saveButtonPushed(self):
        if False:
            for i in range(10):
                print('nop')
        name = self.blendAnimEntry.get()
        if name == '':
            Pmw.MessageDialog(None, title='Caution!', message_text='You have to give the blending animation a name first!', iconpos='s', defaultbutton='Close')
            return
        elif not self.animNameA in self['animList'] or not self.animNameB in self['animList']:
            Pmw.MessageDialog(None, title='Caution!', message_text='The Animations you have selected are not exist!', iconpos='s', defaultbutton='Close')
            return
        else:
            messenger.send('BAW_saveBlendAnim', [self['actor'].getName(), name, self.animNameA, self.animNameB, self.blendRatio])
            self.currentBlendName = name
        return

    def removeButtonPushed(self):
        if False:
            return 10
        name = self.blendAnimEntry.get()
        messenger.send('BAW_removeBlendAnim', [self['actor'].getName(), name])
        return

    def renameButtonPushed(self):
        if False:
            for i in range(10):
                print('nop')
        oName = self.currentBlendName
        name = self.blendAnimEntry.get()
        if self.currentBlendName == None:
            Pmw.MessageDialog(None, title='Caution!', message_text="You haven't select any blended animation!!", iconpos='s', defaultbutton='Close')
            return
        elif name == '':
            Pmw.MessageDialog(None, title='Caution!', message_text='You have to give the blending animation a name first!', iconpos='s', defaultbutton='Close')
            return
        elif not self.animNameA in self['animList'] or not self.animNameB in self['animList']:
            Pmw.MessageDialog(None, title='Caution!', message_text='The Animations you have selected are not exist!', iconpos='s', defaultbutton='Close')
            return
        else:
            messenger.send('BAW_renameBlendAnim', [self['actor'].getName(), name, oName, self.animNameA, self.animNameB, self.blendRatio])
            self.currentBlendName = name
        return

    def onDestroy(self, event):
        if False:
            return 10
        if taskMgr.hasTaskNamed(self.id + '_UpdateTask'):
            taskMgr.remove(self.id + '_UpdateTask')
        messenger.send('BAW_close', [self.nodeName])
        self.actorNode.setControlEffect(self.animNameA, 1.0, 'modelRoot', 'lodRoot')
        self.actorNode.setControlEffect(self.animNameB, 1.0, 'modelRoot', 'lodRoot')
        self.actorNode.disableBlend()
        '\n        If you have open any thing, please rewrite here!\n        '
        pass