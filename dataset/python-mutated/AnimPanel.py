"""DIRECT Animation Control Panel"""
__all__ = ['AnimPanel', 'ActorControl']
from panda3d.core import Filename, getModelPath, ClockObject
from direct.tkwidgets.AppShell import AppShell
from direct.task import Task
from direct.task.TaskManagerGlobal import taskMgr
from tkinter.simpledialog import askfloat
from tkinter.filedialog import askopenfilename
import Pmw
import os
import tkinter as tk
FRAMES = 0
SECONDS = 1

class AnimPanel(AppShell):
    appname = 'Anim Panel'
    frameWidth = 675
    frameHeight = 250
    usecommandarea = 0
    usestatusarea = 0
    index = 0

    def __init__(self, aList=[], parent=None, session=None, **kw):
        if False:
            while True:
                i = 10
        INITOPT = Pmw.INITOPT
        if isinstance(aList, (list, tuple)):
            kw['actorList'] = aList
        else:
            kw['actorList'] = [aList]
        optiondefs = (('title', self.appname, None), ('actorList', [], None), ('Actor_label_width', 12, None))
        self.defineoptions(kw, optiondefs)
        self.session = session
        self.frameHeight = 60 + 50 * len(self['actorList'])
        self.playList = []
        self.id = 'AnimPanel_%d' % AnimPanel.index
        AnimPanel.index += 1
        self.actorControlIndex = 0
        AppShell.__init__(self)
        self.initialiseoptions(AnimPanel)
        self.destroyCallBack = None

    def createInterface(self):
        if False:
            for i in range(10):
                print('nop')
        interior = self.interior()
        menuBar = self.menuBar
        menuBar.addmenu('AnimPanel', 'Anim Panel Operations')
        menuBar.addcascademenu('AnimPanel', 'Control Status', 'Enable/disable actor control panels')
        menuBar.addmenuitem('Control Status', 'command', 'Enable all actor controls', label='Enable all', command=self.enableActorControls)
        menuBar.addmenuitem('Control Status', 'command', 'Disable all actor controls', label='Disable all', command=self.disableActorControls)
        menuBar.addcascademenu('AnimPanel', 'Display Units', 'Select display units')
        menuBar.addmenuitem('Display Units', 'command', 'Display frame counts', label='Frame count', command=self.displayFrameCounts)
        menuBar.addmenuitem('Display Units', 'command', 'Display seconds', label='Seconds', command=self.displaySeconds)
        menuBar.addmenuitem('AnimPanel', 'command', 'Set actor controls to t = 0.0', label='Jump all to zero', command=self.resetAllToZero)
        menuBar.addmenuitem('AnimPanel', 'command', 'Set Actor controls to end time', label='Jump all to end time', command=self.resetAllToEnd)
        self.fToggleAll = 1
        b = self.createcomponent('toggleEnableButton', (), None, tk.Button, (self.menuFrame,), text='Toggle Enable', command=self.toggleAllControls)
        b.pack(side=tk.RIGHT, expand=0)
        b = self.createcomponent('showSecondsButton', (), None, tk.Button, (self.menuFrame,), text='Show Seconds', command=self.displaySeconds)
        b.pack(side=tk.RIGHT, expand=0)
        b = self.createcomponent('showFramesButton', (), None, tk.Button, (self.menuFrame,), text='Show Frames', command=self.displayFrameCounts)
        b.pack(side=tk.RIGHT, expand=0)
        self.actorFrame = None
        self.createActorControls()
        controlFrame = tk.Frame(interior)
        self.toStartButton = self.createcomponent('toStart', (), None, tk.Button, (controlFrame,), text='<<', width=4, command=self.resetAllToZero)
        self.toStartButton.pack(side=tk.LEFT, expand=1, fill=tk.X)
        self.toPreviousFrameButton = self.createcomponent('toPreviousFrame', (), None, tk.Button, (controlFrame,), text='<', width=4, command=self.previousFrame)
        self.toPreviousFrameButton.pack(side=tk.LEFT, expand=1, fill=tk.X)
        self.playButton = self.createcomponent('playButton', (), None, tk.Button, (controlFrame,), text='Play', width=8, command=self.playActorControls)
        self.playButton.pack(side=tk.LEFT, expand=1, fill=tk.X)
        self.stopButton = self.createcomponent('stopButton', (), None, tk.Button, (controlFrame,), text='Stop', width=8, command=self.stopActorControls)
        self.stopButton.pack(side=tk.LEFT, expand=1, fill=tk.X)
        self.toNextFrameButton = self.createcomponent('toNextFrame', (), None, tk.Button, (controlFrame,), text='>', width=4, command=self.nextFrame)
        self.toNextFrameButton.pack(side=tk.LEFT, expand=1, fill=tk.X)
        self.toEndButton = self.createcomponent('toEnd', (), None, tk.Button, (controlFrame,), text='>>', width=4, command=self.resetAllToEnd)
        self.toEndButton.pack(side=tk.LEFT, expand=1, fill=tk.X)
        self.loopVar = tk.IntVar()
        self.loopVar.set(0)
        self.loopButton = self.createcomponent('loopButton', (), None, tk.Checkbutton, (controlFrame,), text='Loop', width=8, variable=self.loopVar)
        self.loopButton.pack(side=tk.LEFT, expand=1, fill=tk.X)
        if self.session:
            menuBar.addmenuitem('File', 'command', 'Set currently selected group of objects as actors to animate.', label='Set Actors', command=self.setActors)
            menuBar.addmenuitem('File', 'command', 'Load animation file', label='Load Anim', command=self.loadAnim)
        controlFrame.pack(fill=tk.X)

    def createActorControls(self):
        if False:
            while True:
                i = 10
        self.actorFrame = tk.Frame(self.interior())
        self.actorControlList = []
        for actor in self['actorList']:
            anims = actor.getAnimNames()
            print('actor animnames: %s' % anims)
            topAnims = []
            if 'neutral' in anims:
                i = anims.index('neutral')
                del anims[i]
                topAnims.append('neutral')
            if 'walk' in anims:
                i = anims.index('walk')
                del anims[i]
                topAnims.append('walk')
            if 'run' in anims:
                i = anims.index('run')
                del anims[i]
                topAnims.append('run')
            anims.sort()
            anims = topAnims + anims
            if len(anims) == 0:
                continue
            ac = self.createcomponent('actorControl%d' % self.actorControlIndex, (), 'Actor', ActorControl, (self.actorFrame,), animPanel=self, text=actor.getName(), animList=anims, actor=actor)
            ac.pack(expand=1, fill=tk.X)
            self.actorControlList.append(ac)
            self.actorControlIndex = self.actorControlIndex + 1
        self.actorFrame.pack(expand=1, fill=tk.BOTH)

    def clearActorControls(self):
        if False:
            while True:
                i = 10
        if self.actorFrame:
            self.actorFrame.forget()
            self.actorFrame.destroy()
            self.actorFrame = None

    def setActors(self):
        if False:
            return 10
        self.stopActorControls()
        actors = self.session.getSelectedActors()
        aList = []
        for currActor in actors:
            aList.append(currActor)
        self['actorList'] = aList
        self.clearActorControls()
        self.createActorControls()

    def loadAnim(self):
        if False:
            return 10
        animFilename = askopenfilename(defaultextension='.mb', filetypes=(('Maya Models', '*.mb'), ('All files', '*')), initialdir='/i/beta', title='Load Animation', parent=self.component('hull'))
        if not animFilename or animFilename == 'None':
            return
        fileDirName = os.path.dirname(animFilename)
        fileBaseName = os.path.basename(animFilename)
        fileBaseNameBase = os.path.splitext(fileBaseName)[0]
        fileDirNameFN = Filename(fileDirName)
        fileDirNameFN.makeCanonical()
        getModelPath().prependDirectory(fileDirNameFN)
        for currActor in self['actorList']:
            currActor.loadAnims({fileBaseNameBase: fileBaseNameBase})
        self.clearActorControls()
        self.createActorControls()

    def playActorControls(self):
        if False:
            return 10
        self.stopActorControls()
        self.lastT = ClockObject.getGlobalClock().getFrameTime()
        self.playList = self.actorControlList[:]
        taskMgr.add(self.play, self.id + '_UpdateTask')

    def play(self, task):
        if False:
            while True:
                i = 10
        if not self.playList:
            return Task.done
        fLoop = self.loopVar.get()
        currT = ClockObject.getGlobalClock().getFrameTime()
        deltaT = currT - self.lastT
        self.lastT = currT
        for actorControl in self.playList:
            actorControl.play(deltaT * actorControl.playRate, fLoop)
        return Task.cont

    def stopActorControls(self):
        if False:
            return 10
        taskMgr.remove(self.id + '_UpdateTask')

    def getActorControlAt(self, index):
        if False:
            return 10
        return self.actorControlList[index]

    def enableActorControlAt(self, index):
        if False:
            print('Hello World!')
        self.getActorControlAt(index).enableControl()

    def toggleAllControls(self):
        if False:
            i = 10
            return i + 15
        if self.fToggleAll:
            self.disableActorControls()
        else:
            self.enableActorControls()
        self.fToggleAll = 1 - self.fToggleAll

    def enableActorControls(self):
        if False:
            i = 10
            return i + 15
        for actorControl in self.actorControlList:
            actorControl.enableControl()

    def disableActorControls(self):
        if False:
            i = 10
            return i + 15
        for actorControl in self.actorControlList:
            actorControl.disableControl()

    def disableActorControlAt(self, index):
        if False:
            return 10
        self.getActorControlAt(index).disableControl()

    def displayFrameCounts(self):
        if False:
            print('Hello World!')
        for actorControl in self.actorControlList:
            actorControl.displayFrameCounts()

    def displaySeconds(self):
        if False:
            i = 10
            return i + 15
        for actorControl in self.actorControlList:
            actorControl.displaySeconds()

    def resetAllToZero(self):
        if False:
            while True:
                i = 10
        for actorControl in self.actorControlList:
            actorControl.resetToZero()

    def resetAllToEnd(self):
        if False:
            return 10
        for actorControl in self.actorControlList:
            actorControl.resetToEnd()

    def nextFrame(self):
        if False:
            return 10
        for actorControl in self.actorControlList:
            actorControl.nextFrame()

    def previousFrame(self):
        if False:
            return 10
        for actorControl in self.actorControlList:
            actorControl.previousFrame()

    def setDestroyCallBack(self, callBack):
        if False:
            print('Hello World!')
        self.destroyCallBack = callBack

    def destroy(self):
        if False:
            print('Hello World!')
        taskMgr.remove(self.id + '_UpdateTask')
        if self.destroyCallBack is not None:
            self.destroyCallBack()
            self.destroyCallBack = None
        AppShell.destroy(self)

class ActorControl(Pmw.MegaWidget):

    def __init__(self, parent=None, **kw):
        if False:
            i = 10
            return i + 15
        INITOPT = Pmw.INITOPT
        DEFAULT_FONT = (('MS', 'Sans', 'Serif'), 12, 'bold')
        DEFAULT_ANIMS = ('neutral', 'run', 'walk')
        animList = kw.get('animList', DEFAULT_ANIMS)
        if len(animList) > 0:
            initActive = animList[0]
        else:
            initActive = DEFAULT_ANIMS[0]
        optiondefs = (('text', 'Actor', self._updateLabelText), ('animPanel', None, None), ('actor', None, None), ('animList', DEFAULT_ANIMS, None), ('active', initActive, None), ('sLabel_width', 5, None), ('sLabel_font', DEFAULT_FONT, None))
        self.defineoptions(kw, optiondefs)
        Pmw.MegaWidget.__init__(self, parent)
        interior = self.interior()
        interior.configure(relief=tk.RAISED, bd=2)
        self.fps = 24
        self.offset = 0.0
        self.maxSeconds = 1.0
        self.currT = 0.0
        self.fScaleCommand = 0
        self.fOneShot = 0
        self._label = self.createcomponent('label', (), None, tk.Menubutton, (interior,), font=('MSSansSerif', 14, 'bold'), relief=tk.RAISED, bd=1, activebackground='#909090', text=self['text'])
        labelMenu = tk.Menu(self._label, tearoff=0)
        self.unitsVar = tk.IntVar()
        self.unitsVar.set(FRAMES)
        displayMenu = tk.Menu(labelMenu, tearoff=0)
        displayMenu.add_radiobutton(label='Frame count', value=FRAMES, variable=self.unitsVar, command=self.updateDisplay)
        displayMenu.add_radiobutton(label='Seconds', value=SECONDS, variable=self.unitsVar, command=self.updateDisplay)
        labelMenu.add_cascade(label='Display Units', menu=displayMenu)
        labelMenu.add_command(label='Jump To Zero', command=self.resetToZero)
        labelMenu.add_command(label='Jump To End Time', command=self.resetToEnd)
        self._label['menu'] = labelMenu
        self._label.pack(side=tk.LEFT, fill=tk.X)
        self.animMenu = self.createcomponent('animMenu', (), None, Pmw.ComboBox, (interior,), labelpos=tk.W, label_text='Anim:', entry_width=12, selectioncommand=self.selectAnimNamed, scrolledlist_items=self['animList'])
        self.animMenu.selectitem(self['active'])
        self.animMenu.pack(side='left', padx=5, expand=0)
        playRateList = ['1/24.0', '0.1', '0.5', '1.0', '2.0', '5.0', '10.0']
        playRate = '%0.1f' % self['actor'].getPlayRate(self['active'])
        if playRate not in playRateList:
            playRateList.append(playRate)
            playRateList.sort(key=lambda s: eval(s))
        playRateMenu = self.createcomponent('playRateMenu', (), None, Pmw.ComboBox, (interior,), labelpos=tk.W, label_text='Play Rate:', entry_width=4, selectioncommand=self.setPlayRate, scrolledlist_items=playRateList)
        playRateMenu.selectitem(playRate)
        playRateMenu.pack(side=tk.LEFT, padx=5, expand=0)
        frameFrame = tk.Frame(interior, relief=tk.SUNKEN, bd=1)
        self.minLabel = self.createcomponent('minLabel', (), 'sLabel', tk.Label, (frameFrame,), text=0)
        self.minLabel.pack(side=tk.LEFT)
        self.frameControl = self.createcomponent('scale', (), None, tk.Scale, (frameFrame,), from_=0, to=24, resolution=1.0, command=self.goTo, orient=tk.HORIZONTAL, showvalue=1)
        self.frameControl.pack(side=tk.LEFT, expand=1)
        self.frameControl.bind('<Button-1>', self.__onPress)
        self.frameControl.bind('<ButtonRelease-1>', self.__onRelease)
        self.maxLabel = self.createcomponent('maxLabel', (), 'sLabel', tk.Label, (frameFrame,), text=24)
        self.maxLabel.pack(side=tk.LEFT)
        frameFrame.pack(side=tk.LEFT, expand=1, fill=tk.X)
        self.frameActiveVar = tk.IntVar()
        self.frameActiveVar.set(1)
        frameActive = self.createcomponent('checkbutton', (), None, tk.Checkbutton, (interior,), variable=self.frameActiveVar)
        frameActive.pack(side=tk.LEFT, expand=1)
        self.initialiseoptions(ActorControl)
        self.playRate = 1.0
        self.updateDisplay()

    def _updateLabelText(self):
        if False:
            while True:
                i = 10
        self._label['text'] = self['text']

    def updateDisplay(self):
        if False:
            for i in range(10):
                print('nop')
        actor = self['actor']
        active = self['active']
        self.fps = actor.getFrameRate(active)
        if self.fps is None:
            print('unable to get animation fps, zeroing out animation info')
            self.fps = 24
            self.duration = 0
            self.maxFrame = 0
            self.maxSeconds = 0
        else:
            self.duration = actor.getDuration(active)
            self.maxFrame = actor.getNumFrames(active) - 1
            self.maxSeconds = self.offset + self.duration
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

    def __onPress(self, event):
        if False:
            return 10
        self.fScaleCommand = 1

    def __onRelease(self, event):
        if False:
            print('Hello World!')
        self.fScaleCommand = 0

    def selectAnimNamed(self, name):
        if False:
            return 10
        self['active'] = name
        self.component('playRateMenu').selectitem('1.0')
        self.setPlayRate('1.0')
        self.resetToZero()

    def setPlayRate(self, rate):
        if False:
            print('Hello World!')
        self['actor'].setPlayRate(eval(rate), self['active'])
        self.playRate = eval(rate)
        self.updateDisplay()

    def setOffset(self):
        if False:
            for i in range(10):
                print('nop')
        newOffset = askfloat(parent=self.interior(), title=self['text'], prompt='Start offset (seconds):')
        if newOffset is not None:
            self.offset = newOffset
            self.updateDisplay()

    def enableControl(self):
        if False:
            return 10
        self.frameActiveVar.set(1)

    def disableControl(self):
        if False:
            while True:
                i = 10
        self.frameActiveVar.set(0)

    def displayFrameCounts(self):
        if False:
            print('Hello World!')
        self.unitsVar.set(FRAMES)
        self.updateDisplay()

    def displaySeconds(self):
        if False:
            return 10
        self.unitsVar.set(SECONDS)
        self.updateDisplay()

    def play(self, deltaT, fLoop):
        if False:
            while True:
                i = 10
        if self.frameActiveVar.get():
            self.currT = self.currT + deltaT
            if fLoop and self.duration:
                loopT = self.currT % self.duration
                self.goToT(loopT)
            elif self.currT > self.maxSeconds:
                self['animPanel'].playList.remove(self)
            else:
                self.goToT(self.currT)
        else:
            self['animPanel'].playList.remove(self)

    def goToF(self, f):
        if False:
            i = 10
            return i + 15
        if self.unitsVar.get() == FRAMES:
            self.frameControl.set(f)
        else:
            self.frameControl.set(f / self.fps)

    def goToT(self, t):
        if False:
            return 10
        if self.unitsVar.get() == FRAMES:
            self.frameControl.set(t * self.fps)
        else:
            self.frameControl.set(t)

    def goTo(self, t):
        if False:
            while True:
                i = 10
        t = float(t)
        if self.unitsVar.get() == FRAMES:
            t = t / self.fps
        if self.fScaleCommand or self.fOneShot:
            self.currT = t
            self.fOneShot = 0
        self['actor'].pose(self['active'], min(self.maxFrame, int(t * self.fps)))

    def resetToZero(self):
        if False:
            return 10
        self.fOneShot = 1
        self.goToT(0)

    def resetToEnd(self):
        if False:
            for i in range(10):
                print('nop')
        self.fOneShot = 1
        self.goToT(self.duration)

    def nextFrame(self):
        if False:
            print('Hello World!')
        "\n        There needed to be a better way to select an exact frame number\n        as the control slider doesn't have the desired resolution\n        "
        self.fOneShot = 1
        self.goToT((self.currT + 1 / self.fps) % self.duration)

    def previousFrame(self):
        if False:
            while True:
                i = 10
        "\n        There needed to be a better way to select an exact frame number\n        as the control slider doesn't have the desired resolution\n        "
        self.fOneShot = 1
        self.goToT((self.currT - 1 / self.fps) % self.duration)
'\n# EXAMPLE CODE\nfrom direct.actor import Actor\nimport AnimPanel\n\na = Actor.Actor({250:{"head":"phase_3/models/char/dogMM_Shorts-head-250",\n                      "torso":"phase_3/models/char/dogMM_Shorts-torso-250",\n                      "legs":"phase_3/models/char/dogMM_Shorts-legs-250"},\n                 500:{"head":"phase_3/models/char/dogMM_Shorts-head-500",\n                      "torso":"phase_3/models/char/dogMM_Shorts-torso-500",\n                      "legs":"phase_3/models/char/dogMM_Shorts-legs-500"},\n                 1000:{"head":"phase_3/models/char/dogMM_Shorts-head-1000",\n                      "torso":"phase_3/models/char/dogMM_Shorts-torso-1000",\n                      "legs":"phase_3/models/char/dogMM_Shorts-legs-1000"}},\n                {"head":{"walk":"phase_3/models/char/dogMM_Shorts-head-walk",                          "run":"phase_3/models/char/dogMM_Shorts-head-run"},                  "torso":{"walk":"phase_3/models/char/dogMM_Shorts-torso-walk",                           "run":"phase_3/models/char/dogMM_Shorts-torso-run"},                  "legs":{"walk":"phase_3/models/char/dogMM_Shorts-legs-walk",                          "run":"phase_3/models/char/dogMM_Shorts-legs-run"}})\na.attach("head", "torso", "joint-head", 250)\na.attach("torso", "legs", "joint-hips", 250)\na.attach("head", "torso", "joint-head", 500)\na.attach("torso", "legs", "joint-hips", 500)\na.attach("head", "torso", "joint-head", 1000)\na.attach("torso", "legs", "joint-hips", 1000)\na.drawInFront("joint-pupil?", "eyes*", -1, lodName=250)\na.drawInFront("joint-pupil?", "eyes*", -1, lodName=500)\na.drawInFront("joint-pupil?", "eyes*", -1, lodName=1000)\na.setLOD(250, 250, 75)\na.setLOD(500, 75, 15)\na.setLOD(1000, 15, 1)\na.fixBounds()\na.reparentTo(render)\n\n\na2 = Actor.Actor({250:{"head":"phase_3/models/char/dogMM_Shorts-head-250",\n                      "torso":"phase_3/models/char/dogMM_Shorts-torso-250",\n                      "legs":"phase_3/models/char/dogMM_Shorts-legs-250"},\n                 500:{"head":"phase_3/models/char/dogMM_Shorts-head-500",\n                      "torso":"phase_3/models/char/dogMM_Shorts-torso-500",\n                      "legs":"phase_3/models/char/dogMM_Shorts-legs-500"},\n                 1000:{"head":"phase_3/models/char/dogMM_Shorts-head-1000",\n                      "torso":"phase_3/models/char/dogMM_Shorts-torso-1000",\n                      "legs":"phase_3/models/char/dogMM_Shorts-legs-1000"}},\n                {"head":{"walk":"phase_3/models/char/dogMM_Shorts-head-walk",                          "run":"phase_3/models/char/dogMM_Shorts-head-run"},                  "torso":{"walk":"phase_3/models/char/dogMM_Shorts-torso-walk",                           "run":"phase_3/models/char/dogMM_Shorts-torso-run"},                  "legs":{"walk":"phase_3/models/char/dogMM_Shorts-legs-walk",                          "run":"phase_3/models/char/dogMM_Shorts-legs-run"}})\na2.attach("head", "torso", "joint-head", 250)\na2.attach("torso", "legs", "joint-hips", 250)\na2.attach("head", "torso", "joint-head", 500)\na2.attach("torso", "legs", "joint-hips", 500)\na2.attach("head", "torso", "joint-head", 1000)\na2.attach("torso", "legs", "joint-hips", 1000)\na2.drawInFront("joint-pupil?", "eyes*", -1, lodName=250)\na2.drawInFront("joint-pupil?", "eyes*", -1, lodName=500)\na2.drawInFront("joint-pupil?", "eyes*", -1, lodName=1000)\na2.setLOD(250, 250, 75)\na2.setLOD(500, 75, 15)\na2.setLOD(1000, 15, 1)\na2.fixBounds()\na2.reparentTo(render)\n\nap = AnimPanel.AnimPanel([a, a2])\n\n# Alternately\nap = a.animPanel()\nap2 = a2.animPanel()\n\n'