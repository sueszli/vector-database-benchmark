from direct.showbase.DirectObject import DirectObject
from direct.tkwidgets.AppShell import AppShell
from seGeometry import *
from seSelection import *
from direct.task.Task import Task
from direct.tkwidgets.Dial import AngleDial
from direct.tkwidgets.Floater import Floater
from direct.tkwidgets.Slider import Slider
from direct.tkwidgets.EntryScale import EntryScale
from direct.tkwidgets.VectorWidgets import Vector2Entry, Vector3Entry
from direct.tkwidgets.VectorWidgets import ColorEntry
import os, Pmw
import math
from tkinter import Button, Frame, Radiobutton, Checkbutton, Label
from tkinter import StringVar, BooleanVar, Entry, Scale
import tkinter
PRF_UTILITIES = ['lambda: camera.lookAt(render)', 'lambda: camera.setZ(render, 0.0)', 'lambda s = self: s.playbackMarker.lookAt(render)', 'lambda s = self: s.playbackMarker.setZ(render, 0.0)', 'lambda s = self: s.followTerrain(10.0)']

class MopathRecorder(AppShell, DirectObject):
    appname = 'Mopath Recorder Panel'
    frameWidth = 450
    frameHeight = 550
    usecommandarea = 0
    usestatusarea = 0
    count = 0

    def __init__(self, parent=None, **kw):
        if False:
            return 10
        INITOPT = Pmw.INITOPT
        name = 'recorder-%d' % MopathRecorder.count
        MopathRecorder.count += 1
        optiondefs = (('title', self.appname, None), ('nodePath', None, None), ('name', name, None))
        self.defineoptions(kw, optiondefs)
        AppShell.__init__(self)
        self.initialiseoptions(MopathRecorder)
        self.selectNodePathNamed('camera')
        self.parent.resizable(False, False)

    def appInit(self):
        if False:
            return 10
        self.mopathRecorderNode = render.attachNewNode('MopathRecorder')
        self.name = self['name']
        self.widgetDict = {}
        self.variableDict = {}
        self.nodePath = self['nodePath']
        self.playbackNodePath = self.nodePath
        self.nodePathParent = render
        self.recorderNodePath = self.mopathRecorderNode.attachNewNode(self.name)
        self.tempCS = self.recorderNodePath.attachNewNode('mopathRecorderTempCS')
        self.playbackMarker = loader.loadModel('models/misc/sphere')
        self.playbackMarker.setName('Playback Marker')
        self.playbackMarker.reparentTo(self.recorderNodePath)
        self.playbackMarkerIds = self.getChildIds(self.playbackMarker.getChild(0))
        self.playbackMarker.hide()
        self.tangentGroup = self.playbackMarker.attachNewNode('Tangent Group')
        self.tangentGroup.hide()
        self.tangentMarker = loader.loadModel('models/misc/sphere')
        self.tangentMarker.reparentTo(self.tangentGroup)
        self.tangentMarker.setScale(0.5)
        self.tangentMarker.setColor(1, 0, 1, 1)
        self.tangentMarker.setName('Tangent Marker')
        self.tangentMarkerIds = self.getChildIds(self.tangentMarker.getChild(0))
        self.tangentLines = LineNodePath(self.tangentGroup)
        self.tangentLines.setColor(VBase4(1, 0, 1, 1))
        self.tangentLines.setThickness(1)
        self.tangentLines.moveTo(0, 0, 0)
        self.tangentLines.drawTo(0, 0, 0)
        self.tangentLines.create()
        self.nodePathDict = {}
        self.nodePathDict['marker'] = self.playbackMarker
        self.nodePathDict['camera'] = camera
        self.nodePathDict['widget'] = SEditor.widget
        self.nodePathDict['mopathRecorderTempCS'] = self.tempCS
        self.nodePathNames = ['marker', 'camera', 'selected']
        self.manipulandumId = None
        self.trace = LineNodePath(self.recorderNodePath)
        self.oldPlaybackNodePath = None
        self.pointSet = []
        self.prePoints = []
        self.postPoints = []
        self.pointSetDict = {}
        self.pointSetCount = 0
        self.pointSetName = self.name + '-ps-' + repr(self.pointSetCount)
        self.samplingMode = 'Continuous'
        self.preRecordFunc = None
        self.startStopHook = 'f6'
        self.keyframeHook = 'f10'
        self.lastPos = Point3(0)
        self.curveFitter = CurveFitter()
        self.numTicks = 1
        self.numSegs = 40
        self.curveCollection = None
        self.nurbsCurveDrawer = NurbsCurveDrawer()
        self.nurbsCurveDrawer.setCurves(ParametricCurveCollection())
        self.nurbsCurveDrawer.setNumSegs(self.numSegs)
        self.nurbsCurveDrawer.setShowHull(0)
        self.nurbsCurveDrawer.setShowCvs(0)
        self.nurbsCurveDrawer.setNumTicks(0)
        self.nurbsCurveDrawer.setTickScale(5.0)
        self.curveNodePath = self.recorderNodePath.attachNewNode(self.nurbsCurveDrawer.getGeomNode())
        useDirectRenderStyle(self.curveNodePath)
        self.maxT = 0.0
        self.playbackTime = 0.0
        self.loopPlayback = 1
        self.playbackSF = 1.0
        self.desampleFrequency = 1
        self.numSamples = 100
        self.recordStart = 0.0
        self.deltaTime = 0.0
        self.controlStart = 0.0
        self.controlStop = 0.0
        self.recordStop = 0.0
        self.cropFrom = 0.0
        self.cropTo = 0.0
        self.fAdjustingValues = 0
        self.iRayCS = self.recorderNodePath.attachNewNode('mopathRecorderIRayCS')
        self.iRay = SelectionRay(self.iRayCS)
        self.actionEvents = [('DIRECT_undo', self.undoHook), ('DIRECT_pushUndo', self.pushUndoHook), ('DIRECT_undoListEmpty', self.undoListEmptyHook), ('DIRECT_redo', self.redoHook), ('DIRECT_pushRedo', self.pushRedoHook), ('DIRECT_redoListEmpty', self.redoListEmptyHook), ('DIRECT_selectedNodePath', self.selectedNodePathHook), ('DIRECT_deselectedNodePath', self.deselectedNodePathHook), ('DIRECT_manipulateObjectStart', self.manipulateObjectStartHook), ('DIRECT_manipulateObjectCleanup', self.manipulateObjectCleanupHook)]
        for (event, method) in self.actionEvents:
            self.accept(event, method)

    def createInterface(self):
        if False:
            print('Hello World!')
        interior = self.interior()
        fileMenu = self.menuBar.component('File-menu')
        fileMenu.insert_command(fileMenu.index('Quit'), label='Load Curve', command=self.loadCurveFromFile)
        fileMenu.insert_command(fileMenu.index('Quit'), label='Save Curve', command=self.saveCurveToFile)
        self.menuBar.addmenu('Recorder', 'Mopath Recorder Panel Operations')
        self.menuBar.addmenuitem('Recorder', 'command', 'Save current curve as a new point set', label='Save Point Set', command=self.extractPointSetFromCurveCollection)
        self.menuBar.addmenuitem('Recorder', 'command', 'Toggle widget visability', label='Toggle Widget Vis', command=self.toggleWidgetVis)
        self.menuBar.addmenuitem('Recorder', 'command', 'Toggle widget manipulation mode', label='Toggle Widget Mode', command=SEditor.manipulationControl.toggleObjectHandlesMode)
        self.historyWidget = self.createComboBox(self.menuFrame, 'Mopath', 'Path:', 'Select input points to fit curve to', '', self.selectPointSetNamed, expand=1)
        self.undoButton = Button(self.menuFrame, text='Undo', command=SEditor.undo)
        if SEditor.undoList:
            self.undoButton['state'] = 'normal'
        else:
            self.undoButton['state'] = 'disabled'
        self.undoButton.pack(side=tkinter.LEFT, expand=0)
        self.bind(self.undoButton, 'Undo last operation')
        self.redoButton = Button(self.menuFrame, text='Redo', command=SEditor.redo)
        if SEditor.redoList:
            self.redoButton['state'] = 'normal'
        else:
            self.redoButton['state'] = 'disabled'
        self.redoButton.pack(side=tkinter.LEFT, expand=0)
        self.bind(self.redoButton, 'Redo last operation')
        mainFrame = Frame(interior, relief=tkinter.SUNKEN, borderwidth=2)
        frame = Frame(mainFrame)
        widget = self.createButton(frame, 'Recording', 'Node Path:', 'Select Active Mopath Node Path', lambda s=self: SEditor.select(s.nodePath), side=tkinter.LEFT, expand=0)
        widget['relief'] = tkinter.FLAT
        self.nodePathMenu = Pmw.ComboBox(frame, entry_width=20, selectioncommand=self.selectNodePathNamed, scrolledlist_items=self.nodePathNames)
        self.nodePathMenu.selectitem('camera')
        self.nodePathMenuEntry = self.nodePathMenu.component('entryfield_entry')
        self.nodePathMenuBG = self.nodePathMenuEntry.configure('background')[3]
        self.nodePathMenu.pack(side=tkinter.LEFT, fill=tkinter.X, expand=1)
        self.bind(self.nodePathMenu, 'Select active node path used for recording and playback')
        self.recordingType = StringVar()
        self.recordingType.set('New Curve')
        widget = self.createRadiobutton(frame, 'left', 'Recording', 'New Curve', 'Next record session records a new path', self.recordingType, 'New Curve', expand=0)
        widget = self.createRadiobutton(frame, 'left', 'Recording', 'Refine', 'Next record session refines existing path', self.recordingType, 'Refine', expand=0)
        widget = self.createRadiobutton(frame, 'left', 'Recording', 'Extend', 'Next record session extends existing path', self.recordingType, 'Extend', expand=0)
        frame.pack(fill=tkinter.X, expand=1)
        frame = Frame(mainFrame)
        widget = self.createCheckbutton(frame, 'Recording', 'Record', 'On: path is being recorded', self.toggleRecord, 0, side=tkinter.LEFT, fill=tkinter.BOTH, expand=1)
        widget.configure(foreground='Red', relief=tkinter.RAISED, borderwidth=2, anchor=tkinter.CENTER, width=16)
        widget = self.createButton(frame, 'Recording', 'Add Keyframe', 'Add Keyframe To Current Path', self.addKeyframe, side=tkinter.LEFT, expand=1)
        widget = self.createButton(frame, 'Recording', 'Bind Path to Node', 'Bind Motion Path to selected Object', self.bindMotionPathToNode, side=tkinter.LEFT, expand=1)
        frame.pack(fill=tkinter.X, expand=1)
        mainFrame.pack(expand=1, fill=tkinter.X, pady=3)
        playbackFrame = Frame(interior, relief=tkinter.SUNKEN, borderwidth=2)
        Label(playbackFrame, text='PLAYBACK CONTROLS', font=('MSSansSerif', 12, 'bold')).pack(fill=tkinter.X)
        widget = self.createEntryScale(playbackFrame, 'Playback', 'Time', 'Set current playback time', resolution=0.01, command=self.playbackGoTo, side=tkinter.TOP)
        widget.component('hull')['relief'] = tkinter.RIDGE
        widget['preCallback'] = self.stopPlayback
        self.createLabeledEntry(widget.labelFrame, 'Resample', 'Path Duration', 'Set total curve duration', command=self.setPathDuration, side=tkinter.LEFT, expand=0)
        frame = Frame(playbackFrame)
        widget = self.createButton(frame, 'Playback', '<<', 'Jump to start of playback', self.jumpToStartOfPlayback, side=tkinter.LEFT, expand=1)
        widget['font'] = ('MSSansSerif', 12, 'bold')
        widget = self.createCheckbutton(frame, 'Playback', 'Play', 'Start/Stop playback', self.startStopPlayback, 0, side=tkinter.LEFT, fill=tkinter.BOTH, expand=1)
        widget.configure(anchor='center', justify='center', relief=tkinter.RAISED, font=('MSSansSerif', 12, 'bold'))
        widget = self.createButton(frame, 'Playback', '>>', 'Jump to end of playback', self.jumpToEndOfPlayback, side=tkinter.LEFT, expand=1)
        widget['font'] = ('MSSansSerif', 12, 'bold')
        self.createCheckbutton(frame, 'Playback', 'Loop', 'On: loop playback', self.setLoopPlayback, self.loopPlayback, side=tkinter.LEFT, fill=tkinter.BOTH, expand=0)
        frame.pack(fill=tkinter.X, expand=1)
        frame = Frame(playbackFrame)
        widget = Button(frame, text='PB Speed Vernier', relief=tkinter.FLAT, command=lambda s=self: s.setSpeedScale(1.0))
        widget.pack(side=tkinter.LEFT, expand=0)
        self.speedScale = Scale(frame, from_=-1, to=1, resolution=0.01, showvalue=0, width=10, orient='horizontal', command=self.setPlaybackSF)
        self.speedScale.pack(side=tkinter.LEFT, fill=tkinter.X, expand=1)
        self.speedVar = StringVar()
        self.speedVar.set('0.00')
        self.speedEntry = Entry(frame, textvariable=self.speedVar, width=8)
        self.speedEntry.bind('<Return>', lambda e=None, s=self: s.setSpeedScale(float(s.speedVar.get())))
        self.speedEntry.pack(side=tkinter.LEFT, expand=0)
        frame.pack(fill=tkinter.X, expand=1)
        playbackFrame.pack(fill=tkinter.X, pady=2)
        self.mainNotebook = Pmw.NoteBook(interior)
        self.mainNotebook.pack(fill=tkinter.BOTH, expand=1)
        self.resamplePage = self.mainNotebook.add('Resample')
        self.refinePage = self.mainNotebook.add('Refine')
        self.extendPage = self.mainNotebook.add('Extend')
        self.cropPage = self.mainNotebook.add('Crop')
        self.drawPage = self.mainNotebook.add('Draw')
        self.optionsPage = self.mainNotebook.add('Options')
        label = Label(self.resamplePage, text='RESAMPLE CURVE', font=('MSSansSerif', 12, 'bold'))
        label.pack(fill=tkinter.X)
        resampleFrame = Frame(self.resamplePage, relief=tkinter.SUNKEN, borderwidth=2)
        label = Label(resampleFrame, text='RESAMPLE CURVE', font=('MSSansSerif', 12, 'bold')).pack()
        widget = self.createSlider(resampleFrame, 'Resample', 'Num. Samples', 'Number of samples in resampled curve', resolution=1, min=2, max=1000, command=self.setNumSamples)
        widget.component('hull')['relief'] = tkinter.RIDGE
        widget['postCallback'] = self.sampleCurve
        frame = Frame(resampleFrame)
        self.createButton(frame, 'Resample', 'Make Even', 'Apply timewarp so resulting path has constant velocity', self.makeEven, side=tkinter.LEFT, fill=tkinter.X, expand=1)
        self.createButton(frame, 'Resample', 'Face Forward', 'Compute HPR so resulting hpr curve faces along xyz tangent', self.faceForward, side=tkinter.LEFT, fill=tkinter.X, expand=1)
        frame.pack(fill=tkinter.X, expand=0)
        resampleFrame.pack(fill=tkinter.X, expand=0, pady=2)
        desampleFrame = Frame(self.resamplePage, relief=tkinter.SUNKEN, borderwidth=2)
        Label(desampleFrame, text='DESAMPLE CURVE', font=('MSSansSerif', 12, 'bold')).pack()
        widget = self.createSlider(desampleFrame, 'Resample', 'Points Between Samples', 'Specify number of points to skip between samples', min=1, max=100, resolution=1, command=self.setDesampleFrequency)
        widget.component('hull')['relief'] = tkinter.RIDGE
        widget['postCallback'] = self.desampleCurve
        desampleFrame.pack(fill=tkinter.X, expand=0, pady=2)
        refineFrame = Frame(self.refinePage, relief=tkinter.SUNKEN, borderwidth=2)
        label = Label(refineFrame, text='REFINE CURVE', font=('MSSansSerif', 12, 'bold'))
        label.pack(fill=tkinter.X)
        widget = self.createSlider(refineFrame, 'Refine Page', 'Refine From', 'Begin time of refine pass', resolution=0.01, command=self.setRecordStart)
        widget['preCallback'] = self.setRefineMode
        widget['postCallback'] = lambda s=self: s.getPrePoints('Refine')
        widget = self.createSlider(refineFrame, 'Refine Page', 'Control Start', 'Time when full control of node path is given during refine pass', resolution=0.01, command=self.setControlStart)
        widget['preCallback'] = self.setRefineMode
        widget = self.createSlider(refineFrame, 'Refine Page', 'Control Stop', 'Time when node path begins transition back to original curve', resolution=0.01, command=self.setControlStop)
        widget['preCallback'] = self.setRefineMode
        widget = self.createSlider(refineFrame, 'Refine Page', 'Refine To', 'Stop time of refine pass', resolution=0.01, command=self.setRefineStop)
        widget['preCallback'] = self.setRefineMode
        widget['postCallback'] = self.getPostPoints
        refineFrame.pack(fill=tkinter.X)
        extendFrame = Frame(self.extendPage, relief=tkinter.SUNKEN, borderwidth=2)
        label = Label(extendFrame, text='EXTEND CURVE', font=('MSSansSerif', 12, 'bold'))
        label.pack(fill=tkinter.X)
        widget = self.createSlider(extendFrame, 'Extend Page', 'Extend From', 'Begin time of extend pass', resolution=0.01, command=self.setRecordStart)
        widget['preCallback'] = self.setExtendMode
        widget['postCallback'] = lambda s=self: s.getPrePoints('Extend')
        widget = self.createSlider(extendFrame, 'Extend Page', 'Control Start', 'Time when full control of node path is given during extend pass', resolution=0.01, command=self.setControlStart)
        widget['preCallback'] = self.setExtendMode
        extendFrame.pack(fill=tkinter.X)
        cropFrame = Frame(self.cropPage, relief=tkinter.SUNKEN, borderwidth=2)
        label = Label(cropFrame, text='CROP CURVE', font=('MSSansSerif', 12, 'bold'))
        label.pack(fill=tkinter.X)
        widget = self.createSlider(cropFrame, 'Crop Page', 'Crop From', 'Delete all curve points before this time', resolution=0.01, command=self.setCropFrom)
        widget = self.createSlider(cropFrame, 'Crop Page', 'Crop To', 'Delete all curve points after this time', resolution=0.01, command=self.setCropTo)
        self.createButton(cropFrame, 'Crop Page', 'Crop Curve', 'Crop curve to specified from to times', self.cropCurve, fill=tkinter.NONE)
        cropFrame.pack(fill=tkinter.X)
        drawFrame = Frame(self.drawPage, relief=tkinter.SUNKEN, borderwidth=2)
        self.sf = Pmw.ScrolledFrame(self.drawPage, horizflex='elastic')
        self.sf.pack(fill='both', expand=1)
        sfFrame = self.sf.interior()
        label = Label(sfFrame, text='CURVE RENDERING STYLE', font=('MSSansSerif', 12, 'bold'))
        label.pack(fill=tkinter.X)
        frame = Frame(sfFrame)
        Label(frame, text='SHOW:').pack(side=tkinter.LEFT, expand=0)
        widget = self.createCheckbutton(frame, 'Style', 'Path', 'On: path is visible', self.setPathVis, 1, side=tkinter.LEFT, fill=tkinter.X, expand=1)
        widget = self.createCheckbutton(frame, 'Style', 'Knots', 'On: path knots are visible', self.setKnotVis, 1, side=tkinter.LEFT, fill=tkinter.X, expand=1)
        widget = self.createCheckbutton(frame, 'Style', 'CVs', 'On: path CVs are visible', self.setCvVis, 0, side=tkinter.LEFT, fill=tkinter.X, expand=1)
        widget = self.createCheckbutton(frame, 'Style', 'Hull', 'On: path hull is visible', self.setHullVis, 0, side=tkinter.LEFT, fill=tkinter.X, expand=1)
        widget = self.createCheckbutton(frame, 'Style', 'Trace', 'On: record is visible', self.setTraceVis, 0, side=tkinter.LEFT, fill=tkinter.X, expand=1)
        widget = self.createCheckbutton(frame, 'Style', 'Marker', 'On: playback marker is visible', self.setMarkerVis, 0, side=tkinter.LEFT, fill=tkinter.X, expand=1)
        frame.pack(fill=tkinter.X, expand=1)
        widget = self.createSlider(sfFrame, 'Style', 'Num Segs', 'Set number of segments used to approximate each parametric unit', min=1.0, max=400, resolution=1.0, value=40, command=self.setNumSegs, side=tkinter.TOP)
        widget.component('hull')['relief'] = tkinter.RIDGE
        widget = self.createSlider(sfFrame, 'Style', 'Num Ticks', 'Set number of tick marks drawn for each unit of time', min=0.0, max=10.0, resolution=1.0, value=0.0, command=self.setNumTicks, side=tkinter.TOP)
        widget.component('hull')['relief'] = tkinter.RIDGE
        widget = self.createSlider(sfFrame, 'Style', 'Tick Scale', 'Set visible size of time tick marks', min=0.01, max=100.0, resolution=0.01, value=5.0, command=self.setTickScale, side=tkinter.TOP)
        widget.component('hull')['relief'] = tkinter.RIDGE
        self.createColorEntry(sfFrame, 'Style', 'Path Color', 'Color of curve', command=self.setPathColor, value=[255.0, 255.0, 255.0, 255.0])
        self.createColorEntry(sfFrame, 'Style', 'Knot Color', 'Color of knots', command=self.setKnotColor, value=[0, 0, 255.0, 255.0])
        self.createColorEntry(sfFrame, 'Style', 'CV Color', 'Color of CVs', command=self.setCvColor, value=[255.0, 0, 0, 255.0])
        self.createColorEntry(sfFrame, 'Style', 'Tick Color', 'Color of Ticks', command=self.setTickColor, value=[255.0, 0, 0, 255.0])
        self.createColorEntry(sfFrame, 'Style', 'Hull Color', 'Color of Hull', command=self.setHullColor, value=[255.0, 128.0, 128.0, 255.0])
        optionsFrame = Frame(self.optionsPage, relief=tkinter.SUNKEN, borderwidth=2)
        label = Label(optionsFrame, text='RECORDING OPTIONS', font=('MSSansSerif', 12, 'bold'))
        label.pack(fill=tkinter.X)
        frame = Frame(optionsFrame)
        widget = self.createLabeledEntry(frame, 'Recording', 'Record Hook', 'Hook used to start/stop recording', value=self.startStopHook, command=self.setStartStopHook)[0]
        label = self.getWidget('Recording', 'Record Hook-Label')
        label.configure(width=16, anchor=tkinter.W)
        self.setStartStopHook()
        widget = self.createLabeledEntry(frame, 'Recording', 'Keyframe Hook', 'Hook used to add a new keyframe', value=self.keyframeHook, command=self.setKeyframeHook)[0]
        label = self.getWidget('Recording', 'Keyframe Hook-Label')
        label.configure(width=16, anchor=tkinter.W)
        self.setKeyframeHook()
        frame.pack(expand=1, fill=tkinter.X)
        frame = Frame(optionsFrame)
        widget = self.createComboBox(frame, 'Recording', 'Pre-Record Func', 'Function called before sampling each point', PRF_UTILITIES, self.setPreRecordFunc, history=1, expand=1)
        widget.configure(label_width=16, label_anchor=tkinter.W)
        widget.configure(entryfield_entry_state='normal')
        self.preRecordFunc = eval(PRF_UTILITIES[0])
        self.createCheckbutton(frame, 'Recording', 'PRF Active', 'On: Pre Record Func enabled', None, 0, side=tkinter.LEFT, fill=tkinter.BOTH, expand=0)
        frame.pack(expand=1, fill=tkinter.X)
        optionsFrame.pack(fill=tkinter.X, pady=2)
        self.mainNotebook.setnaturalsize()

    def pushUndo(self, fResetRedo=1):
        if False:
            while True:
                i = 10
        SEditor.pushUndo([self.nodePath])

    def undoHook(self):
        if False:
            return 10
        pass

    def pushUndoHook(self):
        if False:
            i = 10
            return i + 15
        self.undoButton.configure(state='normal')

    def undoListEmptyHook(self):
        if False:
            for i in range(10):
                print('nop')
        self.undoButton.configure(state='disabled')

    def pushRedo(self):
        if False:
            i = 10
            return i + 15
        SEditor.pushRedo([self.nodePath])

    def redoHook(self):
        if False:
            return 10
        pass

    def pushRedoHook(self):
        if False:
            for i in range(10):
                print('nop')
        self.redoButton.configure(state='normal')

    def redoListEmptyHook(self):
        if False:
            i = 10
            return i + 15
        self.redoButton.configure(state='disabled')

    def selectedNodePathHook(self, nodePath):
        if False:
            while True:
                i = 10
        '\n        Hook called upon selection of a node path used to select playback\n        marker if subnode selected\n        '
        taskMgr.remove(self.name + '-curveEditTask')
        print(nodePath.get_key())
        if nodePath.get_key() in self.playbackMarkerIds:
            SEditor.select(self.playbackMarker)
        elif nodePath.get_key() in self.tangentMarkerIds:
            SEditor.select(self.tangentMarker)
        elif nodePath.get_key() == self.playbackMarker.get_key():
            self.tangentGroup.show()
            taskMgr.add(self.curveEditTask, self.name + '-curveEditTask')
        elif nodePath.get_key() == self.tangentMarker.get_key():
            self.tangentGroup.show()
            taskMgr.add(self.curveEditTask, self.name + '-curveEditTask')
        else:
            self.tangentGroup.hide()

    def getChildIds(self, nodePath):
        if False:
            print('Hello World!')
        ids = [nodePath.get_key()]
        kids = nodePath.getChildren()
        for kid in kids:
            ids += self.getChildIds(kid)
        return ids

    def deselectedNodePathHook(self, nodePath):
        if False:
            i = 10
            return i + 15
        '\n        Hook called upon deselection of a node path used to select playback\n        marker if subnode selected\n        '
        if nodePath.get_key() == self.playbackMarker.get_key() or nodePath.get_key() == self.tangentMarker.get_key():
            self.tangentGroup.hide()

    def curveEditTask(self, state):
        if False:
            while True:
                i = 10
        if self.curveCollection != None:
            if self.manipulandumId == self.playbackMarker.get_key():
                self.playbackMarker.getChild(0).show()
                pos = Point3(0)
                hpr = Point3(0)
                pos = self.playbackMarker.getPos(self.nodePathParent)
                hpr = self.playbackMarker.getHpr(self.nodePathParent)
                self.curveCollection.adjustXyz(self.playbackTime, VBase3(pos[0], pos[1], pos[2]))
                self.curveCollection.adjustHpr(self.playbackTime, VBase3(hpr[0], hpr[1], hpr[2]))
                self.nurbsCurveDrawer.draw()
            if self.manipulandumId == self.tangentMarker.get_key():
                self.playbackMarker.getChild(0).hide()
                tan = self.tangentMarker.getPos()
                tan2Curve = Vec3(self.playbackMarker.getMat(self.nodePathParent).xformVec(tan))
                self.curveCollection.getXyzCurve().adjustTangent(self.playbackTime, tan2Curve[0], tan2Curve[1], tan2Curve[2])
                self.nurbsCurveDrawer.draw()
            else:
                self.playbackMarker.getChild(0).show()
                tan = Point3(0)
                self.curveCollection.getXyzCurve().getTangent(self.playbackTime, tan)
                tan.assign(self.nodePathParent.getMat(self.playbackMarker).xformVec(tan))
                self.tangentMarker.setPos(tan)
            self.tangentLines.setVertex(1, tan[0], tan[1], tan[2])
        return Task.cont

    def manipulateObjectStartHook(self):
        if False:
            while True:
                i = 10
        self.manipulandumId = None
        if SEditor.selected.last:
            if SEditor.selected.last.get_key() == self.playbackMarker.get_key():
                self.manipulandumId = self.playbackMarker.get_key()
            elif SEditor.selected.last.get_key() == self.tangentMarker.get_key():
                self.manipulandumId = self.tangentMarker.get_key()

    def manipulateObjectCleanupHook(self):
        if False:
            while True:
                i = 10
        self.manipulandumId = None

    def onDestroy(self, event):
        if False:
            for i in range(10):
                print('nop')
        for (event, method) in self.actionEvents:
            self.ignore(event)
        self.ignore(self.startStopHook)
        self.ignore(self.keyframeHook)
        self.curveNodePath.reparentTo(self.recorderNodePath)
        self.trace.reparentTo(self.recorderNodePath)
        self.recorderNodePath.removeNode()
        SEditor.deselect(self.playbackMarker)
        SEditor.deselect(self.tangentMarker)
        taskMgr.remove(self.name + '-recordTask')
        taskMgr.remove(self.name + '-playbackTask')
        taskMgr.remove(self.name + '-curveEditTask')
        self.mopathRecorderNode.removeChildren()
        self.mopathRecorderNode.removeNode()
        messenger.send('mPath_close')
        messenger.send('SGE_Update Explorer', [render])

    def createNewPointSet(self, curveName=None):
        if False:
            while True:
                i = 10
        if curveName == None:
            self.pointSetName = self.name + '-ps-' + repr(self.pointSetCount)
        else:
            self.pointSetName = curveName
        self.pointSet = self.pointSetDict[self.pointSetName] = []
        comboBox = self.getWidget('Mopath', 'Path:')
        scrolledList = comboBox.component('scrolledlist')
        listbox = scrolledList.component('listbox')
        names = list(listbox.get(0, 'end'))
        names.append(self.pointSetName)
        scrolledList.setlist(names)
        comboBox.selectitem(self.pointSetName)
        self.pointSetCount += 1

    def extractPointSetFromCurveFitter(self, curveName=None):
        if False:
            i = 10
            return i + 15
        self.createNewPointSet(curveName)
        for i in range(self.curveFitter.getNumSamples()):
            time = self.curveFitter.getSampleT(i)
            pos = Point3(self.curveFitter.getSampleXyz(i))
            hpr = Point3(self.curveFitter.getSampleHpr(i))
            self.pointSet.append([time, pos, hpr])

    def extractPointSetFromCurveCollection(self, curveName=None):
        if False:
            print('Hello World!')
        self.maxT = self.curveCollection.getMaxT()
        samplesPerSegment = min(30.0, 1000.0 / self.curveCollection.getMaxT())
        self.setNumSamples(self.maxT * samplesPerSegment)
        self.sampleCurve(fCompute=0, curveName=curveName)
        self.updateWidgets()

    def selectPointSetNamed(self, name):
        if False:
            while True:
                i = 10
        self.pointSet = self.pointSetDict.get(name, None)
        self.curveFitter.reset()
        for (time, pos, hpr) in self.pointSet:
            self.curveFitter.addXyzHpr(time, pos, hpr)
        self.computeCurves()

    def setPathVis(self):
        if False:
            print('Hello World!')
        if self.getVariable('Style', 'Path').get():
            self.curveNodePath.show()
        else:
            self.curveNodePath.hide()

    def setKnotVis(self):
        if False:
            for i in range(10):
                print('nop')
        self.nurbsCurveDrawer.setShowKnots(self.getVariable('Style', 'Knots').get())

    def setCvVis(self):
        if False:
            while True:
                i = 10
        self.nurbsCurveDrawer.setShowCvs(self.getVariable('Style', 'CVs').get())

    def setHullVis(self):
        if False:
            i = 10
            return i + 15
        self.nurbsCurveDrawer.setShowHull(self.getVariable('Style', 'Hull').get())

    def setTraceVis(self):
        if False:
            while True:
                i = 10
        if self.getVariable('Style', 'Trace').get():
            self.trace.show()
        else:
            self.trace.hide()

    def setMarkerVis(self):
        if False:
            return 10
        if self.getVariable('Style', 'Marker').get():
            self.playbackMarker.reparentTo(self.recorderNodePath)
        else:
            self.playbackMarker.reparentTo(hidden)

    def setNumSegs(self, value):
        if False:
            i = 10
            return i + 15
        self.numSegs = int(value)
        self.nurbsCurveDrawer.setNumSegs(self.numSegs)

    def setNumTicks(self, value):
        if False:
            return 10
        self.nurbsCurveDrawer.setNumTicks(float(value))

    def setTickScale(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.nurbsCurveDrawer.setTickScale(float(value))

    def setPathColor(self, color):
        if False:
            while True:
                i = 10
        self.nurbsCurveDrawer.setColor(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
        self.nurbsCurveDrawer.draw()

    def setKnotColor(self, color):
        if False:
            return 10
        self.nurbsCurveDrawer.setKnotColor(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

    def setCvColor(self, color):
        if False:
            print('Hello World!')
        self.nurbsCurveDrawer.setCvColor(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

    def setTickColor(self, color):
        if False:
            while True:
                i = 10
        self.nurbsCurveDrawer.setTickColor(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

    def setHullColor(self, color):
        if False:
            return 10
        self.nurbsCurveDrawer.setHullColor(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

    def setStartStopHook(self, event=None):
        if False:
            return 10
        self.ignore(self.startStopHook)
        hook = self.getVariable('Recording', 'Record Hook').get()
        self.startStopHook = hook
        self.accept(self.startStopHook, self.toggleRecordVar)

    def setKeyframeHook(self, event=None):
        if False:
            print('Hello World!')
        self.ignore(self.keyframeHook)
        hook = self.getVariable('Recording', 'Keyframe Hook').get()
        self.keyframeHook = hook
        self.accept(self.keyframeHook, self.addKeyframe)

    def reset(self):
        if False:
            return 10
        self.pointSet = []
        self.hasPoints = 0
        self.curveCollection = None
        self.curveFitter.reset()
        self.nurbsCurveDrawer.hide()

    def setSamplingMode(self, mode):
        if False:
            while True:
                i = 10
        self.samplingMode = mode

    def disableKeyframeButton(self):
        if False:
            i = 10
            return i + 15
        self.getWidget('Recording', 'Add Keyframe')['state'] = 'disabled'

    def enableKeyframeButton(self):
        if False:
            i = 10
            return i + 15
        self.getWidget('Recording', 'Add Keyframe')['state'] = 'normal'

    def setRecordingType(self, type):
        if False:
            i = 10
            return i + 15
        self.recordingType.set(type)

    def setNewCurveMode(self):
        if False:
            for i in range(10):
                print('nop')
        self.setRecordingType('New Curve')

    def setRefineMode(self):
        if False:
            print('Hello World!')
        self.setRecordingType('Refine')

    def setExtendMode(self):
        if False:
            return 10
        self.setRecordingType('Extend')

    def toggleRecordVar(self):
        if False:
            for i in range(10):
                print('nop')
        v = self.getVariable('Recording', 'Record')
        v.set(1 - v.get())
        self.toggleRecord()

    def toggleRecord(self):
        if False:
            i = 10
            return i + 15
        if self.getVariable('Recording', 'Record').get():
            self.markingNode = self.nodePath.getParent().attachNewNode('MopthMarkerNode')
            self.nodePath.copyTo(self.markingNode)
            self.markingNode.wrtReparentTo(render)
            taskMgr.remove(self.name + '-recordTask')
            taskMgr.remove(self.name + '-curveEditTask')
            self.nurbsCurveDrawer.hide()
            self.curveFitter.reset()
            if self.samplingMode == 'Continuous':
                self.disableKeyframeButton()
            self.createNewPointSet()
            self.initTrace()
            if self.samplingMode == 'Keyframe':
                self.lastPos.assign(Point3(self.nodePath.getPos(self.nodePathParent)))
                self.deltaTime = 0.0
                self.recordPoint(self.recordStart)
            else:
                if self.recordingType.get() == 'Refine' or self.recordingType.get() == 'Extend':
                    self.loopPlayback = 0
                    self.getVariable('Playback', 'Loop').set(0)
                    self.oldPlaybackNodePath = self.playbackNodePath
                    self.setPlaybackNodePath(self.tempCS)
                    self.nodePath.reparentTo(self.playbackNodePath)
                    self.nodePath.setPosHpr(0, 0, 0, 0, 0, 0)
                    self.playbackGoTo(self.recordStart)
                    self.startPlayback()
                t = taskMgr.add(self.recordTask, self.name + '-recordTask')
                t.startTime = globalClock.getFrameTime()
        else:
            self.markingNode.removeNode()
            if self.samplingMode == 'Continuous':
                taskMgr.remove(self.name + '-recordTask')
                if self.recordingType.get() == 'Refine' or self.recordingType.get() == 'Extend':
                    self.nodePath.wrtReparentTo(self.nodePathParent)
                    self.setPlaybackNodePath(self.oldPlaybackNodePath)
            else:
                self.addKeyframe(0)
            self.setSamplingMode('Continuous')
            self.enableKeyframeButton()
            if self.recordingType.get() == 'Refine' or self.recordingType.get() == 'Extend':
                self.mergePoints()
                self.prePoints = []
                self.postPoints = []
                self.setNewCurveMode()
            self.computeCurves()

    def recordTask(self, state):
        if False:
            return 10
        time = self.recordStart + (globalClock.getFrameTime() - state.startTime)
        self.recordPoint(time)
        return Task.cont

    def addKeyframe(self, fToggleRecord=1):
        if False:
            return 10
        if fToggleRecord and (not self.getVariable('Recording', 'Record').get()):
            self.setSamplingMode('Keyframe')
            self.toggleRecordVar()
        else:
            pos = self.nodePath.getPos(self.nodePathParent)
            deltaPos = Vec3(pos - self.lastPos).length()
            if deltaPos != 0:
                self.deltaTime = self.deltaTime + deltaPos
            else:
                self.deltaTime = self.deltaTime + 1.0
            self.recordPoint(self.recordStart + self.deltaTime)
            self.lastPos.assign(pos)

    def easeInOut(self, t):
        if False:
            return 10
        x = t * t
        return 3 * x - 2 * t * x

    def setPreRecordFunc(self, func):
        if False:
            while True:
                i = 10
        self.preRecordFunc = eval(func)
        self.getVariable('Recording', 'PRF Active').set(1)

    def recordPoint(self, time):
        if False:
            print('Hello World!')
        if self.getVariable('Recording', 'PRF Active').get() and self.preRecordFunc != None:
            self.preRecordFunc()
        pos = self.nodePath.getPos(self.nodePathParent)
        hpr = self.nodePath.getHpr(self.nodePathParent)
        qNP = Quat()
        qNP.setHpr(hpr)
        if self.recordingType.get() == 'Refine' or self.recordingType.get() == 'Extend':
            if time < self.controlStart and self.controlStart - self.recordStart != 0.0:
                rPos = self.playbackNodePath.getPos(self.nodePathParent)
                rHpr = self.playbackNodePath.getHpr(self.nodePathParent)
                qR = Quat()
                qR.setHpr(rHpr)
                t = self.easeInOut((time - self.recordStart) / (self.controlStart - self.recordStart))
                pos = rPos * (1 - t) + pos * t
                q = qSlerp(qR, qNP, t)
                hpr.assign(q.getHpr())
            elif self.recordingType.get() == 'Refine' and time > self.controlStop and (self.recordStop - self.controlStop != 0.0):
                rPos = self.playbackNodePath.getPos(self.nodePathParent)
                rHpr = self.playbackNodePath.getHpr(self.nodePathParent)
                qR = Quat()
                qR.setHpr(rHpr)
                t = self.easeInOut((time - self.controlStop) / (self.recordStop - self.controlStop))
                pos = pos * (1 - t) + rPos * t
                q = qSlerp(qNP, qR, t)
                hpr.assign(q.getHpr())
        self.pointSet.append([time, pos, hpr])
        self.curveFitter.addXyzHpr(time, pos, hpr)
        if self.samplingMode == 'Keyframe':
            self.trace.reset()
            for (t, p, h) in self.pointSet:
                self.trace.drawTo(p[0], p[1], p[2])
            self.trace.create()

    def computeCurves(self):
        if False:
            return 10
        if self.curveFitter.getNumSamples() == 0:
            print('MopathRecorder.computeCurves: Must define curve first')
            return
        self.curveFitter.sortPoints()
        self.curveFitter.wrapHpr()
        self.curveFitter.computeTangents(1)
        self.curveCollection = self.curveFitter.makeNurbs()
        self.nurbsCurveDrawer.setCurves(self.curveCollection)
        self.nurbsCurveDrawer.draw()
        self.updateWidgets()

    def initTrace(self):
        if False:
            return 10
        self.trace.reset()
        self.trace.reparentTo(self.nodePathParent)
        self.trace.show()

    def updateWidgets(self):
        if False:
            while True:
                i = 10
        if not self.curveCollection:
            return
        self.fAdjustingValues = 1
        maxT = self.curveCollection.getMaxT()
        maxT_text = '%0.2f' % maxT
        self.getWidget('Playback', 'Time').configure(max=maxT_text)
        self.getVariable('Resample', 'Path Duration').set(maxT_text)
        widget = self.getWidget('Refine Page', 'Refine From')
        widget.configure(max=maxT)
        widget.set(0.0)
        widget = self.getWidget('Refine Page', 'Control Start')
        widget.configure(max=maxT)
        widget.set(0.0)
        widget = self.getWidget('Refine Page', 'Control Stop')
        widget.configure(max=maxT)
        widget.set(float(maxT))
        widget = self.getWidget('Refine Page', 'Refine To')
        widget.configure(max=maxT)
        widget.set(float(maxT))
        widget = self.getWidget('Extend Page', 'Extend From')
        widget.configure(max=maxT)
        widget.set(float(0.0))
        widget = self.getWidget('Extend Page', 'Control Start')
        widget.configure(max=maxT)
        widget.set(float(0.0))
        widget = self.getWidget('Crop Page', 'Crop From')
        widget.configure(max=maxT)
        widget.set(float(0.0))
        widget = self.getWidget('Crop Page', 'Crop To')
        widget.configure(max=maxT)
        widget.set(float(maxT))
        self.maxT = float(maxT)
        numSamples = self.curveFitter.getNumSamples()
        widget = self.getWidget('Resample', 'Points Between Samples')
        widget.configure(max=numSamples)
        widget = self.getWidget('Resample', 'Num. Samples')
        widget.configure(max=4 * numSamples)
        widget.set(numSamples, 0)
        self.fAdjustingValues = 0

    def selectNodePathNamed(self, name):
        if False:
            return 10
        nodePath = None
        if name == 'init':
            nodePath = self.nodePath
            self.addNodePath(nodePath)
        elif name == 'selected':
            nodePath = SEditor.selected.last
            self.addNodePath(nodePath)
        else:
            nodePath = self.nodePathDict.get(name, None)
            if nodePath == None:
                try:
                    nodePath = eval(name)
                    if isinstance(nodePath, NodePath):
                        self.addNodePath(nodePath)
                    else:
                        nodePath = None
                except:
                    nodePath = None
                    listbox = self.nodePathMenu.component('scrolledlist')
                    listbox.setlist(self.nodePathNames)
            else:
                if name == 'widget':
                    SEditor.selected.getWrtAll()
                if name == 'marker':
                    self.playbackMarker.show()
                    tan = Point3(0)
                    if self.curveCollection != None:
                        self.curveCollection.getXyzCurve().getTangent(self.playbackTime, tan)
                    self.tangentMarker.setPos(tan)
                else:
                    self.playbackMarker.hide()
        self.setNodePath(nodePath)
        messenger.send('mPath_requestCurveList', [nodePath, self.name])
        self.accept('curveListFor' + self.name, self.addCurvesFromNodepath)

    def setNodePath(self, nodePath):
        if False:
            return 10
        self.playbackNodePath = self.nodePath = nodePath
        if self.nodePath:
            self.nodePathParent = self.nodePath.getParent()
            self.curveNodePath.reparentTo(self.nodePathParent)
            self.nodePathMenuEntry.configure(background=self.nodePathMenuBG)
        else:
            self.nodePathMenuEntry.configure(background='Pink')

    def setPlaybackNodePath(self, nodePath):
        if False:
            return 10
        self.playbackNodePath = nodePath

    def addNodePath(self, nodePath):
        if False:
            i = 10
            return i + 15
        self.addNodePathToDict(nodePath, self.nodePathNames, self.nodePathMenu, self.nodePathDict)

    def addNodePathToDict(self, nodePath, names, menu, dict):
        if False:
            while True:
                i = 10
        if not nodePath:
            return
        name = nodePath.getName()
        if name in ['mopathRecorderTempCS', 'widget', 'camera', 'marker']:
            dictName = name
        else:
            dictName = name
        if dictName not in dict:
            names.append(dictName)
            listbox = menu.component('scrolledlist')
            listbox.setlist(names)
            dict[dictName] = nodePath
        menu.selectitem(dictName)

    def setLoopPlayback(self):
        if False:
            return 10
        self.loopPlayback = self.getVariable('Playback', 'Loop').get()

    def playbackGoTo(self, time):
        if False:
            print('Hello World!')
        if self.curveCollection == None:
            return
        self.playbackTime = CLAMP(time, 0.0, self.maxT)
        if self.curveCollection != None:
            pos = Point3(0)
            hpr = Point3(0)
            self.curveCollection.evaluate(self.playbackTime, pos, hpr)
            self.playbackNodePath.setPosHpr(self.nodePathParent, pos, hpr)

    def startPlayback(self):
        if False:
            while True:
                i = 10
        if self.curveCollection == None:
            return
        self.stopPlayback()
        self.getVariable('Playback', 'Play').set(1)
        t = taskMgr.add(self.playbackTask, self.name + '-playbackTask')
        t.currentTime = self.playbackTime
        t.lastTime = globalClock.getFrameTime()

    def setSpeedScale(self, value):
        if False:
            print('Hello World!')
        self.speedScale.set(math.log10(value))

    def setPlaybackSF(self, value):
        if False:
            return 10
        self.playbackSF = pow(10.0, float(value))
        self.speedVar.set('%0.2f' % self.playbackSF)

    def playbackTask(self, state):
        if False:
            print('Hello World!')
        time = globalClock.getFrameTime()
        dTime = self.playbackSF * (time - state.lastTime)
        state.lastTime = time
        if self.loopPlayback:
            cTime = (state.currentTime + dTime) % self.maxT
        else:
            cTime = state.currentTime + dTime
        if self.recordingType.get() == 'Refine' and cTime > self.recordStop:
            self.getWidget('Playback', 'Time').set(self.recordStop)
            self.stopPlayback()
            self.toggleRecordVar()
            return Task.done
        elif self.loopPlayback == 0 and cTime > self.maxT:
            self.getWidget('Playback', 'Time').set(self.maxT)
            self.stopPlayback()
            return Task.done
        elif self.recordingType.get() == 'Extend' and cTime > self.controlStart:
            self.getWidget('Playback', 'Time').set(self.controlStart)
            self.stopPlayback()
            return Task.done
        self.getWidget('Playback', 'Time').set(cTime)
        state.currentTime = cTime
        return Task.cont

    def stopPlayback(self):
        if False:
            i = 10
            return i + 15
        self.getVariable('Playback', 'Play').set(0)
        taskMgr.remove(self.name + '-playbackTask')

    def jumpToStartOfPlayback(self):
        if False:
            while True:
                i = 10
        self.stopPlayback()
        self.getWidget('Playback', 'Time').set(0.0)

    def jumpToEndOfPlayback(self):
        if False:
            for i in range(10):
                print('nop')
        self.stopPlayback()
        if self.curveCollection != None:
            self.getWidget('Playback', 'Time').set(self.maxT)

    def startStopPlayback(self):
        if False:
            return 10
        if self.getVariable('Playback', 'Play').get():
            self.startPlayback()
        else:
            self.stopPlayback()

    def setDesampleFrequency(self, frequency):
        if False:
            for i in range(10):
                print('nop')
        self.desampleFrequency = frequency

    def desampleCurve(self):
        if False:
            for i in range(10):
                print('nop')
        if self.curveFitter.getNumSamples() == 0:
            print('MopathRecorder.desampleCurve: Must define curve first')
            return
        self.curveFitter.desample(self.desampleFrequency)
        self.computeCurves()
        self.extractPointSetFromCurveFitter()

    def setNumSamples(self, numSamples):
        if False:
            for i in range(10):
                print('nop')
        self.numSamples = int(numSamples)

    def sampleCurve(self, fCompute=1, curveName=None):
        if False:
            i = 10
            return i + 15
        if self.curveCollection == None:
            print('MopathRecorder.sampleCurve: Must define curve first')
            return
        self.curveFitter.reset()
        self.curveFitter.sample(self.curveCollection, self.numSamples)
        if fCompute:
            self.computeCurves()
        self.extractPointSetFromCurveFitter(curveName)

    def makeEven(self):
        if False:
            i = 10
            return i + 15
        self.curveCollection.makeEven(self.maxT, 2)
        self.extractPointSetFromCurveCollection()

    def faceForward(self):
        if False:
            while True:
                i = 10
        self.curveCollection.faceForward(2)
        self.extractPointSetFromCurveCollection()

    def setPathDuration(self, event):
        if False:
            i = 10
            return i + 15
        newMaxT = float(self.getWidget('Resample', 'Path Duration').get())
        self.setPathDurationTo(newMaxT)

    def setPathDurationTo(self, newMaxT):
        if False:
            for i in range(10):
                print('nop')
        sf = newMaxT / self.maxT
        self.curveCollection.resetMaxT(newMaxT)
        oldPointSet = self.pointSet
        self.createNewPointSet()
        self.curveFitter.reset()
        for (time, pos, hpr) in oldPointSet:
            newTime = time * sf
            self.pointSet.append([newTime, Point3(pos), Point3(hpr)])
            self.curveFitter.addXyzHpr(newTime, pos, hpr)
        self.updateWidgets()

    def setRecordStart(self, value):
        if False:
            i = 10
            return i + 15
        self.recordStart = value
        if self.fAdjustingValues:
            return
        self.fAdjustingValues = 1
        self.getWidget('Refine Page', 'Refine From').set(self.recordStart)
        self.getWidget('Extend Page', 'Extend From').set(self.recordStart)
        if self.recordStart > self.controlStart:
            self.getWidget('Refine Page', 'Control Start').set(self.recordStart)
            self.getWidget('Extend Page', 'Control Start').set(self.recordStart)
        if self.recordStart > self.controlStop:
            self.getWidget('Refine Page', 'Control Stop').set(self.recordStart)
        if self.recordStart > self.recordStop:
            self.getWidget('Refine Page', 'Refine To').set(self.recordStart)
        self.getWidget('Playback', 'Time').set(value)
        self.fAdjustingValues = 0

    def getPrePoints(self, type='Refine'):
        if False:
            print('Hello World!')
        self.setRecordingType(type)
        self.prePoints = []
        for i in range(len(self.pointSet)):
            if self.recordStart < self.pointSet[i][0]:
                self.prePoints = self.pointSet[:i - 1]
                break

    def setControlStart(self, value):
        if False:
            i = 10
            return i + 15
        self.controlStart = value
        if self.fAdjustingValues:
            return
        self.fAdjustingValues = 1
        self.getWidget('Refine Page', 'Control Start').set(self.controlStart)
        self.getWidget('Extend Page', 'Control Start').set(self.controlStart)
        if self.controlStart < self.recordStart:
            self.getWidget('Refine Page', 'Refine From').set(self.controlStart)
            self.getWidget('Extend Page', 'Extend From').set(self.controlStart)
        if self.controlStart > self.controlStop:
            self.getWidget('Refine Page', 'Control Stop').set(self.controlStart)
        if self.controlStart > self.recordStop:
            self.getWidget('Refine Page', 'Refine To').set(self.controlStart)
        self.getWidget('Playback', 'Time').set(value)
        self.fAdjustingValues = 0

    def setControlStop(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.controlStop = value
        if self.fAdjustingValues:
            return
        self.fAdjustingValues = 1
        if self.controlStop < self.recordStart:
            self.getWidget('Refine Page', 'Refine From').set(self.controlStop)
        if self.controlStop < self.controlStart:
            self.getWidget('Refine Page', 'Control Start').set(self.controlStop)
        if self.controlStop > self.recordStop:
            self.getWidget('Refine Page', 'Refine To').set(self.controlStop)
        self.getWidget('Playback', 'Time').set(value)
        self.fAdjustingValues = 0

    def setRefineStop(self, value):
        if False:
            while True:
                i = 10
        self.recordStop = value
        if self.fAdjustingValues:
            return
        self.fAdjustingValues = 1
        if self.recordStop < self.recordStart:
            self.getWidget('Refine Page', 'Refine From').set(self.recordStop)
        if self.recordStop < self.controlStart:
            self.getWidget('Refine Page', 'Control Start').set(self.recordStop)
        if self.recordStop < self.controlStop:
            self.getWidget('Refine Page', 'Control Stop').set(self.recordStop)
        self.getWidget('Playback', 'Time').set(value)
        self.fAdjustingValues = 0

    def getPostPoints(self):
        if False:
            for i in range(10):
                print('nop')
        self.setRefineMode()
        self.postPoints = []
        for i in range(len(self.pointSet)):
            if self.recordStop < self.pointSet[i][0]:
                self.postPoints = self.pointSet[i:]
                break

    def mergePoints(self):
        if False:
            while True:
                i = 10
        self.pointSet[0:0] = self.prePoints
        for (time, pos, hpr) in self.prePoints:
            self.curveFitter.addXyzHpr(time, pos, hpr)
        endTime = self.pointSet[-1][0]
        for (time, pos, hpr) in self.postPoints:
            adjustedTime = endTime + (time - self.recordStop)
            self.pointSet.append([adjustedTime, pos, hpr])
            self.curveFitter.addXyzHpr(adjustedTime, pos, hpr)

    def setCropFrom(self, value):
        if False:
            while True:
                i = 10
        self.cropFrom = value
        if self.fAdjustingValues:
            return
        self.fAdjustingValues = 1
        if self.cropFrom > self.cropTo:
            self.getWidget('Crop Page', 'Crop To').set(self.cropFrom)
        self.getWidget('Playback', 'Time').set(value)
        self.fAdjustingValues = 0

    def setCropTo(self, value):
        if False:
            print('Hello World!')
        self.cropTo = value
        if self.fAdjustingValues:
            return
        self.fAdjustingValues = 1
        if self.cropTo < self.cropFrom:
            self.getWidget('Crop Page', 'Crop From').set(self.cropTo)
        self.getWidget('Playback', 'Time').set(value)
        self.fAdjustingValues = 0

    def cropCurve(self):
        if False:
            i = 10
            return i + 15
        if self.pointSet == None:
            print('Empty Point Set')
            return
        oldPoints = self.pointSet
        self.createNewPointSet()
        self.curveFitter.reset()
        pos = Point3(0)
        hpr = Point3(0)
        self.curveCollection.evaluate(self.cropFrom, pos, hpr)
        self.curveFitter.addXyzHpr(0.0, pos, hpr)
        for (time, pos, hpr) in oldPoints:
            if time > self.cropFrom and time < self.cropTo:
                t = time - self.cropFrom
                self.curveFitter.addXyzHpr(t, pos, hpr)
                self.pointSet.append([t, pos, hpr])
        pos = Vec3(0)
        hpr = Vec3(0)
        self.curveCollection.evaluate(self.cropTo, pos, hpr)
        self.curveFitter.addXyzHpr(self.cropTo - self.cropFrom, pos, hpr)
        self.computeCurves()

    def loadCurveFromFile(self):
        if False:
            print('Hello World!')
        mPath = getModelPath()
        if mPath.getNumDirectories() > 0:
            if repr(mPath.getDirectory(0)) == '.':
                path = '.'
            else:
                path = mPath.getDirectory(0).toOsSpecific()
        else:
            path = '.'
        if not os.path.isdir(path):
            print('MopathRecorder Info: Empty Model Path!')
            print('Using current directory')
            path = '.'
        mopathFilename = askopenfilename(defaultextension='.egg', filetypes=(('Egg Files', '*.egg'), ('Bam Files', '*.bam'), ('All files', '*')), initialdir=path, title='Load Nurbs Curve', parent=self.parent)
        if mopathFilename:
            self.reset()
            nodePath = loader.loadModel(Filename.fromOsSpecific(mopathFilename))
            self.curveCollection = ParametricCurveCollection()
            self.curveCollection.addCurves(nodePath.node())
            nodePath.removeNode()
            if self.curveCollection:
                self.nurbsCurveDrawer.setCurves(self.curveCollection)
                self.nurbsCurveDrawer.draw()
                self.extractPointSetFromCurveCollection()
            else:
                self.reset()

    def saveCurveToFile(self):
        if False:
            return 10
        mPath = getModelPath()
        if mPath.getNumDirectories() > 0:
            if repr(mPath.getDirectory(0)) == '.':
                path = '.'
            else:
                path = mPath.getDirectory(0).toOsSpecific()
        else:
            path = '.'
        if not os.path.isdir(path):
            print('MopathRecorder Info: Empty Model Path!')
            print('Using current directory')
            path = '.'
        mopathFilename = asksaveasfilename(defaultextension='.egg', filetypes=(('Egg Files', '*.egg'), ('Bam Files', '*.bam'), ('All files', '*')), initialdir=path, title='Save Nurbs Curve as', parent=self.parent)
        if mopathFilename:
            self.curveCollection.writeEgg(Filename(mopathFilename))

    def followTerrain(self, height=1.0):
        if False:
            return 10
        self.iRay.rayCollisionNodePath.reparentTo(self.nodePath)
        entry = self.iRay.pickGeom3D()
        if entry:
            fromNodePath = entry.getFromNodePath()
            hitPtDist = Vec3(entry.getSurfacePoint(fromNodePath))
            self.nodePath.setZ(self.nodePath, height - hitPtDist)
        self.iRay.rayCollisionNodePath.reparentTo(self.recorderNodePath)

    def addWidget(self, widget, category, text):
        if False:
            return 10
        self.widgetDict[category + '-' + text] = widget

    def getWidget(self, category, text):
        if False:
            for i in range(10):
                print('nop')
        return self.widgetDict[category + '-' + text]

    def getVariable(self, category, text):
        if False:
            for i in range(10):
                print('nop')
        return self.variableDict[category + '-' + text]

    def createLabeledEntry(self, parent, category, text, balloonHelp, value='', command=None, relief='sunken', side=tkinter.LEFT, expand=1, width=12):
        if False:
            print('Hello World!')
        frame = Frame(parent)
        variable = StringVar()
        variable.set(value)
        label = Label(frame, text=text)
        label.pack(side=tkinter.LEFT, fill=tkinter.X)
        self.bind(label, balloonHelp)
        self.widgetDict[category + '-' + text + '-Label'] = label
        entry = Entry(frame, width=width, relief=relief, textvariable=variable)
        entry.pack(side=tkinter.LEFT, fill=tkinter.X, expand=expand)
        self.bind(entry, balloonHelp)
        self.widgetDict[category + '-' + text] = entry
        self.variableDict[category + '-' + text] = variable
        if command:
            entry.bind('<Return>', command)
        frame.pack(side=side, fill=tkinter.X, expand=expand)
        return (frame, label, entry)

    def createButton(self, parent, category, text, balloonHelp, command, side='top', expand=0, fill=tkinter.X):
        if False:
            i = 10
            return i + 15
        widget = Button(parent, text=text)
        widget['command'] = command
        widget.pack(side=side, fill=fill, expand=expand)
        self.bind(widget, balloonHelp)
        self.widgetDict[category + '-' + text] = widget
        return widget

    def createCheckbutton(self, parent, category, text, balloonHelp, command, initialState, side='top', fill=tkinter.X, expand=0):
        if False:
            i = 10
            return i + 15
        bool = BooleanVar()
        bool.set(initialState)
        widget = Checkbutton(parent, text=text, anchor=tkinter.W, variable=bool)
        widget['command'] = command
        widget.pack(side=side, fill=fill, expand=expand)
        self.bind(widget, balloonHelp)
        self.widgetDict[category + '-' + text] = widget
        self.variableDict[category + '-' + text] = bool
        return widget

    def createRadiobutton(self, parent, side, category, text, balloonHelp, variable, value, command=None, fill=tkinter.X, expand=0):
        if False:
            for i in range(10):
                print('nop')
        widget = Radiobutton(parent, text=text, anchor=tkinter.W, variable=variable, value=value)
        widget['command'] = command
        widget.pack(side=side, fill=fill, expand=expand)
        self.bind(widget, balloonHelp)
        self.widgetDict[category + '-' + text] = widget
        return widget

    def createFloater(self, parent, category, text, balloonHelp, command=None, min=0.0, resolution=None, maxVelocity=10.0, **kw):
        if False:
            i = 10
            return i + 15
        kw['text'] = text
        kw['min'] = min
        kw['maxVelocity'] = maxVelocity
        kw['resolution'] = resolution
        widget = Floater(parent, **kw)
        widget['command'] = command
        widget.pack(fill=tkinter.X)
        self.bind(widget, balloonHelp)
        self.widgetDict[category + '-' + text] = widget
        return widget

    def createAngleDial(self, parent, category, text, balloonHelp, command=None, **kw):
        if False:
            return 10
        kw['text'] = text
        widget = AngleDial(parent, **kw)
        widget['command'] = command
        widget.pack(fill=tkinter.X)
        self.bind(widget, balloonHelp)
        self.widgetDict[category + '-' + text] = widget
        return widget

    def createSlider(self, parent, category, text, balloonHelp, command=None, min=0.0, max=1.0, resolution=None, side=tkinter.TOP, fill=tkinter.X, expand=1, **kw):
        if False:
            return 10
        kw['text'] = text
        kw['min'] = min
        kw['max'] = max
        kw['resolution'] = resolution
        widget = Slider(parent, **kw)
        widget['command'] = command
        widget.pack(side=side, fill=fill, expand=expand)
        self.bind(widget, balloonHelp)
        self.widgetDict[category + '-' + text] = widget
        return widget

    def createEntryScale(self, parent, category, text, balloonHelp, command=None, min=0.0, max=1.0, resolution=None, side=tkinter.TOP, fill=tkinter.X, expand=1, **kw):
        if False:
            print('Hello World!')
        kw['text'] = text
        kw['min'] = min
        kw['max'] = max
        kw['resolution'] = resolution
        widget = EntryScale(parent, **kw)
        widget['command'] = command
        widget.pack(side=side, fill=fill, expand=expand)
        self.bind(widget, balloonHelp)
        self.widgetDict[category + '-' + text] = widget
        return widget

    def createVector2Entry(self, parent, category, text, balloonHelp, command=None, **kw):
        if False:
            return 10
        kw['text'] = text
        widget = Vector2Entry(parent, **kw)
        widget['command'] = command
        widget.pack(fill=tkinter.X)
        self.bind(widget, balloonHelp)
        self.widgetDict[category + '-' + text] = widget
        return widget

    def createVector3Entry(self, parent, category, text, balloonHelp, command=None, **kw):
        if False:
            for i in range(10):
                print('nop')
        kw['text'] = text
        widget = Vector3Entry(parent, **kw)
        widget['command'] = command
        widget.pack(fill=tkinter.X)
        self.bind(widget, balloonHelp)
        self.widgetDict[category + '-' + text] = widget
        return widget

    def createColorEntry(self, parent, category, text, balloonHelp, command=None, **kw):
        if False:
            i = 10
            return i + 15
        kw['text'] = text
        widget = ColorEntry(parent, **kw)
        widget['command'] = command
        widget.pack(fill=tkinter.X)
        self.bind(widget, balloonHelp)
        self.widgetDict[category + '-' + text] = widget
        return widget

    def createOptionMenu(self, parent, category, text, balloonHelp, items, command):
        if False:
            i = 10
            return i + 15
        optionVar = StringVar()
        if len(items) > 0:
            optionVar.set(items[0])
        widget = Pmw.OptionMenu(parent, labelpos=tkinter.W, label_text=text, label_width=12, menu_tearoff=1, menubutton_textvariable=optionVar, items=items)
        widget['command'] = command
        widget.pack(fill=tkinter.X)
        self.bind(widget.component('menubutton'), balloonHelp)
        self.widgetDict[category + '-' + text] = widget
        self.variableDict[category + '-' + text] = optionVar
        return optionVar

    def createComboBox(self, parent, category, text, balloonHelp, items, command, history=0, side=tkinter.LEFT, expand=0, fill=tkinter.X):
        if False:
            print('Hello World!')
        widget = Pmw.ComboBox(parent, labelpos=tkinter.W, label_text=text, label_anchor='e', label_width=12, entry_width=16, history=history, scrolledlist_items=items)
        widget.configure(entryfield_entry_state='disabled')
        if len(items) > 0:
            widget.selectitem(items[0])
        widget['selectioncommand'] = command
        widget.pack(side=side, fill=fill, expand=expand)
        self.bind(widget, balloonHelp)
        self.widgetDict[category + '-' + text] = widget
        return widget

    def makeCameraWindow(self):
        if False:
            print('Hello World!')
        chan = base.win.getChannel(0)
        self.cLayer = chan.makeLayer(1)
        self.layerIndex = 1
        self.cDr = self.cLayer.makeDisplayRegion(0.6, 1.0, 0, 0.4)
        self.cDr.setClearDepthActive(1)
        self.cDr.setClearColorActive(1)
        self.cDr.setClearColor(Vec4(0))
        self.cCamera = render.attachNewNode('cCamera')
        self.cCamNode = Camera('cCam')
        self.cLens = PerspectiveLens()
        self.cLens.setFov(40, 40)
        self.cLens.setNear(0.1)
        self.cLens.setFar(100.0)
        self.cCamNode.setLens(self.cLens)
        self.cCamNode.setScene(render)
        self.cCam = self.cCamera.attachNewNode(self.cCamNode)
        self.cDr.setCamera(self.cCam)

    def toggleWidgetVis(self):
        if False:
            print('Hello World!')
        messenger.send('SEditor-ToggleWidgetVis')
        SEditor.toggleWidgetVis()

    def bindMotionPathToNode(self):
        if False:
            print('Hello World!')
        if self.curveCollection == None:
            print('----Error: you need to select or create a curve first!')
            return
        self.accept('MP_checkName', self.bindMotionPath)
        self.askName = namePathPanel(MopathRecorder.count)
        return

    def bindMotionPath(self, name=None, test=None):
        if False:
            while True:
                i = 10
        print(test)
        self.ignore('MP_checkName')
        del self.askName
        self.curveCollection.getCurve(0).setName(name)
        comboBox = self.getWidget('Mopath', 'Path:')
        oldName = comboBox.get()
        self.pointSetDict[name] = self.pointSetDict[oldName]
        del self.pointSetDict[oldName]
        scrolledList = comboBox.component('scrolledlist')
        listbox = scrolledList.component('listbox')
        names = list(listbox.get(0, 'end'))
        num = names.index(oldName)
        names.pop(num)
        names.append(name)
        scrolledList.setlist(names)
        comboBox.selectitem(name)
        messenger.send('mPath_bindPathToNode', [self.playbackNodePath, self.curveCollection])
        return

    def addCurvesFromNodepath(self, curveList):
        if False:
            print('Hello World!')
        'addCurvesFromNodepath(self,curveList)\n        This function will take a curveCollection list as a input.\n        If the list is not None, it will put the vurve back into the curve list.\n        else, do nothing.\n        '
        print(curveList)
        self.ignore('curveListFor' + self.name)
        if curveList != None:
            for collection in curveList:
                self.curveCollection = collection
                self.extractPointSetFromCurveCollection(curveName=self.curveCollection.getCurve(0).getName())
        else:
            pass
        return

class namePathPanel(AppShell):
    appname = 'Name the Path'
    frameWidth = 575
    frameHeight = 200
    usecommandarea = 0
    usestatusarea = 0
    index = 0

    def __init__(self, count, parent=None, **kw):
        if False:
            return 10
        INITOPT = Pmw.INITOPT
        self.id = 'Name the Path'
        self.appname = self.id
        optiondefs = (('title', self.appname, None),)
        self.defineoptions(kw, optiondefs)
        AppShell.__init__(self)
        self.parent.resizable(False, False)
        self.initialiseoptions(namePathPanel)

    def createInterface(self):
        if False:
            while True:
                i = 10
        self.menuBar.destroy()
        interior = self.interior()
        mainFrame = Frame(interior)
        dataFrame = Frame(mainFrame)
        label = Label(dataFrame, text='This name will be used as a reference for this Path.', font=('MSSansSerif', 10))
        label.pack(side=tkinter.TOP, expand=0, fill=tkinter.X)
        dataFrame.pack(side=tkinter.TOP, expand=0, fill=tkinter.X, padx=5, pady=10)
        dataFrame = Frame(mainFrame)
        self.inputZone = Pmw.EntryField(dataFrame, labelpos='w', label_text='Name Selected Path: ', entry_font=('MSSansSerif', 10), label_font=('MSSansSerif', 10), validate=None, entry_width=20)
        self.inputZone.pack(side=tkinter.LEFT, fill=tkinter.X, expand=0)
        self.button_ok = Button(dataFrame, text='OK', command=self.ok_press, width=10)
        self.button_ok.pack(fill=tkinter.X, expand=0, side=tkinter.LEFT, padx=3)
        dataFrame.pack(side=tkinter.TOP, expand=0, fill=tkinter.X, padx=10, pady=10)
        mainFrame.pack(expand=1, fill=tkinter.BOTH)

    def onDestroy(self, event):
        if False:
            return 10
        '\n        If you have open any thing, please rewrite here!\n        '
        pass

    def ok_press(self):
        if False:
            print('Hello World!')
        name = self.inputZone.getvalue()
        messenger.send('MP_checkName', [name])
        self.quit()
        return