""" This module contains `.ShowBase`, an application framework responsible
for opening a graphical display, setting up input devices and creating
the scene graph.

The simplest way to open a ShowBase instance is to execute this code:

.. code-block:: python

   from direct.showbase.ShowBase import ShowBase

   base = ShowBase()
   base.run()

A common approach is to create your own subclass inheriting from ShowBase.

Built-in global variables
-------------------------

Some key variables used in all Panda3D scripts are actually attributes of the
ShowBase instance.  When creating an instance of this class, it will write many
of these variables to the built-in scope of the Python interpreter, so that
they are accessible to any Python module, without the need for extra imports.
For example, the ShowBase instance itself is accessible anywhere through the
:data:`~builtins.base` variable.

While these are handy for prototyping, we do not recommend using them in bigger
projects, as it can make the code confusing to read to other Python developers,
to whom it may not be obvious where these variables are originating.

Refer to the :mod:`builtins` page for a listing of the variables written to the
built-in scope.

"""
__all__ = ['ShowBase', 'WindowControls']
from panda3d.core import AntialiasAttrib, AudioManager, AudioSound, BitMask32, ButtonThrower, Camera, ClockObject, CollisionTraverser, ColorBlendAttrib, ConfigPageManager, ConfigVariableBool, ConfigVariableDouble, ConfigVariableFilename, ConfigVariableInt, ConfigVariableManager, ConfigVariableString, DataGraphTraverser, DepthTestAttrib, DepthWriteAttrib, DriveInterface, ExecutionEnvironment, Filename, FisheyeMaker, FrameBufferProperties, FrameRateMeter, GeomNode, GraphicsEngine, GraphicsOutput, GraphicsPipe, GraphicsPipeSelection, GraphicsWindow, InputDeviceManager, InputDeviceNode, KeyboardButton, LensNode, Mat4, ModelNode, ModifierButtons, MouseAndKeyboard, MouseRecorder, MouseWatcher, NodePath, Notify, OrthographicLens, PandaNode, PandaSystem, PerspectiveLens, PGMouseWatcherBackground, PGTop, PNMImage, PStatClient, PythonCallbackObject, RecorderController, RenderModeAttrib, RenderState, RescaleNormalAttrib, SceneGraphAnalyzerMeter, TexGenAttrib, Texture, TextureStage, Thread, Trackball, Transform2SG, TransformState, TrueClock, VBase4, VirtualFileSystem, WindowProperties, getModelPath
from panda3d.direct import throw_new_frame, init_app_for_gui
from panda3d.direct import storeAccessibilityShortcutKeys, allowAccessibilityShortcutKeys
from . import DConfig
from direct.extensions_native import NodePath_extensions
import sys
import builtins
builtins.config = DConfig
from direct.directnotify.DirectNotifyGlobal import directNotify, giveNotify
from direct.directnotify.Notifier import Notifier
from .MessengerGlobal import messenger
from .BulletinBoardGlobal import bulletinBoard
from direct.task.TaskManagerGlobal import taskMgr
from .JobManagerGlobal import jobMgr
from .EventManagerGlobal import eventMgr
from .PythonUtil import Stack
from direct.interval import IntervalManager
from direct.showbase.BufferViewer import BufferViewer
from direct.task import Task
from . import Loader
import time
import atexit
import importlib
from direct.showbase import ExceptionVarDump
from . import DirectObject
from . import SfxPlayer
from typing import ClassVar, Optional
if __debug__:
    from direct.showbase import GarbageReport
    from direct.directutil import DeltaProfiler
    from . import OnScreenDebug
    import warnings

@atexit.register
def exitfunc():
    if False:
        print('Hello World!')
    if getattr(builtins, 'base', None) is not None:
        builtins.base.destroy()

class ShowBase(DirectObject.DirectObject):
    config: ClassVar = DConfig
    notify: ClassVar[Notifier] = directNotify.newCategory('ShowBase')
    guiItems: ClassVar[dict]
    render2d: NodePath
    aspect2d: NodePath
    pixel2d: NodePath

    def __init__(self, fStartDirect=True, windowType=None):
        if False:
            return 10
        "Opens a window, sets up a 3-D and several 2-D scene graphs, and\n        everything else needed to render the scene graph to the window.\n\n        To prevent a window from being opened, set windowType to the string\n        'none' (or 'offscreen' to create an offscreen buffer).  If this is not\n        specified, the default value is taken from the 'window-type'\n        configuration variable.\n\n        This constructor will add various things to the Python builtins scope,\n        including this instance itself (under the name ``base``).\n        "
        from . import ShowBaseGlobal
        self.__dev__ = ShowBaseGlobal.__dev__
        builtins.__dev__ = self.__dev__
        logStackDump = ConfigVariableBool('log-stack-dump', False).value or ConfigVariableBool('client-log-stack-dump', False).value
        uploadStackDump = ConfigVariableBool('upload-stack-dump', False).value
        if logStackDump or uploadStackDump:
            ExceptionVarDump.install(logStackDump, uploadStackDump)
        if __debug__:
            self.__autoGarbageLogging = self.__dev__ and ConfigVariableBool('auto-garbage-logging', False)
        self.mainDir = ExecutionEnvironment.getEnvironmentVariable('MAIN_DIR')
        self.main_dir = self.mainDir
        self.appRunner = None
        self.app_runner = self.appRunner
        self.debugRunningMultiplier = 4
        if ConfigVariableBool('disable-sticky-keys', False):
            storeAccessibilityShortcutKeys()
            allowAccessibilityShortcutKeys(False)
            self.__disabledStickyKeys = True
        else:
            self.__disabledStickyKeys = False
        self.printEnvDebugInfo()
        vfs = VirtualFileSystem.getGlobalPtr()
        self.nextWindowIndex = 1
        self.__directStarted = False
        self.__deadInputs = 0
        self.sfxActive = ConfigVariableBool('audio-sfx-active', True).value
        self.musicActive = ConfigVariableBool('audio-music-active', True).value
        self.wantFog = ConfigVariableBool('want-fog', True).value
        self.wantRender2dp = ConfigVariableBool('want-render2dp', True).value
        self.screenshotExtension = ConfigVariableString('screenshot-extension', 'jpg').value
        self.musicManager = None
        self.musicManagerIsValid = None
        self.sfxManagerList = []
        self.sfxManagerIsValidList = []
        self.wantStats = ConfigVariableBool('want-pstats', False).value
        self.wantTk = False
        self.wantWx = False
        self.wantDirect = False
        self.exitFunc = None
        self.finalExitCallbacks = []
        taskMgr.resumeFunc = PStatClient.resumeAfterPause
        if self.__dev__:
            self.__setupProfile()
        self.__configAspectRatio = ConfigVariableDouble('aspect-ratio', 0).value
        self.__oldAspectRatio = None
        self.windowType = windowType
        if self.windowType is None:
            self.windowType = ConfigVariableString('window-type', 'onscreen').value
        self.requireWindow = ConfigVariableBool('require-window', True).value
        self.win = None
        self.frameRateMeter = None
        self.sceneGraphAnalyzerMeter = None
        self.winList = []
        self.winControls = []
        self.mainWinMinimized = 0
        self.mainWinForeground = 0
        self.pipe = None
        self.pipeList = []
        self.mouse2cam = None
        self.buttonThrowers = None
        self.mouseWatcher = None
        self.mouseWatcherNode = None
        self.pointerWatcherNodes = None
        self.mouseInterface = None
        self.drive = None
        self.trackball = None
        self.texmem = None
        self.showVertices = None
        self.deviceButtonThrowers = []
        self.cam = None
        self.cam2d = None
        self.cam2dp = None
        self.camera = None
        self.camera2d = None
        self.camera2dp = None
        self.camList = []
        self.camNode = None
        self.camLens = None
        self.camFrustumVis = None
        self.direct = None
        self.wxApp = None
        self.wxAppCreated = False
        self.tkRoot = None
        self.tkRootCreated = False
        if hasattr(builtins, 'clusterSyncFlag'):
            self.clusterSyncFlag = builtins.clusterSyncFlag
        else:
            self.clusterSyncFlag = ConfigVariableBool('cluster-sync', False)
        self.hidden = ShowBaseGlobal.hidden
        self.graphicsEngine = GraphicsEngine.getGlobalPtr()
        self.graphics_engine = self.graphicsEngine
        self.setupRender()
        self.setupRender2d()
        self.setupDataGraph()
        if self.wantRender2dp:
            self.setupRender2dp()
        self.cTrav = 0
        self.shadowTrav = 0
        self.cTravStack = Stack()
        self.appTrav = 0
        self.dgTrav = DataGraphTraverser()
        self.recorder = None
        playbackSession = ConfigVariableFilename('playback-session', '')
        recordSession = ConfigVariableFilename('record-session', '')
        if not playbackSession.empty():
            self.recorder = RecorderController()
            self.recorder.beginPlayback(playbackSession.value)
        elif not recordSession.empty():
            self.recorder = RecorderController()
            self.recorder.beginRecord(recordSession.value)
        if self.recorder:
            import random
            seed = self.recorder.getRandomSeed()
            random.seed(seed)
        if sys.platform == 'darwin':
            if ConfigVariableBool('want-wx', False):
                wx = importlib.import_module('wx')
                self.wxApp = wx.App()
            if ConfigVariableBool('want-tk', False):
                Pmw = importlib.import_module('Pmw')
                self.tkRoot = Pmw.initialise()
        if self.windowType != 'none':
            props = WindowProperties.getDefault()
            if ConfigVariableBool('read-raw-mice', False):
                props.setRawMice(1)
            self.openDefaultWindow(startDirect=False, props=props)
        self.mouseInterface = self.trackball
        self.useTrackball()
        self.loader = Loader.Loader(self)
        self.graphicsEngine.setDefaultLoader(self.loader.loader)
        ShowBaseGlobal.loader = self.loader
        self.eventMgr = eventMgr
        self.messenger = messenger
        self.bboard = bulletinBoard
        self.taskMgr = taskMgr
        self.task_mgr = taskMgr
        self.jobMgr = jobMgr
        self.particleMgr = None
        self.particleMgrEnabled = 0
        self.physicsMgr = None
        self.physicsMgrEnabled = 0
        self.physicsMgrAngular = 0
        self.devices = InputDeviceManager.getGlobalPtr()
        self.__inputDeviceNodes = {}
        self.createStats()
        self.AppHasAudioFocus = 1
        clock = ClockObject.getGlobalClock()
        self.clock = clock
        trueClock = TrueClock.getGlobalPtr()
        clock.setRealTime(trueClock.getShortTime())
        clock.tick()
        taskMgr.globalClock = clock
        affinityMask = ConfigVariableInt('client-cpu-affinity-mask', -1).value
        if affinityMask != -1:
            TrueClock.getGlobalPtr().setCpuAffinity(affinityMask)
        else:
            autoAffinity = ConfigVariableBool('auto-single-cpu-affinity', False).value
            affinity = None
            if autoAffinity and hasattr(builtins, 'clientIndex'):
                affinity = abs(int(builtins.clientIndex))
            else:
                affinity = ConfigVariableInt('client-cpu-affinity', -1).value
            if affinity in (None, -1) and autoAffinity:
                affinity = 0
            if affinity not in (None, -1):
                TrueClock.getGlobalPtr().setCpuAffinity(1 << affinity % 32)
        if hasattr(builtins, 'base'):
            raise Exception('Attempt to spawn multiple ShowBase instances!')
        builtins.base = self
        builtins.render2d = self.render2d
        builtins.aspect2d = self.aspect2d
        builtins.pixel2d = self.pixel2d
        builtins.render = self.render
        builtins.hidden = self.hidden
        builtins.camera = self.camera
        builtins.loader = self.loader
        builtins.taskMgr = self.taskMgr
        builtins.jobMgr = self.jobMgr
        builtins.eventMgr = self.eventMgr
        builtins.messenger = self.messenger
        builtins.bboard = self.bboard
        builtins.ostream = Notify.out()
        builtins.directNotify = directNotify
        builtins.giveNotify = giveNotify
        builtins.globalClock = clock
        builtins.vfs = vfs
        builtins.cpMgr = ConfigPageManager.getGlobalPtr()
        builtins.cvMgr = ConfigVariableManager.getGlobalPtr()
        builtins.pandaSystem = PandaSystem.getGlobalPtr()
        if __debug__:
            builtins.deltaProfiler = DeltaProfiler.DeltaProfiler('ShowBase')
            self.onScreenDebug = OnScreenDebug.OnScreenDebug()
            builtins.onScreenDebug = self.onScreenDebug
        if self.wantRender2dp:
            builtins.render2dp = self.render2dp
            builtins.aspect2dp = self.aspect2dp
            builtins.pixel2dp = self.pixel2dp
        builtins.run = ShowBaseGlobal.run
        ShowBaseGlobal.base = self
        ShowBaseGlobal.__dev__ = self.__dev__
        if self.__dev__:
            ShowBase.notify.debug('__dev__ == %s' % self.__dev__)
        else:
            ShowBase.notify.info('__dev__ == %s' % self.__dev__)
        self.createBaseAudioManagers()
        if self.__dev__ and ConfigVariableBool('track-gui-items', False):
            if not hasattr(ShowBase, 'guiItems'):
                ShowBase.guiItems = {}
        if ConfigVariableBool('orig-gui-sounds', False).value:
            from direct.gui import DirectGuiGlobals as DGG
            DGG.setDefaultClickSound(self.loader.loadSfx('audio/sfx/GUI_click.wav'))
            DGG.setDefaultRolloverSound(self.loader.loadSfx('audio/sfx/GUI_rollover.wav'))
        self.__directObject = DirectObject.DirectObject()
        self.__prevWindowProperties = None
        self.__directObject.accept('window-event', self.windowEvent)
        from . import Transitions
        self.transitions = Transitions.Transitions(self.loader)
        if self.win:
            self.setupWindowControls()
        sleepTime = ConfigVariableDouble('client-sleep', 0.0)
        self.clientSleep = 0.0
        self.setSleep(sleepTime.value)
        self.multiClientSleep = ConfigVariableBool('multi-sleep', False)
        self.bufferViewer = BufferViewer(self.win, self.render2dp if self.wantRender2dp else self.render2d)
        if self.windowType != 'none':
            if fStartDirect:
                self.__doStartDirect()
            if ConfigVariableBool('show-tex-mem', False):
                if not self.texmem or self.texmem.cleanedUp:
                    self.toggleTexMem()
        taskMgr.finalInit()
        self.restart()

    def pushCTrav(self, cTrav):
        if False:
            print('Hello World!')
        self.cTravStack.push(self.cTrav)
        self.cTrav = cTrav

    def popCTrav(self):
        if False:
            print('Hello World!')
        self.cTrav = self.cTravStack.pop()

    def __setupProfile(self):
        if False:
            print('Hello World!')
        ' Sets up the Python profiler, if available, according to\n        some Panda config settings. '
        try:
            profile = importlib.import_module('profile')
            pstats = importlib.import_module('pstats')
        except ImportError:
            return
        profile.Profile.bias = ConfigVariableDouble('profile-bias', 0.0).value

        def f8(x):
            if False:
                for i in range(10):
                    print('nop')
            return ('%' + '8.%df' % ConfigVariableInt('profile-decimals', 3)) % x
        pstats.f8 = f8

    def getExitErrorCode(self):
        if False:
            return 10
        return 0

    def printEnvDebugInfo(self):
        if False:
            while True:
                i = 10
        'Print some information about the environment that we are running\n        in.  Stuff like the model paths and other paths.  Feel free to\n        add stuff to this.\n        '
        if ConfigVariableBool('want-env-debug-info', False):
            print('\n\nEnvironment Debug Info {')
            print('* model path:')
            print(getModelPath())
            print('}')

    def destroy(self):
        if False:
            i = 10
            return i + 15
        ' Call this function to destroy the ShowBase and stop all\n        its tasks, freeing all of the Panda resources.  Normally, you\n        should not need to call it explicitly, as it is bound to the\n        exitfunc and will be called at application exit time\n        automatically.\n\n        This function is designed to be safe to call multiple times.\n\n        When called from a thread other than the main thread, this will create\n        a task to schedule the destroy on the main thread, and wait for this to\n        complete.\n        '
        if Thread.getCurrentThread() != Thread.getMainThread():
            task = taskMgr.add(self.destroy, extraArgs=[])
            task.wait()
            return
        for cb in self.finalExitCallbacks[:]:
            cb()
        if getattr(builtins, 'base', None) is self:
            del builtins.run
            del builtins.base
            del builtins.loader
            del builtins.taskMgr
            ShowBaseGlobal = sys.modules.get('direct.showbase.ShowBaseGlobal', None)
            if ShowBaseGlobal:
                del ShowBaseGlobal.base
        self.aspect2d.node().removeAllChildren()
        self.render2d.node().removeAllChildren()
        self.aspect2d.reparent_to(self.render2d)
        if self.__disabledStickyKeys:
            allowAccessibilityShortcutKeys(True)
            self.__disabledStickyKeys = False
        self.__directObject.ignoreAll()
        self.ignoreAll()
        self.shutdown()
        if getattr(self, 'musicManager', None):
            self.musicManager.shutdown()
            self.musicManager = None
            for sfxManager in self.sfxManagerList:
                sfxManager.shutdown()
            self.sfxManagerList = []
        if getattr(self, 'loader', None):
            self.loader.destroy()
            self.loader = None
        if getattr(self, 'graphicsEngine', None):
            self.graphicsEngine.removeAllWindows()
        try:
            self.direct.panel.destroy()
        except Exception:
            pass
        self.win = None
        self.winList.clear()
        self.pipe = None

    def makeDefaultPipe(self, printPipeTypes=None):
        if False:
            print('Hello World!')
        '\n        Creates the default GraphicsPipe, which will be used to make\n        windows unless otherwise specified.\n        '
        assert self.pipe is None
        if printPipeTypes is None:
            printPipeTypes = ConfigVariableBool('print-pipe-types', True).value
        selection = GraphicsPipeSelection.getGlobalPtr()
        if printPipeTypes:
            selection.printPipeTypes()
        self.pipe = selection.makeDefaultPipe()
        if not self.pipe:
            self.notify.error('No graphics pipe is available!\nYour Config.prc file must name at least one valid panda display\nlibrary via load-display or aux-display.')
        self.notify.info('Default graphics pipe is %s (%s).' % (self.pipe.getType().getName(), self.pipe.getInterfaceName()))
        self.pipeList.append(self.pipe)

    def makeModulePipe(self, moduleName):
        if False:
            i = 10
            return i + 15
        "\n        Returns a GraphicsPipe from the indicated module,\n        e.g. 'pandagl' or 'pandadx9'.  Does not affect base.pipe or\n        base.pipeList.\n\n        :rtype: panda3d.core.GraphicsPipe\n        "
        selection = GraphicsPipeSelection.getGlobalPtr()
        return selection.makeModulePipe(moduleName)

    def makeAllPipes(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates all GraphicsPipes that the system knows about and fill up\n        `pipeList` with them.\n        '
        selection = GraphicsPipeSelection.getGlobalPtr()
        selection.loadAuxModules()
        if self.pipe is None:
            self.makeDefaultPipe()
        numPipeTypes = selection.getNumPipeTypes()
        for i in range(numPipeTypes):
            pipeType = selection.getPipeType(i)
            already = 0
            for pipe in self.pipeList:
                if pipe.getType() == pipeType:
                    already = 1
            if not already:
                pipe = selection.makePipe(pipeType)
                if pipe:
                    self.notify.info('Got aux graphics pipe %s (%s).' % (pipe.getType().getName(), pipe.getInterfaceName()))
                    self.pipeList.append(pipe)
                else:
                    self.notify.info('Could not make graphics pipe %s.' % pipeType.getName())

    def openWindow(self, props=None, fbprops=None, pipe=None, gsg=None, host=None, type=None, name=None, size=None, aspectRatio=None, makeCamera=True, keepCamera=False, scene=None, stereo=None, unexposedDraw=None, callbackWindowDict=None, requireWindow=None):
        if False:
            return 10
        "\n        Creates a window and adds it to the list of windows that are\n        to be updated every frame.\n\n        :param props: the :class:`~panda3d.core.WindowProperties` that\n                      describes the window.\n\n        :param fbprops: the :class:`~panda3d.core.FrameBufferProperties`\n                        indicating the requested framebuffer properties.\n\n        :param type: Either 'onscreen', 'offscreen', or 'none'.\n\n        :param keepCamera: If True, the existing base.cam is set up to\n                           render into the new window.\n\n        :param makeCamera: If True (and keepCamera is False), a new camera is\n                           set up to render into the new window.\n\n        :param unexposedDraw: If not None, it specifies the initial value\n                              of :meth:`~panda3d.core.GraphicsWindow.setUnexposedDraw()`.\n\n        :param callbackWindowDict: If not None, a\n                                   :class:`~panda3d.core.CallbackGraphicsWindow`\n                                   is created instead, which allows the caller\n                                   to create the actual window with its own\n                                   OpenGL context, and direct Panda's rendering\n                                   into that window.\n\n        :param requireWindow: If True, the function should raise an exception\n                              if the window fails to open correctly.\n\n        :rtype: panda3d.core.GraphicsWindow\n        "
        func = lambda : self._doOpenWindow(props=props, fbprops=fbprops, pipe=pipe, gsg=gsg, host=host, type=type, name=name, size=size, aspectRatio=aspectRatio, makeCamera=makeCamera, keepCamera=keepCamera, scene=scene, stereo=stereo, unexposedDraw=unexposedDraw, callbackWindowDict=callbackWindowDict)
        if self.win:
            win = func()
            self.graphicsEngine.openWindows()
            return win
        if type is None:
            type = self.windowType
        if requireWindow is None:
            requireWindow = self.requireWindow
        win = func()
        self.graphicsEngine.openWindows()
        if win is not None and (not win.isValid()):
            self.notify.info('Window did not open, removing.')
            self.closeWindow(win)
            win = None
        if win is None and pipe is None:
            self.makeAllPipes()
            try:
                self.pipeList.remove(self.pipe)
            except ValueError:
                pass
            while self.win is None and self.pipeList:
                self.pipe = self.pipeList[0]
                self.notify.info('Trying pipe type %s (%s)' % (self.pipe.getType(), self.pipe.getInterfaceName()))
                win = func()
                self.graphicsEngine.openWindows()
                if win is not None and (not win.isValid()):
                    self.notify.info('Window did not open, removing.')
                    self.closeWindow(win)
                    win = None
                if win is None:
                    self.pipeList.remove(self.pipe)
        if win is None:
            self.notify.warning("Unable to open '%s' window." % type)
            if requireWindow:
                raise Exception('Could not open window.')
        else:
            self.notify.info('Successfully opened window of type %s (%s)' % (win.getType(), win.getPipe().getInterfaceName()))
        return win

    def _doOpenWindow(self, props=None, fbprops=None, pipe=None, gsg=None, host=None, type=None, name=None, size=None, aspectRatio=None, makeCamera=True, keepCamera=False, scene=None, stereo=None, unexposedDraw=None, callbackWindowDict=None):
        if False:
            while True:
                i = 10
        if pipe is None:
            pipe = self.pipe
            if pipe is None:
                self.makeDefaultPipe()
                pipe = self.pipe
            if pipe is None:
                return None
        if isinstance(gsg, GraphicsOutput):
            host = gsg
            gsg = gsg.getGsg()
        if pipe.getType().getName().startswith('wdx'):
            gsg = None
        if type is None:
            type = self.windowType
        if props is None:
            props = WindowProperties.getDefault()
        if fbprops is None:
            fbprops = FrameBufferProperties.getDefault()
        if size is not None:
            props = WindowProperties(props)
            props.setSize(size[0], size[1])
        if name is None:
            name = 'window%s' % self.nextWindowIndex
            self.nextWindowIndex += 1
        win = None
        flags = GraphicsPipe.BFFbPropsOptional
        if type == 'onscreen':
            flags = flags | GraphicsPipe.BFRequireWindow
        elif type == 'offscreen':
            flags = flags | GraphicsPipe.BFRefuseWindow
        if callbackWindowDict:
            flags = flags | GraphicsPipe.BFRequireCallbackWindow
        if host:
            assert host.isValid()
            win = self.graphicsEngine.makeOutput(pipe, name, 0, fbprops, props, flags, host.getGsg(), host)
        elif gsg:
            win = self.graphicsEngine.makeOutput(pipe, name, 0, fbprops, props, flags, gsg)
        else:
            win = self.graphicsEngine.makeOutput(pipe, name, 0, fbprops, props, flags)
        if win is None:
            return None
        if unexposedDraw is not None and hasattr(win, 'setUnexposedDraw'):
            win.setUnexposedDraw(unexposedDraw)
        if callbackWindowDict:
            for callbackName in ['Events', 'Properties', 'Render']:
                func = callbackWindowDict.get(callbackName, None)
                if not func:
                    continue
                setCallbackName = 'set%sCallback' % callbackName
                setCallback = getattr(win, setCallbackName)
                setCallback(PythonCallbackObject(func))
            for inputName in callbackWindowDict.get('inputDevices', ['mouse']):
                win.createInputDevice(inputName)
        if hasattr(win, 'requestProperties'):
            win.requestProperties(props)
        mainWindow = False
        if self.win is None:
            mainWindow = True
            self.win = win
            if hasattr(self, 'bufferViewer'):
                self.bufferViewer.win = win
        self.winList.append(win)
        if keepCamera:
            self.makeCamera(win, scene=scene, aspectRatio=aspectRatio, stereo=stereo, useCamera=self.cam)
        elif makeCamera:
            self.makeCamera(win, scene=scene, aspectRatio=aspectRatio, stereo=stereo)
        messenger.send('open_window', [win, mainWindow])
        if mainWindow:
            messenger.send('open_main_window')
        return win

    def closeWindow(self, win, keepCamera=False, removeWindow=True):
        if False:
            while True:
                i = 10
        '\n        Closes the indicated window and removes it from the list of\n        windows.  If it is the main window, clears the main window\n        pointer to None.\n        '
        win.setActive(False)
        numRegions = win.getNumDisplayRegions()
        for i in range(numRegions):
            dr = win.getDisplayRegion(i)
            if self.direct is not None:
                for drc in self.direct.drList:
                    if drc.cam == dr.getCamera():
                        self.direct.drList.displayRegionList.remove(drc)
                        break
            cam = NodePath(dr.getCamera())
            dr.setCamera(NodePath())
            if not cam.isEmpty() and cam.node().getNumDisplayRegions() == 0 and (not keepCamera):
                if self.camList.count(cam) != 0:
                    self.camList.remove(cam)
                if cam == self.cam:
                    self.cam = None
                if cam == self.cam2d:
                    self.cam2d = None
                if cam == self.cam2dp:
                    self.cam2dp = None
                cam.removeNode()
        for winCtrl in self.winControls:
            if winCtrl.win == win:
                self.winControls.remove(winCtrl)
                break
        if removeWindow:
            self.graphicsEngine.removeWindow(win)
        self.winList.remove(win)
        mainWindow = False
        if win == self.win:
            mainWindow = True
            self.win = None
            if self.frameRateMeter:
                self.frameRateMeter.clearWindow()
                self.frameRateMeter = None
            if self.sceneGraphAnalyzerMeter:
                self.sceneGraphAnalyzerMeter.clearWindow()
                self.sceneGraphAnalyzerMeter = None
        messenger.send('close_window', [win, mainWindow])
        if mainWindow:
            messenger.send('close_main_window')
        if not self.winList:
            self.graphicsEngine.renderFrame()

    def openDefaultWindow(self, *args, **kw):
        if False:
            return 10
        "\n        Creates the main window for the first time, without being too\n        particular about the kind of graphics API that is chosen.\n        The suggested window type from the load-display config variable is\n        tried first; if that fails, the first window type that can be\n        successfully opened at all is accepted.\n\n        This is intended to be called only once, at application startup.\n        It is normally called automatically unless window-type is configured\n        to 'none'.\n\n        :returns: True on success, False on failure.\n        "
        startDirect = kw.get('startDirect', True)
        if 'startDirect' in kw:
            del kw['startDirect']
        self.openMainWindow(*args, **kw)
        if startDirect:
            self.__doStartDirect()
        return self.win is not None

    def openMainWindow(self, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates the initial, main window for the application, and sets\n        up the mouse and render2d structures appropriately for it.  If\n        this method is called a second time, it will close the\n        previous main window and open a new one, preserving the lens\n        properties in base.camLens.\n\n        :returns: True on success, or False on failure (in which case base.win\n                  may be either None, or the previous, closed window).\n        '
        keepCamera = kw.get('keepCamera', False)
        success = 1
        oldWin = self.win
        oldLens = self.camLens
        oldClearColorActive = None
        if self.win is not None:
            oldClearColorActive = self.win.getClearColorActive()
            oldClearColor = VBase4(self.win.getClearColor())
            oldClearDepthActive = self.win.getClearDepthActive()
            oldClearDepth = self.win.getClearDepth()
            oldClearStencilActive = self.win.getClearStencilActive()
            oldClearStencil = self.win.getClearStencil()
            self.closeWindow(self.win, keepCamera=keepCamera)
        self.openWindow(*args, **kw)
        if self.win is None:
            self.win = oldWin
            self.winList.append(oldWin)
            success = 0
        if self.win is not None:
            if isinstance(self.win, GraphicsWindow):
                self.setupMouse(self.win)
            self.makeCamera2d(self.win)
            if self.wantRender2dp:
                self.makeCamera2dp(self.win)
            if oldLens is not None:
                self.camNode.setLens(oldLens)
                self.camLens = oldLens
            if oldClearColorActive is not None:
                self.win.setClearColorActive(oldClearColorActive)
                self.win.setClearColor(oldClearColor)
                self.win.setClearDepthActive(oldClearDepthActive)
                self.win.setClearDepth(oldClearDepth)
                self.win.setClearStencilActive(oldClearStencilActive)
                self.win.setClearStencil(oldClearStencil)
            flag = ConfigVariableBool('show-frame-rate-meter', False)
            self.setFrameRateMeter(flag.value)
            flag = ConfigVariableBool('show-scene-graph-analyzer-meter', False)
            self.setSceneGraphAnalyzerMeter(flag.value)
        return success

    def setSleep(self, amount):
        if False:
            return 10
        "\n        Sets up a task that calls python 'sleep' every frame.  This is a simple\n        way to reduce the CPU usage (and frame rate) of a panda program.\n        "
        if self.clientSleep == amount:
            return
        self.clientSleep = amount
        if amount == 0.0:
            self.taskMgr.remove('clientSleep')
        else:
            self.taskMgr.remove('clientSleep')
            self.taskMgr.add(self.__sleepCycleTask, 'clientSleep', sort=55)

    def __sleepCycleTask(self, task):
        if False:
            for i in range(10):
                print('nop')
        Thread.sleep(self.clientSleep)
        return Task.cont

    def setFrameRateMeter(self, flag):
        if False:
            for i in range(10):
                print('nop')
        '\n        Turns on or off (according to flag) a standard frame rate\n        meter in the upper-right corner of the main window.\n        '
        if flag:
            if not self.frameRateMeter:
                self.frameRateMeter = FrameRateMeter('frameRateMeter')
                self.frameRateMeter.setupWindow(self.win)
        elif self.frameRateMeter:
            self.frameRateMeter.clearWindow()
            self.frameRateMeter = None

    def setSceneGraphAnalyzerMeter(self, flag):
        if False:
            for i in range(10):
                print('nop')
        '\n        Turns on or off (according to flag) a standard frame rate\n        meter in the upper-right corner of the main window.\n        '
        if flag:
            if not self.sceneGraphAnalyzerMeter:
                self.sceneGraphAnalyzerMeter = SceneGraphAnalyzerMeter('sceneGraphAnalyzerMeter', self.render.node())
                self.sceneGraphAnalyzerMeter.setupWindow(self.win)
        elif self.sceneGraphAnalyzerMeter:
            self.sceneGraphAnalyzerMeter.clearWindow()
            self.sceneGraphAnalyzerMeter = None

    def setupWindowControls(self, winCtrl=None):
        if False:
            for i in range(10):
                print('nop')
        if winCtrl is None:
            winCtrl = WindowControls(self.win, mouseWatcher=self.mouseWatcher, cam=self.camera, camNode=self.camNode, cam2d=self.camera2d, mouseKeyboard=self.dataRoot.find('**/*'))
        self.winControls.append(winCtrl)

    def setupRender(self):
        if False:
            print('Hello World!')
        '\n        Creates the render scene graph, the primary scene graph for\n        rendering 3-d geometry.\n        '
        self.render = NodePath('render')
        self.render.setAttrib(RescaleNormalAttrib.makeDefault())
        self.render.setTwoSided(0)
        self.backfaceCullingEnabled = 1
        self.textureEnabled = 1
        self.wireframeEnabled = 0

    def setupRender2d(self):
        if False:
            return 10
        '\n        Creates the render2d scene graph, the primary scene graph for\n        2-d objects and gui elements that are superimposed over the\n        3-d geometry in the window.\n        '
        from . import ShowBaseGlobal
        self.render2d = ShowBaseGlobal.render2d
        self.render2d.setDepthTest(0)
        self.render2d.setDepthWrite(0)
        self.render2d.setMaterialOff(1)
        self.render2d.setTwoSided(1)
        self.aspect2d = ShowBaseGlobal.aspect2d
        aspectRatio = self.getAspectRatio()
        self.aspect2d.setScale(1.0 / aspectRatio, 1.0, 1.0)
        self.a2dBackground = self.aspect2d.attachNewNode('a2dBackground')
        self.a2dTop = 1.0
        self.a2dBottom = -1.0
        self.a2dLeft = -aspectRatio
        self.a2dRight = aspectRatio
        self.a2dTopCenter = self.aspect2d.attachNewNode('a2dTopCenter')
        self.a2dTopCenterNs = self.aspect2d.attachNewNode('a2dTopCenterNS')
        self.a2dBottomCenter = self.aspect2d.attachNewNode('a2dBottomCenter')
        self.a2dBottomCenterNs = self.aspect2d.attachNewNode('a2dBottomCenterNS')
        self.a2dLeftCenter = self.aspect2d.attachNewNode('a2dLeftCenter')
        self.a2dLeftCenterNs = self.aspect2d.attachNewNode('a2dLeftCenterNS')
        self.a2dRightCenter = self.aspect2d.attachNewNode('a2dRightCenter')
        self.a2dRightCenterNs = self.aspect2d.attachNewNode('a2dRightCenterNS')
        self.a2dTopLeft = self.aspect2d.attachNewNode('a2dTopLeft')
        self.a2dTopLeftNs = self.aspect2d.attachNewNode('a2dTopLeftNS')
        self.a2dTopRight = self.aspect2d.attachNewNode('a2dTopRight')
        self.a2dTopRightNs = self.aspect2d.attachNewNode('a2dTopRightNS')
        self.a2dBottomLeft = self.aspect2d.attachNewNode('a2dBottomLeft')
        self.a2dBottomLeftNs = self.aspect2d.attachNewNode('a2dBottomLeftNS')
        self.a2dBottomRight = self.aspect2d.attachNewNode('a2dBottomRight')
        self.a2dBottomRightNs = self.aspect2d.attachNewNode('a2dBottomRightNS')
        self.a2dTopCenter.setPos(0, 0, self.a2dTop)
        self.a2dTopCenterNs.setPos(0, 0, self.a2dTop)
        self.a2dBottomCenter.setPos(0, 0, self.a2dBottom)
        self.a2dBottomCenterNs.setPos(0, 0, self.a2dBottom)
        self.a2dLeftCenter.setPos(self.a2dLeft, 0, 0)
        self.a2dLeftCenterNs.setPos(self.a2dLeft, 0, 0)
        self.a2dRightCenter.setPos(self.a2dRight, 0, 0)
        self.a2dRightCenterNs.setPos(self.a2dRight, 0, 0)
        self.a2dTopLeft.setPos(self.a2dLeft, 0, self.a2dTop)
        self.a2dTopLeftNs.setPos(self.a2dLeft, 0, self.a2dTop)
        self.a2dTopRight.setPos(self.a2dRight, 0, self.a2dTop)
        self.a2dTopRightNs.setPos(self.a2dRight, 0, self.a2dTop)
        self.a2dBottomLeft.setPos(self.a2dLeft, 0, self.a2dBottom)
        self.a2dBottomLeftNs.setPos(self.a2dLeft, 0, self.a2dBottom)
        self.a2dBottomRight.setPos(self.a2dRight, 0, self.a2dBottom)
        self.a2dBottomRightNs.setPos(self.a2dRight, 0, self.a2dBottom)
        self.pixel2d = self.render2d.attachNewNode(PGTop('pixel2d'))
        self.pixel2d.setPos(-1, 0, 1)
        (xsize, ysize) = self.getSize()
        if xsize > 0 and ysize > 0:
            self.pixel2d.setScale(2.0 / xsize, 1.0, 2.0 / ysize)

    def setupRender2dp(self):
        if False:
            return 10
        '\n        Creates a render2d scene graph, the secondary scene graph for\n        2-d objects and gui elements that are superimposed over the\n        2-d and 3-d geometry in the window.\n        '
        self.render2dp = NodePath('render2dp')
        dt = DepthTestAttrib.make(DepthTestAttrib.MNone)
        dw = DepthWriteAttrib.make(DepthWriteAttrib.MOff)
        self.render2dp.setDepthTest(0)
        self.render2dp.setDepthWrite(0)
        self.render2dp.setMaterialOff(1)
        self.render2dp.setTwoSided(1)
        self.aspect2dp = self.render2dp.attachNewNode(PGTop('aspect2dp'))
        self.aspect2dp.node().setStartSort(16384)
        aspectRatio = self.getAspectRatio()
        self.aspect2dp.setScale(1.0 / aspectRatio, 1.0, 1.0)
        self.a2dpTop = 1.0
        self.a2dpBottom = -1.0
        self.a2dpLeft = -aspectRatio
        self.a2dpRight = aspectRatio
        self.a2dpTopCenter = self.aspect2dp.attachNewNode('a2dpTopCenter')
        self.a2dpBottomCenter = self.aspect2dp.attachNewNode('a2dpBottomCenter')
        self.a2dpLeftCenter = self.aspect2dp.attachNewNode('a2dpLeftCenter')
        self.a2dpRightCenter = self.aspect2dp.attachNewNode('a2dpRightCenter')
        self.a2dpTopLeft = self.aspect2dp.attachNewNode('a2dpTopLeft')
        self.a2dpTopRight = self.aspect2dp.attachNewNode('a2dpTopRight')
        self.a2dpBottomLeft = self.aspect2dp.attachNewNode('a2dpBottomLeft')
        self.a2dpBottomRight = self.aspect2dp.attachNewNode('a2dpBottomRight')
        self.a2dpTopCenter.setPos(0, 0, self.a2dpTop)
        self.a2dpBottomCenter.setPos(0, 0, self.a2dpBottom)
        self.a2dpLeftCenter.setPos(self.a2dpLeft, 0, 0)
        self.a2dpRightCenter.setPos(self.a2dpRight, 0, 0)
        self.a2dpTopLeft.setPos(self.a2dpLeft, 0, self.a2dpTop)
        self.a2dpTopRight.setPos(self.a2dpRight, 0, self.a2dpTop)
        self.a2dpBottomLeft.setPos(self.a2dpLeft, 0, self.a2dpBottom)
        self.a2dpBottomRight.setPos(self.a2dpRight, 0, self.a2dpBottom)
        self.pixel2dp = self.render2dp.attachNewNode(PGTop('pixel2dp'))
        self.pixel2dp.node().setStartSort(16384)
        self.pixel2dp.setPos(-1, 0, 1)
        (xsize, ysize) = self.getSize()
        if xsize > 0 and ysize > 0:
            self.pixel2dp.setScale(2.0 / xsize, 1.0, 2.0 / ysize)

    def setAspectRatio(self, aspectRatio):
        if False:
            for i in range(10):
                print('nop')
        ' Sets the global aspect ratio of the main window.  Set it\n        to None to restore automatic scaling. '
        self.__configAspectRatio = aspectRatio
        self.adjustWindowAspectRatio(self.getAspectRatio())

    def getAspectRatio(self, win=None):
        if False:
            return 10
        if self.__configAspectRatio:
            return self.__configAspectRatio
        aspectRatio = 1
        if win is None:
            win = self.win
        if win is not None and win.hasSize() and (win.getSbsLeftYSize() != 0):
            aspectRatio = float(win.getSbsLeftXSize()) / float(win.getSbsLeftYSize())
        else:
            if win is None or not hasattr(win, 'getRequestedProperties'):
                props = WindowProperties.getDefault()
            else:
                props = win.getRequestedProperties()
                if not props.hasSize():
                    props = WindowProperties.getDefault()
            if props.hasSize() and props.getYSize() != 0:
                aspectRatio = float(props.getXSize()) / float(props.getYSize())
        if aspectRatio == 0:
            return 1
        return aspectRatio

    def getSize(self, win=None):
        if False:
            print('Hello World!')
        '\n        Returns the actual size of the indicated (or main window), or the\n        default size if there is not yet a main window.\n        '
        if win is None:
            win = self.win
        if win is not None and win.hasSize():
            return (win.getXSize(), win.getYSize())
        else:
            if win is None or not hasattr(win, 'getRequestedProperties'):
                props = WindowProperties.getDefault()
            else:
                props = win.getRequestedProperties()
                if not props.hasSize():
                    props = WindowProperties.getDefault()
            return (props.getXSize(), props.getYSize())

    def makeCamera(self, win, sort=0, scene=None, displayRegion=(0, 1, 0, 1), stereo=None, aspectRatio=None, clearDepth=0, clearColor=None, lens=None, camName='cam', mask=None, useCamera=None):
        if False:
            print('Hello World!')
        '\n        Makes a new 3-d camera associated with the indicated window,\n        and creates a display region in the indicated subrectangle.\n\n        If stereo is True, then a stereo camera is created, with a\n        pair of DisplayRegions.  If stereo is False, then a standard\n        camera is created.  If stereo is None or omitted, a stereo\n        camera is created if the window says it can render in stereo.\n\n        If useCamera is not None, it is a NodePath to be used as the\n        camera to apply to the window, rather than creating a new\n        camera.\n\n        :rtype: panda3d.core.NodePath\n        '
        if self.camera is None:
            self.camera = self.render.attachNewNode(ModelNode('camera'))
            self.camera.node().setPreserveTransform(ModelNode.PTLocal)
            builtins.camera = self.camera
            self.mouse2cam.node().setNode(self.camera.node())
        if useCamera:
            cam = useCamera
            camNode = useCamera.node()
            assert isinstance(camNode, Camera)
            lens = camNode.getLens()
            cam.reparentTo(self.camera)
        else:
            camNode = Camera(camName)
            if lens is None:
                lens = PerspectiveLens()
                if aspectRatio is None:
                    aspectRatio = self.getAspectRatio(win)
                lens.setAspectRatio(aspectRatio)
            cam = self.camera.attachNewNode(camNode)
        if lens is not None:
            camNode.setLens(lens)
        if scene is not None:
            camNode.setScene(scene)
        if mask is not None:
            if isinstance(mask, int):
                mask = BitMask32(mask)
            camNode.setCameraMask(mask)
        if self.cam is None:
            self.cam = cam
            self.camNode = camNode
            self.camLens = lens
        self.camList.append(cam)
        if stereo is not None:
            if stereo:
                dr = win.makeStereoDisplayRegion(*displayRegion)
            else:
                dr = win.makeMonoDisplayRegion(*displayRegion)
        else:
            dr = win.makeDisplayRegion(*displayRegion)
        dr.setSort(sort)
        if clearDepth:
            dr.setClearDepthActive(1)
        if clearColor:
            dr.setClearColorActive(1)
            dr.setClearColor(clearColor)
        dr.setCamera(cam)
        return cam

    def makeCamera2d(self, win, sort=10, displayRegion=(0, 1, 0, 1), coords=(-1, 1, -1, 1), lens=None, cameraName=None):
        if False:
            print('Hello World!')
        '\n        Makes a new camera2d associated with the indicated window, and\n        assigns it to render the indicated subrectangle of render2d.\n\n        :rtype: panda3d.core.NodePath\n        '
        dr = win.makeMonoDisplayRegion(*displayRegion)
        dr.setSort(sort)
        dr.setClearDepthActive(1)
        dr.setIncompleteRender(False)
        (left, right, bottom, top) = coords
        if cameraName:
            cam2dNode = Camera('cam2d_' + cameraName)
        else:
            cam2dNode = Camera('cam2d')
        if lens is None:
            lens = OrthographicLens()
            lens.setFilmSize(right - left, top - bottom)
            lens.setFilmOffset((right + left) * 0.5, (top + bottom) * 0.5)
            lens.setNearFar(-1000, 1000)
        cam2dNode.setLens(lens)
        if self.camera2d is None:
            self.camera2d = self.render2d.attachNewNode('camera2d')
        camera2d = self.camera2d.attachNewNode(cam2dNode)
        dr.setCamera(camera2d)
        if self.cam2d is None:
            self.cam2d = camera2d
        return camera2d

    def makeCamera2dp(self, win, sort=20, displayRegion=(0, 1, 0, 1), coords=(-1, 1, -1, 1), lens=None, cameraName=None):
        if False:
            return 10
        '\n        Makes a new camera2dp associated with the indicated window, and\n        assigns it to render the indicated subrectangle of render2dp.\n\n        :rtype: panda3d.core.NodePath\n        '
        dr = win.makeMonoDisplayRegion(*displayRegion)
        dr.setSort(sort)
        if hasattr(dr, 'setIncompleteRender'):
            dr.setIncompleteRender(False)
        (left, right, bottom, top) = coords
        if cameraName:
            cam2dNode = Camera('cam2dp_' + cameraName)
        else:
            cam2dNode = Camera('cam2dp')
        if lens is None:
            lens = OrthographicLens()
            lens.setFilmSize(right - left, top - bottom)
            lens.setFilmOffset((right + left) * 0.5, (top + bottom) * 0.5)
            lens.setNearFar(-1000, 1000)
        cam2dNode.setLens(lens)
        if self.camera2dp is None:
            self.camera2dp = self.render2dp.attachNewNode('camera2dp')
        camera2dp = self.camera2dp.attachNewNode(cam2dNode)
        dr.setCamera(camera2dp)
        if self.cam2dp is None:
            self.cam2dp = camera2dp
        return camera2dp

    def setupDataGraph(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates the data graph and populates it with the basic input\n        devices.\n        '
        self.dataRoot = NodePath('dataRoot')
        self.dataRootNode = self.dataRoot.node()
        self.trackball = NodePath(Trackball('trackball'))
        self.drive = NodePath(DriveInterface('drive'))
        self.mouse2cam = NodePath(Transform2SG('mouse2cam'))

    def setupMouse(self, win, fMultiWin=False):
        if False:
            print('Hello World!')
        '\n        Creates the structures necessary to monitor the mouse input,\n        using the indicated window.  If the mouse has already been set\n        up for a different window, those structures are deleted first.\n\n        :param fMultiWin: If True, then the previous mouse structures are not\n                          deleted; instead, multiple windows are allowed to\n                          monitor the mouse input.  However, in this case, the\n                          trackball controls are not set up, and must be set up\n                          by hand if desired.\n\n        :returns: The ButtonThrower NodePath created for this window.\n        '
        if not fMultiWin and self.buttonThrowers is not None:
            for bt in self.buttonThrowers:
                mw = bt.getParent()
                mk = mw.getParent()
                bt.removeNode()
                mw.removeNode()
                mk.removeNode()
        (bts, pws) = self.setupMouseCB(win)
        if fMultiWin:
            return bts[0]
        self.buttonThrowers = bts[:]
        self.pointerWatcherNodes = pws[:]
        self.mouseWatcher = self.buttonThrowers[0].getParent()
        self.mouseWatcherNode = self.mouseWatcher.node()
        if self.mouseInterface:
            self.mouseInterface.reparentTo(self.mouseWatcher)
        if self.recorder:
            mw = self.buttonThrowers[0].getParent()
            mouseRecorder = MouseRecorder('mouse')
            self.recorder.addRecorder('mouse', mouseRecorder)
            np = mw.getParent().attachNewNode(mouseRecorder)
            mw.reparentTo(np)
        mw = self.buttonThrowers[0].getParent()
        self.timeButtonThrower = mw.attachNewNode(ButtonThrower('timeButtons'))
        self.timeButtonThrower.node().setPrefix('time-')
        self.timeButtonThrower.node().setTimeFlag(1)
        self.aspect2d.node().setMouseWatcher(mw.node())
        self.pixel2d.node().setMouseWatcher(mw.node())
        if self.wantRender2dp:
            self.aspect2dp.node().setMouseWatcher(mw.node())
            self.pixel2dp.node().setMouseWatcher(mw.node())
        mw.node().addRegion(PGMouseWatcherBackground())
        return self.buttonThrowers[0]

    def setupMouseCB(self, win):
        if False:
            for i in range(10):
                print('nop')
        buttonThrowers = []
        pointerWatcherNodes = []
        for i in range(win.getNumInputDevices()):
            name = win.getInputDeviceName(i)
            mk = self.dataRoot.attachNewNode(MouseAndKeyboard(win, i, name))
            mw = mk.attachNewNode(MouseWatcher('watcher%s' % i))
            if win.getSideBySideStereo():
                mw.node().setDisplayRegion(win.getOverlayDisplayRegion())
            mb = mw.node().getModifierButtons()
            mb.addButton(KeyboardButton.shift())
            mb.addButton(KeyboardButton.control())
            mb.addButton(KeyboardButton.alt())
            mb.addButton(KeyboardButton.meta())
            mw.node().setModifierButtons(mb)
            bt = mw.attachNewNode(ButtonThrower('buttons%s' % i))
            if i != 0:
                bt.node().setPrefix('mousedev%s-' % i)
            mods = ModifierButtons()
            mods.addButton(KeyboardButton.shift())
            mods.addButton(KeyboardButton.control())
            mods.addButton(KeyboardButton.alt())
            mods.addButton(KeyboardButton.meta())
            bt.node().setModifierButtons(mods)
            buttonThrowers.append(bt)
            if win.hasPointer(i):
                pointerWatcherNodes.append(mw.node())
        return (buttonThrowers, pointerWatcherNodes)

    def enableSoftwareMousePointer(self):
        if False:
            i = 10
            return i + 15
        '\n        Creates some geometry and parents it to render2d to show\n        the currently-known mouse position.  Useful if the mouse\n        pointer is invisible for some reason.\n        '
        mouseViz = self.render2d.attachNewNode('mouseViz')
        lilsmiley = self.loader.loadModel('lilsmiley')
        lilsmiley.reparentTo(mouseViz)
        aspectRatio = self.getAspectRatio()
        height = self.win.getSbsLeftYSize()
        lilsmiley.setScale(32.0 / height / aspectRatio, 1.0, 32.0 / height)
        self.mouseWatcherNode.setGeometry(mouseViz.node())

    def getAlt(self):
        if False:
            return 10
        '\n        Returns True if the alt key is currently held down.\n        '
        return self.mouseWatcherNode.getModifierButtons().isDown(KeyboardButton.alt())

    def getShift(self):
        if False:
            return 10
        '\n        Returns True if the shift key is currently held down.\n        '
        return self.mouseWatcherNode.getModifierButtons().isDown(KeyboardButton.shift())

    def getControl(self):
        if False:
            return 10
        '\n        Returns True if the control key is currently held down.\n        '
        return self.mouseWatcherNode.getModifierButtons().isDown(KeyboardButton.control())

    def getMeta(self):
        if False:
            return 10
        '\n        Returns True if the meta key is currently held down.\n        '
        return self.mouseWatcherNode.getModifierButtons().isDown(KeyboardButton.meta())

    def attachInputDevice(self, device, prefix=None, watch=False):
        if False:
            return 10
        '\n        This function attaches an input device to the data graph, which will\n        cause the device to be polled and generate events.  If a prefix is\n        given and not None, it is used to prefix events generated by this\n        device, separated by a hyphen.\n\n        The watch argument can be set to True (as of Panda3D 1.10.3) to set up\n        the default MouseWatcher to receive inputs from this device, allowing\n        it to be polled via mouseWatcherNode and control user interfaces.\n        Setting this to True will also make it generate unprefixed events,\n        regardless of the specified prefix.\n\n        If you call this, you should consider calling detachInputDevice when\n        you are done with the device or when it is disconnected.\n        '
        assert device not in self.__inputDeviceNodes
        idn = self.dataRoot.attachNewNode(InputDeviceNode(device, device.name))
        if prefix is not None or not watch:
            bt = idn.attachNewNode(ButtonThrower(device.name))
            if prefix is not None:
                bt.node().setPrefix(prefix + '-')
            self.deviceButtonThrowers.append(bt)
        assert self.notify.debug('Attached input device {0} with prefix {1}'.format(device, prefix))
        self.__inputDeviceNodes[device] = idn
        if watch:
            idn.node().addChild(self.mouseWatcherNode)

    def detachInputDevice(self, device):
        if False:
            while True:
                i = 10
        "\n        This should be called after attaching an input device using\n        attachInputDevice and the device is disconnected or you no longer wish\n        to keep polling this device for events.\n\n        You do not strictly need to call this if you expect the device to be\n        reconnected (but be careful that you don't reattach it).\n        "
        if device not in self.__inputDeviceNodes:
            assert device in self.__inputDeviceNodes
            return
        assert self.notify.debug('Detached device {0}'.format(device.name))
        idn = self.__inputDeviceNodes[device]
        for bt in self.deviceButtonThrowers:
            if idn.isAncestorOf(bt):
                self.deviceButtonThrowers.remove(bt)
                break
        idn.removeNode()
        del self.__inputDeviceNodes[device]

    def addAngularIntegrator(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds a :class:`~panda3d.physics.AngularEulerIntegrator` to the default\n        physics manager.  By default, only a\n        :class:`~panda3d.physics.LinearEulerIntegrator` is attached.\n        '
        if not self.physicsMgrAngular:
            physics = importlib.import_module('panda3d.physics')
            self.physicsMgrAngular = 1
            integrator = physics.AngularEulerIntegrator()
            self.physicsMgr.attachAngularIntegrator(integrator)

    def enableParticles(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Enables the particle and physics managers, which are stored in\n        `particleMgr` and `physicsMgr` members, respectively.  Also starts a\n        task to periodically update these managers.\n\n        By default, only a :class:`~panda3d.physics.LinearEulerIntegrator` is\n        attached to the physics manager.  To attach an angular integrator,\n        follow this up with a call to `addAngularIntegrator()`.\n        '
        if not self.particleMgrEnabled:
            if not self.particleMgr:
                PMG = importlib.import_module('direct.particles.ParticleManagerGlobal')
                self.particleMgr = PMG.particleMgr
                self.particleMgr.setFrameStepping(1)
            if not self.physicsMgr:
                PMG = importlib.import_module('direct.showbase.PhysicsManagerGlobal')
                physics = importlib.import_module('panda3d.physics')
                self.physicsMgr = PMG.physicsMgr
                integrator = physics.LinearEulerIntegrator()
                self.physicsMgr.attachLinearIntegrator(integrator)
            self.particleMgrEnabled = 1
            self.physicsMgrEnabled = 1
            self.taskMgr.remove('manager-update')
            self.taskMgr.add(self.updateManagers, 'manager-update')

    def disableParticles(self):
        if False:
            print('Hello World!')
        '\n        The opposite of `enableParticles()`.\n        '
        if self.particleMgrEnabled:
            self.particleMgrEnabled = 0
            self.physicsMgrEnabled = 0
            self.taskMgr.remove('manager-update')

    def toggleParticles(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calls `enableParticles()` or `disableParticles()` depending on the\n        current state.\n        '
        if self.particleMgrEnabled == 0:
            self.enableParticles()
        else:
            self.disableParticles()

    def isParticleMgrEnabled(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns True if `enableParticles()` has been called.\n        '
        return self.particleMgrEnabled

    def isPhysicsMgrEnabled(self):
        if False:
            while True:
                i = 10
        '\n        Returns True if `enableParticles()` has been called.\n        '
        return self.physicsMgrEnabled

    def updateManagers(self, state):
        if False:
            while True:
                i = 10
        dt = self.clock.dt
        if self.particleMgrEnabled:
            self.particleMgr.doParticles(dt)
        if self.physicsMgrEnabled:
            self.physicsMgr.doPhysics(dt)
        return Task.cont

    def createStats(self, hostname=None, port=None):
        if False:
            print('Hello World!')
        '\n        If want-pstats is set in Config.prc, or the `wantStats` member is\n        otherwise set to True, connects to the PStats server.\n        This is normally called automatically from the ShowBase constructor.\n        '
        if not self.wantStats:
            return False
        if PStatClient.isConnected():
            PStatClient.disconnect()
        if hostname is None:
            hostname = ''
        if port is None:
            port = -1
        PStatClient.connect(hostname, port)
        if PStatClient.isConnected():
            PStatClient.mainTick()
            return True
        else:
            return False

    def addSfxManager(self, extraSfxManager):
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds an additional SFX audio manager to `sfxManagerList`, the list of\n        managers managed by ShowBase.\n        '
        self.sfxManagerList.append(extraSfxManager)
        newSfxManagerIsValid = extraSfxManager is not None and extraSfxManager.isValid()
        self.sfxManagerIsValidList.append(newSfxManagerIsValid)
        if newSfxManagerIsValid:
            extraSfxManager.setActive(self.sfxActive)

    def createBaseAudioManagers(self):
        if False:
            i = 10
            return i + 15
        '\n        Creates the default SFX and music manager.  Called automatically from\n        the ShowBase constructor.\n        '
        self.sfxPlayer = SfxPlayer.SfxPlayer()
        sfxManager = AudioManager.createAudioManager()
        self.addSfxManager(sfxManager)
        self.musicManager = AudioManager.createAudioManager()
        self.musicManagerIsValid = self.musicManager is not None and self.musicManager.isValid()
        if self.musicManagerIsValid:
            self.musicManager.setConcurrentSoundLimit(1)
            self.musicManager.setActive(self.musicActive)

    def enableMusic(self, bEnableMusic):
        if False:
            for i in range(10):
                print('nop')
        '\n        Enables or disables the music manager.\n        '
        if self.AppHasAudioFocus and self.musicManagerIsValid:
            self.musicManager.setActive(bEnableMusic)
        self.musicActive = bEnableMusic
        if bEnableMusic:
            messenger.send('MusicEnabled')
            self.notify.debug('Enabling music')
        else:
            self.notify.debug('Disabling music')

    def SetAllSfxEnables(self, bEnabled):
        if False:
            return 10
        'Calls ``setActive(bEnabled)`` on all valid SFX managers.'
        for i in range(len(self.sfxManagerList)):
            if self.sfxManagerIsValidList[i]:
                self.sfxManagerList[i].setActive(bEnabled)

    def enableSoundEffects(self, bEnableSoundEffects):
        if False:
            i = 10
            return i + 15
        '\n        Enables or disables SFX managers.\n        '
        if self.AppHasAudioFocus or not bEnableSoundEffects:
            self.SetAllSfxEnables(bEnableSoundEffects)
        self.sfxActive = bEnableSoundEffects
        if bEnableSoundEffects:
            self.notify.debug('Enabling sound effects')
        else:
            self.notify.debug('Disabling sound effects')

    def disableAllAudio(self):
        if False:
            print('Hello World!')
        '\n        Disables all SFX and music managers, meant to be called when the app\n        loses audio focus.\n        '
        self.AppHasAudioFocus = 0
        self.SetAllSfxEnables(0)
        if self.musicManagerIsValid:
            self.musicManager.setActive(0)
        self.notify.debug('Disabling audio')

    def enableAllAudio(self):
        if False:
            while True:
                i = 10
        '\n        Reenables the SFX and music managers that were active at the time\n        `disableAllAudio()` was called.  Meant to be called when the app regains\n        audio focus.\n        '
        self.AppHasAudioFocus = 1
        self.SetAllSfxEnables(self.sfxActive)
        if self.musicManagerIsValid:
            self.musicManager.setActive(self.musicActive)
        self.notify.debug('Enabling audio')

    def loadSfx(self, name):
        if False:
            while True:
                i = 10
        '\n        :deprecated: Use `.Loader.Loader.loadSfx()` instead.\n        '
        if __debug__:
            warnings.warn('base.loadSfx is deprecated, use base.loader.loadSfx instead.', DeprecationWarning, stacklevel=2)
        return self.loader.loadSfx(name)

    def loadMusic(self, name):
        if False:
            return 10
        '\n        :deprecated: Use `.Loader.Loader.loadMusic()` instead.\n        '
        if __debug__:
            warnings.warn('base.loadMusic is deprecated, use base.loader.loadMusic instead.', DeprecationWarning, stacklevel=2)
        return self.loader.loadMusic(name)

    def playSfx(self, sfx, looping=0, interrupt=1, volume=None, time=0.0, node=None, listener=None, cutoff=None):
        if False:
            i = 10
            return i + 15
        return self.sfxPlayer.playSfx(sfx, looping, interrupt, volume, time, node, listener, cutoff)

    def playMusic(self, music, looping=0, interrupt=1, volume=None, time=0.0):
        if False:
            return 10
        if music:
            if volume is not None:
                music.setVolume(volume)
            if interrupt or music.status() != AudioSound.PLAYING:
                music.setTime(time)
                music.setLoop(looping)
                music.play()

    def __resetPrevTransform(self, state):
        if False:
            i = 10
            return i + 15
        PandaNode.resetAllPrevTransform()
        return Task.cont

    def __dataLoop(self, state):
        if False:
            print('Hello World!')
        self.devices.update()
        self.dgTrav.traverse(self.dataRootNode)
        return Task.cont

    def __ivalLoop(self, state):
        if False:
            i = 10
            return i + 15
        IntervalManager.ivalMgr.step()
        return Task.cont

    def initShadowTrav(self):
        if False:
            return 10
        if not self.shadowTrav:
            self.shadowTrav = CollisionTraverser('base.shadowTrav')
            self.shadowTrav.setRespectPrevTransform(False)

    def __shadowCollisionLoop(self, state):
        if False:
            print('Hello World!')
        if self.shadowTrav:
            self.shadowTrav.traverse(self.render)
        return Task.cont

    def __collisionLoop(self, state):
        if False:
            i = 10
            return i + 15
        if self.cTrav:
            self.cTrav.traverse(self.render)
        if self.appTrav:
            self.appTrav.traverse(self.render)
        if self.shadowTrav:
            self.shadowTrav.traverse(self.render)
        messenger.send('collisionLoopFinished')
        return Task.cont

    def __audioLoop(self, state):
        if False:
            return 10
        if self.musicManager is not None:
            self.musicManager.update()
        for x in self.sfxManagerList:
            x.update()
        return Task.cont

    def __garbageCollectStates(self, state):
        if False:
            print('Hello World!')
        " This task is started only when we have\n        garbage-collect-states set in the Config.prc file, in which\n        case we're responsible for taking out Panda's garbage from\n        time to time.  This is not to be confused with Python's\n        garbage collection.  "
        TransformState.garbageCollect()
        RenderState.garbageCollect()
        return Task.cont

    def __igLoop(self, state):
        if False:
            i = 10
            return i + 15
        if __debug__:
            self.onScreenDebug.render()
        if self.recorder:
            self.recorder.recordFrame()
        self.graphicsEngine.renderFrame()
        if self.clusterSyncFlag:
            self.graphicsEngine.syncFrame()
        if self.multiClientSleep:
            time.sleep(0)
        if __debug__:
            self.onScreenDebug.clear()
        if self.recorder:
            self.recorder.playFrame()
        if self.mainWinMinimized:
            time.sleep(0.1)
        throw_new_frame()
        return Task.cont

    def __igLoopSync(self, state):
        if False:
            while True:
                i = 10
        if __debug__:
            self.onScreenDebug.render()
        if self.recorder:
            self.recorder.recordFrame()
        self.cluster.collectData()
        self.graphicsEngine.renderFrame()
        if self.clusterSyncFlag:
            self.graphicsEngine.syncFrame()
        if self.multiClientSleep:
            time.sleep(0)
        if __debug__:
            self.onScreenDebug.clear()
        if self.recorder:
            self.recorder.playFrame()
        if self.mainWinMinimized:
            time.sleep(0.1)
        self.graphicsEngine.readyFlip()
        self.cluster.waitForFlipCommand()
        self.graphicsEngine.flipFrame()
        throw_new_frame()
        return Task.cont

    def restart(self, clusterSync=False, cluster=None):
        if False:
            while True:
                i = 10
        self.shutdown()
        self.taskMgr.add(self.__resetPrevTransform, 'resetPrevTransform', sort=-51)
        self.taskMgr.add(self.__dataLoop, 'dataLoop', sort=-50)
        self.__deadInputs = 0
        self.taskMgr.add(self.__ivalLoop, 'ivalLoop', sort=20)
        self.taskMgr.add(self.__collisionLoop, 'collisionLoop', sort=30)
        if ConfigVariableBool('garbage-collect-states').value:
            self.taskMgr.add(self.__garbageCollectStates, 'garbageCollectStates', sort=46)
        self.cluster = cluster
        if not clusterSync or cluster is None:
            self.taskMgr.add(self.__igLoop, 'igLoop', sort=50)
        else:
            self.taskMgr.add(self.__igLoopSync, 'igLoop', sort=50)
        self.taskMgr.add(self.__audioLoop, 'audioLoop', sort=60)
        self.eventMgr.restart()

    def shutdown(self):
        if False:
            for i in range(10):
                print('nop')
        self.taskMgr.remove('audioLoop')
        self.taskMgr.remove('igLoop')
        self.taskMgr.remove('shadowCollisionLoop')
        self.taskMgr.remove('collisionLoop')
        self.taskMgr.remove('dataLoop')
        self.taskMgr.remove('resetPrevTransform')
        self.taskMgr.remove('ivalLoop')
        self.taskMgr.remove('garbageCollectStates')
        self.eventMgr.shutdown()

    def getBackgroundColor(self, win=None):
        if False:
            return 10
        '\n        Returns the current window background color.  This assumes\n        the window is set up to clear the color each frame (this is\n        the normal setting).\n\n        :rtype: panda3d.core.VBase4\n        '
        if win is None:
            win = self.win
        return VBase4(win.getClearColor())

    def setBackgroundColor(self, r=None, g=None, b=None, a=0.0, win=None):
        if False:
            i = 10
            return i + 15
        '\n        Sets the window background color to the indicated value.\n        This assumes the window is set up to clear the color each\n        frame (this is the normal setting).\n\n        The color may be either a VBase3 or a VBase4, or a 3-component\n        tuple, or the individual r, g, b parameters.\n        '
        if g is not None:
            color = VBase4(r, g, b, a)
        else:
            arg = r
            if isinstance(arg, VBase4):
                color = arg
            else:
                color = VBase4(arg[0], arg[1], arg[2], a)
        if win is None:
            win = self.win
        if win:
            win.setClearColor(color)

    def toggleBackface(self):
        if False:
            while True:
                i = 10
        '\n        Toggles between `backfaceCullingOn()` and `backfaceCullingOff()`.\n        '
        if self.backfaceCullingEnabled:
            self.backfaceCullingOff()
        else:
            self.backfaceCullingOn()

    def backfaceCullingOn(self):
        if False:
            return 10
        '\n        Disables two-sided rendering on the entire 3D scene graph.\n        '
        if not self.backfaceCullingEnabled:
            self.render.setTwoSided(0)
        self.backfaceCullingEnabled = 1

    def backfaceCullingOff(self):
        if False:
            print('Hello World!')
        '\n        Enables two-sided rendering on the entire 3D scene graph.\n        '
        if self.backfaceCullingEnabled:
            self.render.setTwoSided(1)
        self.backfaceCullingEnabled = 0

    def toggleTexture(self):
        if False:
            return 10
        '\n        Toggles between `textureOn()` and `textureOff()`.\n        '
        if self.textureEnabled:
            self.textureOff()
        else:
            self.textureOn()

    def textureOn(self):
        if False:
            while True:
                i = 10
        '\n        Undoes the effects of a previous call to `textureOff()`.\n        '
        self.render.clearTexture()
        self.textureEnabled = 1

    def textureOff(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Disables texturing on the entire 3D scene graph.\n        '
        self.render.setTextureOff(100)
        self.textureEnabled = 0

    def toggleWireframe(self):
        if False:
            while True:
                i = 10
        '\n        Toggles between `wireframeOn()` and `wireframeOff()`.\n        '
        if self.wireframeEnabled:
            self.wireframeOff()
        else:
            self.wireframeOn()

    def wireframeOn(self):
        if False:
            print('Hello World!')
        '\n        Enables wireframe rendering on the entire 3D scene graph.\n        '
        self.render.setRenderModeWireframe(100)
        self.render.setTwoSided(1)
        self.wireframeEnabled = 1

    def wireframeOff(self):
        if False:
            print('Hello World!')
        '\n        Undoes the effects of a previous call to `wireframeOn()`.\n        '
        self.render.clearRenderMode()
        render.setTwoSided(not self.backfaceCullingEnabled)
        self.wireframeEnabled = 0

    def disableMouse(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Temporarily disable the mouse control of the camera, either\n        via the drive interface or the trackball, whichever is\n        currently in use.\n        '
        if self.mouse2cam:
            self.mouse2cam.detachNode()

    def enableMouse(self):
        if False:
            return 10
        '\n        Reverse the effect of a previous call to `disableMouse()`.\n        `useDrive()` also implicitly enables the mouse.\n        '
        if self.mouse2cam:
            self.mouse2cam.reparentTo(self.mouseInterface)

    def silenceInput(self):
        if False:
            while True:
                i = 10
        '\n        This is a heavy-handed way of temporarily turning off\n        all inputs.  Bring them back with `reviveInput()`.\n        '
        if not self.__deadInputs:
            self.__deadInputs = taskMgr.remove('dataLoop')

    def reviveInput(self):
        if False:
            return 10
        '\n        Restores inputs after a previous call to `silenceInput()`.\n        '
        if self.__deadInputs:
            self.eventMgr.doEvents()
            self.dgTrav.traverse(self.dataRootNode)
            self.eventMgr.eventQueue.clear()
            self.taskMgr.add(self.__dataLoop, 'dataLoop', sort=-50)
            self.__deadInputs = 0

    def setMouseOnNode(self, newNode):
        if False:
            return 10
        if self.mouse2cam:
            self.mouse2cam.node().setNode(newNode)

    def changeMouseInterface(self, changeTo):
        if False:
            print('Hello World!')
        '\n        Change the mouse interface used to control the camera.\n        '
        self.mouseInterface.detachNode()
        self.mouseInterface = changeTo
        self.mouseInterfaceNode = self.mouseInterface.node()
        if self.mouseWatcher:
            self.mouseInterface.reparentTo(self.mouseWatcher)
        if self.mouse2cam:
            self.mouse2cam.reparentTo(self.mouseInterface)

    def useDrive(self):
        if False:
            return 10
        '\n        Changes the mouse interface used for camera control to drive mode.\n        '
        if self.drive:
            self.changeMouseInterface(self.drive)
            self.mouseInterfaceNode.reset()
            self.mouseInterfaceNode.setZ(4.0)

    def useTrackball(self):
        if False:
            while True:
                i = 10
        '\n        Changes the mouse interface used for camera control to trackball mode.\n        '
        if self.trackball:
            self.changeMouseInterface(self.trackball)

    def toggleTexMem(self):
        if False:
            i = 10
            return i + 15
        '\n        Toggles a handy texture memory watcher utility.\n        See :mod:`direct.showutil.TexMemWatcher` for more information.\n        '
        if self.texmem and (not self.texmem.cleanedUp):
            self.texmem.cleanup()
            self.texmem = None
            return
        TMW = importlib.import_module('direct.showutil.TexMemWatcher')
        self.texmem = TMW.TexMemWatcher()

    def toggleShowVertices(self):
        if False:
            i = 10
            return i + 15
        ' Toggles a mode that visualizes vertex density per screen\n        area. '
        if self.showVertices:
            self.showVertices.node().setActive(0)
            dr = self.showVertices.node().getDisplayRegion(0)
            self.win.removeDisplayRegion(dr)
            self.showVertices.removeNode()
            self.showVertices = None
            return
        dr = self.win.makeDisplayRegion()
        dr.setSort(1000)
        cam = Camera('showVertices')
        cam.setLens(self.camLens)
        override = 100000
        t = NodePath('t')
        t.setColor(1, 0, 1, 0.02, override)
        t.setColorScale(1, 1, 1, 1, override)
        t.setAttrib(ColorBlendAttrib.make(ColorBlendAttrib.MAdd, ColorBlendAttrib.OIncomingAlpha, ColorBlendAttrib.OOneMinusIncomingAlpha), override)
        t.setAttrib(RenderModeAttrib.make(RenderModeAttrib.MPoint, 10), override)
        t.setTwoSided(True, override)
        t.setBin('fixed', 0, override)
        t.setDepthTest(False, override)
        t.setDepthWrite(False, override)
        t.setLightOff(override)
        t.setShaderOff(override)
        t.setFogOff(override)
        t.setAttrib(AntialiasAttrib.make(AntialiasAttrib.MNone), override)
        t.setAttrib(RescaleNormalAttrib.make(RescaleNormalAttrib.MNone), override)
        t.setTextureOff(override)
        if ConfigVariableBool('round-show-vertices', False):
            spot = PNMImage(256, 256, 1)
            spot.renderSpot((1, 1, 1, 1), (0, 0, 0, 0), 0.8, 1)
            tex = Texture('spot')
            tex.load(spot)
            tex.setFormat(tex.FAlpha)
            t.setTexture(tex, override)
            t.setAttrib(TexGenAttrib.make(TextureStage.getDefault(), TexGenAttrib.MPointSprite), override)
        cam.setInitialState(t.getState())
        cam.setCameraMask(~PandaNode.getOverallBit())
        self.showVertices = self.cam.attachNewNode(cam)
        dr.setCamera(self.showVertices)

    def oobe(self, cam=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Enable a special "out-of-body experience" mouse-interface\n        mode.  This can be used when a "god" camera is needed; it\n        moves the camera node out from under its normal node and sets\n        the world up in trackball state.  Button events are still sent\n        to the normal mouse action node (e.g. the DriveInterface), and\n        mouse events, if needed, may be sent to the normal node by\n        holding down the Control key.\n\n        This is different than `useTrackball()`, which simply changes\n        the existing mouse action to a trackball interface.  In fact,\n        OOBE mode doesn\'t care whether `useDrive()` or `useTrackball()` is\n        in effect; it just temporarily layers a new trackball\n        interface on top of whatever the basic interface is.  You can\n        even switch between `useDrive()` and `useTrackball()` while OOBE\n        mode is in effect.\n\n        This is a toggle; the second time this function is called, it\n        disables the mode.\n        '
        if cam is None:
            cam = self.cam
        if not hasattr(self, 'oobeMode'):
            self.oobeMode = 0
            self.oobeCamera = self.hidden.attachNewNode('oobeCamera')
            self.oobeCameraTrackball = self.oobeCamera.attachNewNode('oobeCameraTrackball')
            self.oobeLens = PerspectiveLens()
            self.oobeLens.setAspectRatio(self.getAspectRatio())
            self.oobeLens.setNearFar(0.1, 10000.0)
            self.oobeLens.setMinFov(40)
            self.oobeTrackball = NodePath(Trackball('oobeTrackball'))
            self.oobe2cam = self.oobeTrackball.attachNewNode(Transform2SG('oobe2cam'))
            self.oobe2cam.node().setNode(self.oobeCameraTrackball.node())
            self.oobeVis = base.loader.loadModel('models/misc/camera', okMissing=True)
            if not self.oobeVis:
                self.oobeVis = base.loader.loadModel('models/misc/camera.bam', okMissing=True)
            if not self.oobeVis:
                self.oobeVis = NodePath('oobeVis')
            self.oobeVis.node().setFinal(1)
            self.oobeVis.setLightOff(1)
            self.oobeCullFrustum = None
            self.__directObject.accept('oobe-down', self.__oobeButton, extraArgs=[''])
            self.__directObject.accept('oobe-repeat', self.__oobeButton, extraArgs=['-repeat'])
            self.__directObject.accept('oobe-up', self.__oobeButton, extraArgs=['-up'])
        if self.oobeMode:
            if self.oobeCullFrustum is not None:
                self.oobeCull(cam=cam)
            if self.oobeVis:
                self.oobeVis.reparentTo(self.hidden)
            self.mouseInterfaceNode.clearButton(KeyboardButton.shift())
            self.oobeTrackball.detachNode()
            bt = self.buttonThrowers[0].node()
            bt.setSpecificFlag(1)
            bt.setButtonDownEvent('')
            bt.setButtonRepeatEvent('')
            bt.setButtonUpEvent('')
            cam.reparentTo(self.camera)
            self.oobeCamera.reparentTo(self.hidden)
            self.oobeMode = 0
            self.bboard.post('oobeEnabled', False)
        else:
            self.bboard.post('oobeEnabled', True)
            try:
                cameraParent = localAvatar
            except NameError:
                cameraParent = self.camera.getParent()
            self.oobeCamera.reparentTo(cameraParent)
            self.oobeCamera.clearMat()
            self.mouseInterfaceNode.requireButton(KeyboardButton.shift(), True)
            self.oobeTrackball.node().requireButton(KeyboardButton.shift(), False)
            self.oobeTrackball.reparentTo(self.mouseWatcher)
            mat = Mat4.translateMat(0, -10, 3) * self.camera.getMat(cameraParent)
            mat.invertInPlace()
            self.oobeTrackball.node().setMat(mat)
            cam.reparentTo(self.oobeCameraTrackball)
            bt = self.buttonThrowers[0].node()
            bt.setSpecificFlag(0)
            bt.setButtonDownEvent('oobe-down')
            bt.setButtonRepeatEvent('oobe-repeat')
            bt.setButtonUpEvent('oobe-up')
            if self.oobeVis:
                self.oobeVis.reparentTo(self.camera)
            self.oobeMode = 1

    def __oobeButton(self, suffix, button):
        if False:
            print('Hello World!')
        if button.startswith('mouse'):
            return
        messenger.send(button + suffix)

    def oobeCull(self, cam=None):
        if False:
            print('Hello World!')
        '\n        While in OOBE mode (see above), cull the viewing frustum as if\n        it were still attached to our original camera.  This allows us\n        to visualize the effectiveness of our bounding volumes.\n        '
        if cam is None:
            cam = self.cam
        if not getattr(self, 'oobeMode', False):
            self.oobe(cam=cam)
        if self.oobeCullFrustum is None:
            pnode = LensNode('oobeCull')
            pnode.setLens(self.camLens)
            pnode.showFrustum()
            self.oobeCullFrustum = self.camera.attachNewNode(pnode)
            for c in self.camList:
                c.node().setCullCenter(self.oobeCullFrustum)
            if cam.node().isOfType(Camera):
                cam.node().setCullCenter(self.oobeCullFrustum)
            for c in cam.findAllMatches('**/+Camera'):
                c.node().setCullCenter(self.oobeCullFrustum)
        else:
            for c in self.camList:
                c.node().setCullCenter(NodePath())
            if cam.node().isOfType(Camera):
                cam.node().setCullCenter(self.oobeCullFrustum)
            for c in cam.findAllMatches('**/+Camera'):
                c.node().setCullCenter(NodePath())
            self.oobeCullFrustum.removeNode()
            self.oobeCullFrustum = None

    def showCameraFrustum(self):
        if False:
            for i in range(10):
                print('nop')
        self.removeCameraFrustum()
        geom = self.camLens.makeGeometry()
        if geom is not None:
            gn = GeomNode('frustum')
            gn.addGeom(geom)
            self.camFrustumVis = self.camera.attachNewNode(gn)

    def removeCameraFrustum(self):
        if False:
            while True:
                i = 10
        if self.camFrustumVis:
            self.camFrustumVis.removeNode()

    def screenshot(self, namePrefix='screenshot', defaultFilename=1, source=None, imageComment='', blocking=True):
        if False:
            for i in range(10):
                print('nop')
        ' Captures a screenshot from the main window or from the\n        specified window or Texture and writes it to a filename in the\n        current directory (or to a specified directory).\n\n        If defaultFilename is True, the filename is synthesized by\n        appending namePrefix to a default filename suffix (including\n        the filename extension) specified in the Config variable\n        screenshot-filename.  Otherwise, if defaultFilename is False,\n        the entire namePrefix is taken to be the filename to write,\n        and this string should include a suitable filename extension\n        that will be used to determine the type of image file to\n        write.\n\n        Normally, the source is a GraphicsWindow, GraphicsBuffer or\n        DisplayRegion.  If a Texture is supplied instead, it must have\n        a ram image (that is, if it was generated by\n        makeTextureBuffer() or makeCubeMap(), the parameter toRam\n        should have been set true).  If it is a cube map texture as\n        generated by makeCubeMap(), namePrefix should contain the hash\n        mark (\'#\') character.\n\n        Normally, this call will block until the screenshot is fully\n        written.  To write the screenshot in a background thread\n        instead, pass blocking = False.  In this case, the return value\n        is a future that can be awaited.\n\n        A "screenshot" event will be sent once the screenshot is saved.\n\n        :returns: The filename if successful, or None if there is a problem.\n        '
        if source is None:
            source = self.win
        if defaultFilename:
            filename = GraphicsOutput.makeScreenshotFilename(namePrefix)
        else:
            filename = Filename(namePrefix)
        if isinstance(source, Texture):
            if source.getZSize() > 1:
                saved = source.write(filename, 0, 0, 1, 0)
            else:
                saved = source.write(filename)
        elif blocking:
            saved = source.saveScreenshot(filename, imageComment)
        else:
            request = source.saveAsyncScreenshot(filename, imageComment)
            request.addDoneCallback(lambda fut, filename=filename: messenger.send('screenshot', [filename]))
            return request
        if saved:
            messenger.send('screenshot', [filename])
            return filename
        return None

    def saveCubeMap(self, namePrefix='cube_map_#.png', defaultFilename=0, source=None, camera=None, size=128, cameraMask=PandaNode.getAllCameraMask(), sourceLens=None):
        if False:
            i = 10
            return i + 15
        '\n        Similar to :meth:`screenshot()`, this sets up a temporary cube\n        map Texture which it uses to take a series of six snapshots of\n        the current scene, one in each of the six cube map directions.\n        This requires rendering a new frame.\n\n        Unlike `screenshot()`, source may only be a GraphicsWindow,\n        GraphicsBuffer, or DisplayRegion; it may not be a Texture.\n\n        camera should be the node to which the cubemap cameras will be\n        parented.  The default is the camera associated with source,\n        if source is a DisplayRegion, or base.camera otherwise.\n\n        :returns: The filename if successful, or None if there is a problem.\n        '
        if source is None:
            source = self.win
        if camera is None:
            if hasattr(source, 'getCamera'):
                camera = source.getCamera()
            if camera is None:
                camera = self.camera
        if sourceLens is None:
            sourceLens = self.camLens
        if hasattr(source, 'getWindow'):
            source = source.getWindow()
        rig = NodePath(namePrefix)
        buffer = source.makeCubeMap(namePrefix, size, rig, cameraMask, 1)
        if buffer is None:
            raise Exception('Could not make cube map.')
        lens = rig.find('**/+Camera').node().getLens()
        lens.setNearFar(sourceLens.getNear(), sourceLens.getFar())
        rig.reparentTo(camera)
        self.graphicsEngine.openWindows()
        self.graphicsEngine.renderFrame()
        self.graphicsEngine.renderFrame()
        self.graphicsEngine.syncFrame()
        tex = buffer.getTexture()
        saved = self.screenshot(namePrefix=namePrefix, defaultFilename=defaultFilename, source=tex)
        self.graphicsEngine.removeWindow(buffer)
        rig.removeNode()
        return saved

    def saveSphereMap(self, namePrefix='spheremap.png', defaultFilename=0, source=None, camera=None, size=256, cameraMask=PandaNode.getAllCameraMask(), numVertices=1000, sourceLens=None):
        if False:
            return 10
        "\n        This works much like :meth:`saveCubeMap()`, and uses the\n        graphics API's hardware cube-mapping ability to get a 360-degree\n        view of the world.  But then it converts the six cube map faces\n        into a single fisheye texture, suitable for applying as a static\n        environment map (sphere map).\n\n        For eye-relative static environment maps, sphere maps are often\n        preferable to cube maps because they require only a single\n        texture and because they are supported on a broader range of\n        hardware.\n\n        :returns: The filename if successful, or None if there is a problem.\n        "
        if source is None:
            source = self.win
        if camera is None:
            if hasattr(source, 'getCamera'):
                camera = source.getCamera()
            if camera is None:
                camera = self.camera
        if sourceLens is None:
            sourceLens = self.camLens
        if hasattr(source, 'getWindow'):
            source = source.getWindow()
        toSphere = source.makeTextureBuffer(namePrefix, size, size, Texture(), 1)
        rig = NodePath(namePrefix)
        buffer = toSphere.makeCubeMap(namePrefix, size, rig, cameraMask, 0)
        if buffer is None:
            self.graphicsEngine.removeWindow(toSphere)
            raise Exception('Could not make cube map.')
        lens = rig.find('**/+Camera').node().getLens()
        lens.setNearFar(sourceLens.getNear(), sourceLens.getFar())
        dr = toSphere.makeMonoDisplayRegion()
        camNode = Camera('camNode')
        lens = OrthographicLens()
        lens.setFilmSize(2, 2)
        lens.setNearFar(-1000, 1000)
        camNode.setLens(lens)
        root = NodePath('buffer')
        cam = root.attachNewNode(camNode)
        dr.setCamera(cam)
        fm = FisheyeMaker('card')
        fm.setNumVertices(numVertices)
        fm.setSquareInscribed(1, 1.1)
        fm.setReflection(1)
        card = root.attachNewNode(fm.generate())
        card.setTexture(buffer.getTexture())
        rig.reparentTo(camera)
        self.graphicsEngine.openWindows()
        self.graphicsEngine.renderFrame()
        self.graphicsEngine.renderFrame()
        self.graphicsEngine.syncFrame()
        saved = self.screenshot(namePrefix=namePrefix, defaultFilename=defaultFilename, source=toSphere.getTexture())
        self.graphicsEngine.removeWindow(buffer)
        self.graphicsEngine.removeWindow(toSphere)
        rig.removeNode()
        return saved

    def movie(self, namePrefix='movie', duration=1.0, fps=30, format='png', sd=4, source=None):
        if False:
            i = 10
            return i + 15
        "\n        Spawn a task to capture a movie using the screenshot function.\n\n        Args:\n            namePrefix (str): used to form output file names (can\n                include path information (e.g. '/i/beta/frames/myMovie')\n            duration (float): the length of the movie in seconds\n            fps (float): the frame rate of the resulting movie\n            format (str): specifies output file format (e.g. png, bmp)\n            sd (int): specifies number of significant digits for frame\n                count in the output file name (e.g. if sd = 4, the name\n                will be something like movie_0001.png)\n            source: the Window, Buffer, DisplayRegion, or Texture from\n                which to save the resulting images.  The default is the\n                main window.\n\n        Returns:\n            A `~direct.task.Task` that can be awaited.\n        "
        clock = self.clock
        clock.mode = ClockObject.MNonRealTime
        clock.dt = 1.0 / fps
        t = self.taskMgr.add(self._movieTask, namePrefix + '_task')
        t.frameIndex = 0
        t.numFrames = int(duration * fps)
        t.source = source
        t.outputString = namePrefix + '_%0' + repr(sd) + 'd.' + format
        t.setUponDeath(lambda state: clock.setMode(ClockObject.MNormal))
        return t

    def _movieTask(self, state):
        if False:
            print('Hello World!')
        if state.frameIndex != 0:
            frameName = state.outputString % state.frameIndex
            self.notify.info('Capturing frame: ' + frameName)
            self.screenshot(namePrefix=frameName, defaultFilename=0, source=state.source)
        state.frameIndex += 1
        if state.frameIndex > state.numFrames:
            return Task.done
        else:
            return Task.cont

    def windowEvent(self, win):
        if False:
            for i in range(10):
                print('nop')
        if win != self.win:
            return
        properties = win.getProperties()
        if properties != self.__prevWindowProperties:
            self.__prevWindowProperties = properties
            self.notify.debug('Got window event: %s' % repr(properties))
            if not properties.getOpen():
                self.notify.info('User closed main window.')
                if __debug__:
                    if self.__autoGarbageLogging:
                        GarbageReport.b_checkForGarbageLeaks()
                self.userExit()
            if properties.getForeground() and (not self.mainWinForeground):
                self.mainWinForeground = 1
            elif not properties.getForeground() and self.mainWinForeground:
                self.mainWinForeground = 0
                if __debug__:
                    if self.__autoGarbageLogging:
                        GarbageReport.b_checkForGarbageLeaks()
            if properties.getMinimized() and (not self.mainWinMinimized):
                self.mainWinMinimized = 1
                messenger.send('PandaPaused')
            elif not properties.getMinimized() and self.mainWinMinimized:
                self.mainWinMinimized = 0
                messenger.send('PandaRestarted')
            self.adjustWindowAspectRatio(self.getAspectRatio())
            if win.hasSize() and win.getSbsLeftYSize() != 0:
                self.pixel2d.setScale(2.0 / win.getSbsLeftXSize(), 1.0, 2.0 / win.getSbsLeftYSize())
                if self.wantRender2dp:
                    self.pixel2dp.setScale(2.0 / win.getSbsLeftXSize(), 1.0, 2.0 / win.getSbsLeftYSize())
            else:
                (xsize, ysize) = self.getSize()
                if xsize > 0 and ysize > 0:
                    self.pixel2d.setScale(2.0 / xsize, 1.0, 2.0 / ysize)
                    if self.wantRender2dp:
                        self.pixel2dp.setScale(2.0 / xsize, 1.0, 2.0 / ysize)

    def adjustWindowAspectRatio(self, aspectRatio):
        if False:
            i = 10
            return i + 15
        ' This function is normally called internally by\n        `windowEvent()`, but it may also be called to explicitly adjust\n        the aspect ratio of the render/render2d DisplayRegion, by a\n        class that has redefined these. '
        if self.__configAspectRatio:
            aspectRatio = self.__configAspectRatio
        if aspectRatio != self.__oldAspectRatio:
            self.__oldAspectRatio = aspectRatio
            if self.camLens:
                self.camLens.setAspectRatio(aspectRatio)
            if aspectRatio < 1:
                self.aspect2d.setScale(1.0, aspectRatio, aspectRatio)
                self.a2dTop = 1.0 / aspectRatio
                self.a2dBottom = -1.0 / aspectRatio
                self.a2dLeft = -1
                self.a2dRight = 1.0
                if self.wantRender2dp:
                    self.aspect2dp.setScale(1.0, aspectRatio, aspectRatio)
                    self.a2dpTop = 1.0 / aspectRatio
                    self.a2dpBottom = -1.0 / aspectRatio
                    self.a2dpLeft = -1
                    self.a2dpRight = 1.0
            else:
                self.aspect2d.setScale(1.0 / aspectRatio, 1.0, 1.0)
                self.a2dTop = 1.0
                self.a2dBottom = -1.0
                self.a2dLeft = -aspectRatio
                self.a2dRight = aspectRatio
                if self.wantRender2dp:
                    self.aspect2dp.setScale(1.0 / aspectRatio, 1.0, 1.0)
                    self.a2dpTop = 1.0
                    self.a2dpBottom = -1.0
                    self.a2dpLeft = -aspectRatio
                    self.a2dpRight = aspectRatio
            self.a2dTopCenter.setPos(0, 0, self.a2dTop)
            self.a2dTopCenterNs.setPos(0, 0, self.a2dTop)
            self.a2dBottomCenter.setPos(0, 0, self.a2dBottom)
            self.a2dBottomCenterNs.setPos(0, 0, self.a2dBottom)
            self.a2dLeftCenter.setPos(self.a2dLeft, 0, 0)
            self.a2dLeftCenterNs.setPos(self.a2dLeft, 0, 0)
            self.a2dRightCenter.setPos(self.a2dRight, 0, 0)
            self.a2dRightCenterNs.setPos(self.a2dRight, 0, 0)
            self.a2dTopLeft.setPos(self.a2dLeft, 0, self.a2dTop)
            self.a2dTopLeftNs.setPos(self.a2dLeft, 0, self.a2dTop)
            self.a2dTopRight.setPos(self.a2dRight, 0, self.a2dTop)
            self.a2dTopRightNs.setPos(self.a2dRight, 0, self.a2dTop)
            self.a2dBottomLeft.setPos(self.a2dLeft, 0, self.a2dBottom)
            self.a2dBottomLeftNs.setPos(self.a2dLeft, 0, self.a2dBottom)
            self.a2dBottomRight.setPos(self.a2dRight, 0, self.a2dBottom)
            self.a2dBottomRightNs.setPos(self.a2dRight, 0, self.a2dBottom)
            if self.wantRender2dp:
                self.a2dpTopCenter.setPos(0, 0, self.a2dpTop)
                self.a2dpBottomCenter.setPos(0, 0, self.a2dpBottom)
                self.a2dpLeftCenter.setPos(self.a2dpLeft, 0, 0)
                self.a2dpRightCenter.setPos(self.a2dpRight, 0, 0)
                self.a2dpTopLeft.setPos(self.a2dpLeft, 0, self.a2dpTop)
                self.a2dpTopRight.setPos(self.a2dpRight, 0, self.a2dpTop)
                self.a2dpBottomLeft.setPos(self.a2dpLeft, 0, self.a2dpBottom)
                self.a2dpBottomRight.setPos(self.a2dpRight, 0, self.a2dpBottom)
            messenger.send('aspectRatioChanged')

    def userExit(self):
        if False:
            return 10
        if self.exitFunc:
            self.exitFunc()
        self.notify.info('Exiting ShowBase.')
        self.finalizeExit()

    def finalizeExit(self):
        if False:
            i = 10
            return i + 15
        '\n        Called by `userExit()` to quit the application.  The default\n        implementation just calls `sys.exit()`.\n        '
        sys.exit()

    def startWx(self, fWantWx=True):
        if False:
            return 10
        fWantWx = bool(fWantWx)
        if self.wantWx != fWantWx:
            self.wantWx = fWantWx
            if self.wantWx:
                self.spawnWxLoop()

    def spawnWxLoop(self):
        if False:
            for i in range(10):
                print('nop')
        ' Call this method to hand the main loop over to wxPython.\n        This sets up a wxTimer callback so that Panda still gets\n        updated, but wxPython owns the main loop (which seems to make\n        it happier than the other way around). '
        if self.wxAppCreated:
            return
        init_app_for_gui()
        wx = importlib.import_module('wx')
        if not self.wxApp:
            self.wxApp = wx.App(redirect=False)
        if ConfigVariableBool('wx-main-loop', True):
            wxFrameRate = ConfigVariableDouble('wx-frame-rate', 60.0)
            self.wxTimer = wx.Timer(self.wxApp)
            self.wxTimer.Start(int(round(1000.0 / wxFrameRate.value)))
            self.wxApp.Bind(wx.EVT_TIMER, self.__wxTimerCallback)
            self.run = self.wxRun
            self.taskMgr.run = self.wxRun
            builtins.run = self.wxRun
            if self.appRunner:
                self.appRunner.run = self.wxRun
        else:

            def wxLoop(task):
                if False:
                    while True:
                        i = 10
                self.wxApp.Yield()
                while self.wxApp.Pending():
                    self.wxApp.Dispatch()
                return task.again
            self.taskMgr.add(wxLoop, 'wxLoop')
        self.wxAppCreated = True

    def __wxTimerCallback(self, event):
        if False:
            return 10
        if Thread.getCurrentThread().getCurrentTask():
            return
        self.taskMgr.step()

    def wxRun(self):
        if False:
            i = 10
            return i + 15
        ' This method replaces `run()` after we have called `spawnWxLoop()`.\n        Since at this point wxPython now owns the main loop, this method is a\n        call to wxApp.MainLoop(). '
        if Thread.getCurrentThread().getCurrentTask():
            return
        self.wxApp.MainLoop()

    def startTk(self, fWantTk=True):
        if False:
            i = 10
            return i + 15
        fWantTk = bool(fWantTk)
        if self.wantTk != fWantTk:
            self.wantTk = fWantTk
            if self.wantTk:
                self.spawnTkLoop()

    def spawnTkLoop(self):
        if False:
            print('Hello World!')
        ' Call this method to hand the main loop over to Tkinter.\n        This sets up a timer callback so that Panda still gets\n        updated, but Tkinter owns the main loop (which seems to make\n        it happier than the other way around). '
        if self.tkRootCreated:
            return
        tkinter = importlib.import_module('_tkinter')
        Pmw = importlib.import_module('Pmw')
        if not self.tkRoot:
            self.tkRoot = Pmw.initialise()
        builtins.tkroot = self.tkRoot
        init_app_for_gui()
        if self.graphicsEngine.getThreadingModel().getDrawStage() == 0:
            ConfigVariableBool('disable-message-loop', False).value = True
        if ConfigVariableBool('tk-main-loop', True):
            tkFrameRate = ConfigVariableDouble('tk-frame-rate', 60.0)
            self.tkDelay = int(1000.0 / tkFrameRate.value)
            self.tkRoot.after(self.tkDelay, self.__tkTimerCallback)
            self.run = self.tkRun
            self.taskMgr.run = self.tkRun
            builtins.run = self.tkRun
            if self.appRunner:
                self.appRunner.run = self.tkRun
        else:

            def tkLoop(task):
                if False:
                    print('Hello World!')
                while self.tkRoot.dooneevent(tkinter.ALL_EVENTS | tkinter.DONT_WAIT):
                    pass
                return task.again
            self.taskMgr.add(tkLoop, 'tkLoop')
        self.tkRootCreated = True

    def __tkTimerCallback(self):
        if False:
            print('Hello World!')
        if not Thread.getCurrentThread().getCurrentTask():
            self.taskMgr.step()
        self.tkRoot.after(self.tkDelay, self.__tkTimerCallback)

    def tkRun(self):
        if False:
            for i in range(10):
                print('nop')
        ' This method replaces `run()` after we have called `spawnTkLoop()`.\n        Since at this point Tkinter now owns the main loop, this method is a\n        call to tkRoot.mainloop(). '
        if Thread.getCurrentThread().getCurrentTask():
            return
        self.tkRoot.mainloop()

    def startDirect(self, fWantDirect=1, fWantTk=1, fWantWx=0):
        if False:
            while True:
                i = 10
        self.startTk(fWantTk)
        self.startWx(fWantWx)
        if self.wantDirect == fWantDirect:
            return
        self.wantDirect = fWantDirect
        if self.wantDirect:
            DirectSession = importlib.import_module('direct.directtools.DirectSession')
            self.direct = DirectSession.DirectSession()
            self.direct.enable()
            builtins.direct = self.direct
        else:
            builtins.direct = self.direct = None

    def getRepository(self):
        if False:
            while True:
                i = 10
        return None

    def getAxes(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Loads and returns the ``models/misc/xyzAxis.bam`` model.\n\n        :rtype: panda3d.core.NodePath\n        '
        return self.loader.loadModel('models/misc/xyzAxis.bam')

    def __doStartDirect(self):
        if False:
            return 10
        if self.__directStarted:
            return
        self.__directStarted = False
        fTk = ConfigVariableBool('want-tk', False).value
        fWx = ConfigVariableBool('want-wx', False).value
        fDirect = ConfigVariableBool('want-directtools', 0).value or not ConfigVariableString('cluster-mode', '').empty()
        self.startDirect(fWantDirect=fDirect, fWantTk=fTk, fWantWx=fWx)

    def run(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "This method runs the :class:`~direct.task.Task.TaskManager`\n        when ``self.appRunner is None``, which is to say, when we are\n        not running from within a p3d file.  When we *are* within a p3d\n        file, the Panda3D runtime has to be responsible for running the\n        main loop, so we can't allow the application to do it.\n\n        This method must be called from the main thread, otherwise an error is\n        thrown.\n        "
        if Thread.getCurrentThread() != Thread.getMainThread():
            self.notify.error('run() must be called from the main thread.')
            return
        if self.appRunner is None or self.appRunner.dummy or (self.appRunner.interactiveConsole and (not self.appRunner.initialAppImport)):
            self.taskMgr.run()
    make_default_pipe = makeDefaultPipe
    make_module_pipe = makeModulePipe
    make_all_pipes = makeAllPipes
    open_window = openWindow
    close_window = closeWindow
    open_default_window = openDefaultWindow
    open_main_window = openMainWindow
    set_sleep = setSleep
    set_frame_rate_meter = setFrameRateMeter
    set_scene_graph_analyzer_meter = setSceneGraphAnalyzerMeter
    setup_window_controls = setupWindowControls
    setup_render = setupRender
    setup_render2d = setupRender2d
    setup_render2dp = setupRender2dp
    set_aspect_ratio = setAspectRatio
    get_aspect_ratio = getAspectRatio
    get_size = getSize
    make_camera = makeCamera
    make_camera2d = makeCamera2d
    make_camera2dp = makeCamera2dp
    setup_data_graph = setupDataGraph
    setup_mouse = setupMouse
    setup_mouse_cb = setupMouseCB
    enable_software_mouse_pointer = enableSoftwareMousePointer
    detach_input_device = detachInputDevice
    attach_input_device = attachInputDevice
    add_angular_integrator = addAngularIntegrator
    enable_particles = enableParticles
    disable_particles = disableParticles
    toggle_particles = toggleParticles
    create_stats = createStats
    add_sfx_manager = addSfxManager
    enable_music = enableMusic
    enable_sound_effects = enableSoundEffects
    disable_all_audio = disableAllAudio
    enable_all_audio = enableAllAudio
    init_shadow_trav = initShadowTrav
    get_background_color = getBackgroundColor
    set_background_color = setBackgroundColor
    toggle_backface = toggleBackface
    backface_culling_on = backfaceCullingOn
    backface_culling_off = backfaceCullingOff
    toggle_texture = toggleTexture
    texture_on = textureOn
    texture_off = textureOff
    toggle_wireframe = toggleWireframe
    wireframe_on = wireframeOn
    wireframe_off = wireframeOff
    disable_mouse = disableMouse
    enable_mouse = enableMouse
    silence_input = silenceInput
    revive_input = reviveInput
    set_mouse_on_node = setMouseOnNode
    change_mouse_interface = changeMouseInterface
    use_drive = useDrive
    use_trackball = useTrackball
    toggle_tex_mem = toggleTexMem
    toggle_show_vertices = toggleShowVertices
    oobe_cull = oobeCull
    show_camera_frustum = showCameraFrustum
    remove_camera_frustum = removeCameraFrustum
    save_cube_map = saveCubeMap
    save_sphere_map = saveSphereMap
    start_wx = startWx
    start_tk = startTk
    start_direct = startDirect

class WindowControls:

    def __init__(self, win, cam=None, camNode=None, cam2d=None, mouseWatcher=None, mouseKeyboard=None, closeCmd=lambda : 0, grid=None):
        if False:
            while True:
                i = 10
        self.win = win
        self.camera = cam
        if camNode is None and cam is not None:
            camNode = cam.node()
        self.camNode = camNode
        self.camera2d = cam2d
        self.mouseWatcher = mouseWatcher
        self.mouseKeyboard = mouseKeyboard
        self.closeCommand = closeCmd
        self.grid = grid

    def __str__(self):
        if False:
            while True:
                i = 10
        s = 'window = ' + str(self.win) + '\n'
        s += 'camera = ' + str(self.camera) + '\n'
        s += 'camNode = ' + str(self.camNode) + '\n'
        s += 'camera2d = ' + str(self.camera2d) + '\n'
        s += 'mouseWatcher = ' + str(self.mouseWatcher) + '\n'
        s += 'mouseAndKeyboard = ' + str(self.mouseKeyboard) + '\n'
        return s