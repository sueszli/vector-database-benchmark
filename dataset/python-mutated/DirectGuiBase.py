"""
Base class for all DirectGui items.  Handles composite widgets and
command line argument parsing.

Code overview:

1)  Each widget defines a set of options (optiondefs) as a list of tuples
    of the form ``('name', defaultValue, handler)``.
    'name' is the name of the option (used during construction of configure)
    handler can be: None, method, or INITOPT.  If a method is specified, it
    will be called during widget construction (via initialiseoptions), if the
    Handler is specified as an INITOPT, this is an option that can only be set
    during widget construction.

2)  :func:`~DirectGuiBase.defineoptions` is called.  defineoption creates:

    self._constructorKeywords = { keyword: [value, useFlag] }
        A dictionary of the keyword options specified as part of the
        constructor keywords can be of the form 'component_option', where
        component is the name of a widget's component, a component group or a
        component alias.

    self._dynamicGroups
        A list of group names for which it is permissible to specify options
        before components of that group are created.
        If a widget is a derived class the order of execution would be::

          foo.optiondefs = {}
          foo.defineoptions()
            fooParent()
               fooParent.optiondefs = {}
               fooParent.defineoptions()

3)  :func:`~DirectGuiBase.addoptions` is called.  This combines options
    specified as keywords to the widget constructor (stored in
    self._constructorKeywords) with the default options (stored in optiondefs).
    Results are stored in
    ``self._optionInfo = { keyword: [default, current, handler] }``.
    If a keyword is of the form 'component_option' it is left in the
    self._constructorKeywords dictionary (for use by component constructors),
    otherwise it is 'used', and deleted from self._constructorKeywords.

    Notes:

    - constructor keywords override the defaults.
    - derived class default values override parent class defaults
    - derived class handler functions override parent class functions

4)  Superclass initialization methods are called (resulting in nested calls
    to define options (see 2 above)

5)  Widget components are created via calls to
    :func:`~DirectGuiBase.createcomponent`.  User can specify aliases and groups
    for each component created.

    Aliases are alternate names for components, e.g. a widget may have a
    component with a name 'entryField', which itself may have a component
    named 'entry', you could add an alias 'entry' for the 'entryField_entry'
    These are stored in self.__componentAliases.  If an alias is found,
    all keyword entries which use that alias are expanded to their full
    form (to avoid conversion later)

    Groups allow option specifications that apply to all members of the group.
    If a widget has components: 'text1', 'text2', and 'text3' which all belong
    to the 'text' group, they can be all configured with keywords of the form:
    'text_keyword' (e.g. ``text_font='comic.rgb'``).  A component's group
    is stored as the fourth element of its entry in self.__componentInfo.

    Note: the widget constructors have access to all remaining keywords in
    _constructorKeywords (those not transferred to _optionInfo by
    define/addoptions).  If a component defines an alias that applies to
    one of the keywords, that keyword is replaced with a new keyword with
    the alias expanded.

    If a keyword (or substituted alias keyword) is used during creation of the
    component, it is deleted from self._constructorKeywords.  If a group
    keyword applies to the component, that keyword is marked as used, but is
    not deleted from self._constructorKeywords, in case it applies to another
    component.  If any constructor keywords remain at the end of component
    construction (and initialisation), an error is raised.

5)  :func:`~DirectGuiBase.initialiseoptions` is called.  This method calls any
    option handlers to respond to any keyword/default values, then checks to
    see if any keywords are left unused.  If so, an error is raised.
"""
from __future__ import annotations
__all__ = ['DirectGuiBase', 'DirectGuiWidget']
from panda3d.core import ConfigVariableBool, KeyboardButton, MouseWatcherRegion, NodePath, PGFrameStyle, PGItem, Point3, Texture, Vec3
from direct.showbase import ShowBaseGlobal
from direct.showbase.ShowBase import ShowBase
from direct.showbase.MessengerGlobal import messenger
from . import DirectGuiGlobals as DGG
from direct.directtools.DirectUtil import ROUND_TO
from direct.showbase import DirectObject
from direct.task import Task
from direct.task.TaskManagerGlobal import taskMgr
_track_gui_items = ConfigVariableBool('track-gui-items', False)

class DirectGuiBase(DirectObject.DirectObject):
    """Base class of all DirectGUI widgets."""

    def __init__(self):
        if False:
            while True:
                i = 10
        self.guiId = 'guiObject'
        self.postInitialiseFuncList = []
        self.fInit = 1
        self.__componentInfo = {}
        self.__componentAliases = {}

    def defineoptions(self, keywords, optionDefs, dynamicGroups=()):
        if False:
            for i in range(10):
                print('nop')
        ' defineoptions(keywords, optionDefs, dynamicGroups = {}) '
        if not hasattr(self, '_constructorKeywords'):
            tmp = {}
            for (option, value) in keywords.items():
                tmp[option] = [value, 0]
            self._constructorKeywords = tmp
            self._optionInfo = {}
        if not hasattr(self, '_dynamicGroups'):
            self._dynamicGroups = ()
        self._dynamicGroups = self._dynamicGroups + tuple(dynamicGroups)
        self.addoptions(optionDefs, keywords)

    def addoptions(self, optionDefs, optionkeywords):
        if False:
            while True:
                i = 10
        ' addoptions(optionDefs) - add option def to option info '
        optionInfo = self._optionInfo
        optionInfo_has_key = optionInfo.__contains__
        keywords = self._constructorKeywords
        keywords_has_key = keywords.__contains__
        FUNCTION = DGG._OPT_FUNCTION
        for (name, default, function) in optionDefs:
            if '_' not in name:
                default = optionkeywords.get(name, default)
                if not optionInfo_has_key(name):
                    if keywords_has_key(name):
                        value = keywords[name][0]
                        optionInfo[name] = [default, value, function]
                        del keywords[name]
                    else:
                        optionInfo[name] = [default, default, function]
                elif optionInfo[name][FUNCTION] is None:
                    optionInfo[name][FUNCTION] = function
            elif not keywords_has_key(name):
                keywords[name] = [default, 0]

    def initialiseoptions(self, myClass):
        if False:
            while True:
                i = 10
        '\n        Call all initialisation functions to initialize widget\n        options to default of keyword value\n        '
        if self.__class__ is myClass:
            FUNCTION = DGG._OPT_FUNCTION
            self.fInit = 1
            for info in self._optionInfo.values():
                func = info[FUNCTION]
                if func is not None and func is not DGG.INITOPT:
                    func()
            self.fInit = 0
            unusedOptions = []
            keywords = self._constructorKeywords
            for name in keywords:
                used = keywords[name][1]
                if not used:
                    index = name.find('_')
                    if index < 0 or name[:index] not in self._dynamicGroups:
                        unusedOptions.append(name)
            self._constructorKeywords = {}
            if len(unusedOptions) > 0:
                if len(unusedOptions) == 1:
                    text = 'Unknown option "'
                else:
                    text = 'Unknown options "'
                raise KeyError(text + ', '.join(unusedOptions) + '" for ' + myClass.__name__)
            self.postInitialiseFunc()

    def postInitialiseFunc(self):
        if False:
            print('Hello World!')
        for func in self.postInitialiseFuncList:
            func()

    def isinitoption(self, option):
        if False:
            i = 10
            return i + 15
        '\n        Is this opition one that can only be specified at construction?\n        '
        return self._optionInfo[option][DGG._OPT_FUNCTION] is DGG.INITOPT

    def options(self):
        if False:
            i = 10
            return i + 15
        '\n        Print out a list of available widget options.\n        Does not include subcomponent options.\n        '
        options = []
        if hasattr(self, '_optionInfo'):
            for (option, info) in self._optionInfo.items():
                isinit = info[DGG._OPT_FUNCTION] is DGG.INITOPT
                default = info[DGG._OPT_DEFAULT]
                options.append((option, default, isinit))
            options.sort()
        return options

    def configure(self, option=None, **kw):
        if False:
            while True:
                i = 10
        '\n        configure(option = None)\n        Query or configure the megawidget options.\n        '
        if len(kw) == 0:
            if option is None:
                rtn = {}
                for (option, config) in self._optionInfo.items():
                    rtn[option] = (option, config[DGG._OPT_DEFAULT], config[DGG._OPT_VALUE])
                return rtn
            else:
                config = self._optionInfo[option]
                return (option, config[DGG._OPT_DEFAULT], config[DGG._OPT_VALUE])
        optionInfo = self._optionInfo
        optionInfo_has_key = optionInfo.__contains__
        componentInfo = self.__componentInfo
        componentInfo_has_key = componentInfo.__contains__
        componentAliases = self.__componentAliases
        componentAliases_has_key = componentAliases.__contains__
        VALUE = DGG._OPT_VALUE
        FUNCTION = DGG._OPT_FUNCTION
        directOptions = []
        indirectOptions = {}
        indirectOptions_has_key = indirectOptions.__contains__
        for (option, value) in kw.items():
            if optionInfo_has_key(option):
                if optionInfo[option][FUNCTION] is DGG.INITOPT:
                    print('Cannot configure initialisation option "' + option + '" for ' + self.__class__.__name__)
                    break
                optionInfo[option][VALUE] = value
                directOptions.append(option)
            else:
                index = option.find('_')
                if index >= 0:
                    component = option[:index]
                    componentOption = option[index + 1:]
                    if componentAliases_has_key(component):
                        (component, subComponent) = componentAliases[component]
                        if subComponent is not None:
                            componentOption = subComponent + '_' + componentOption
                        option = component + '_' + componentOption
                    if componentInfo_has_key(component):
                        componentConfigFuncs = [componentInfo[component][1]]
                    else:
                        componentConfigFuncs = []
                        for info in componentInfo.values():
                            if info[4] == component:
                                componentConfigFuncs.append(info[1])
                        if len(componentConfigFuncs) == 0 and component not in self._dynamicGroups:
                            raise KeyError('Unknown option "' + option + '" for ' + self.__class__.__name__)
                    for componentConfigFunc in componentConfigFuncs:
                        if not indirectOptions_has_key(componentConfigFunc):
                            indirectOptions[componentConfigFunc] = {}
                        indirectOptions[componentConfigFunc][componentOption] = value
                else:
                    raise KeyError('Unknown option "' + option + '" for ' + self.__class__.__name__)
        for (func, options) in indirectOptions.items():
            func(**options)
        for option in directOptions:
            info = optionInfo[option]
            func = info[DGG._OPT_FUNCTION]
            if func is not None:
                func()

    def __setitem__(self, key, value):
        if False:
            return 10
        self.configure(**{key: value})

    def cget(self, option):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get current configuration setting for this option\n        '
        if option in self._optionInfo:
            return self._optionInfo[option][DGG._OPT_VALUE]
        else:
            index = option.find('_')
            if index >= 0:
                component = option[:index]
                componentOption = option[index + 1:]
                if component in self.__componentAliases:
                    (component, subComponent) = self.__componentAliases[component]
                    if subComponent is not None:
                        componentOption = subComponent + '_' + componentOption
                    option = component + '_' + componentOption
                if component in self.__componentInfo:
                    componentCget = self.__componentInfo[component][3]
                    return componentCget(componentOption)
                else:
                    for info in self.__componentInfo.values():
                        if info[4] == component:
                            componentCget = info[3]
                            return componentCget(componentOption)
        raise KeyError('Unknown option "' + option + '" for ' + self.__class__.__name__)
    __getitem__ = cget

    def createcomponent(self, componentName, componentAliases, componentGroup, widgetClass, *widgetArgs, **kw):
        if False:
            i = 10
            return i + 15
        '\n        Create a component (during construction or later) for this widget.\n        '
        if '_' in componentName:
            raise ValueError('Component name "%s" must not contain "_"' % componentName)
        if hasattr(self, '_constructorKeywords'):
            keywords = self._constructorKeywords
        else:
            keywords = {}
        for (alias, component) in componentAliases:
            index = component.find('_')
            if index < 0:
                self.__componentAliases[alias] = (component, None)
            else:
                mainComponent = component[:index]
                subComponent = component[index + 1:]
                self.__componentAliases[alias] = (mainComponent, subComponent)
            alias = alias + '_'
            aliasLen = len(alias)
            for option in keywords.copy():
                if len(option) > aliasLen and option[:aliasLen] == alias:
                    newkey = component + '_' + option[aliasLen:]
                    keywords[newkey] = keywords[option]
                    del keywords[option]
        componentPrefix = componentName + '_'
        nameLen = len(componentPrefix)
        for option in keywords:
            index = option.find('_')
            if index >= 0 and componentGroup == option[:index]:
                rest = option[index + 1:]
                kw[rest] = keywords[option][0]
                keywords[option][1] = 1
        for option in keywords.copy():
            if len(option) > nameLen and option[:nameLen] == componentPrefix:
                kw[option[nameLen:]] = keywords[option][0]
                del keywords[option]
        if widgetClass is None:
            return None
        if len(widgetArgs) == 1 and isinstance(widgetArgs[0], tuple):
            widgetArgs = widgetArgs[0]
        widget = widgetClass(*widgetArgs, **kw)
        componentClass = widget.__class__.__name__
        self.__componentInfo[componentName] = (widget, widget.configure, componentClass, widget.cget, componentGroup)
        return widget

    def component(self, name):
        if False:
            return 10
        index = name.find('_')
        if index < 0:
            component = name
            remainingComponents = None
        else:
            component = name[:index]
            remainingComponents = name[index + 1:]
        if component in self.__componentAliases:
            (component, subComponent) = self.__componentAliases[component]
            if subComponent is not None:
                if remainingComponents is None:
                    remainingComponents = subComponent
                else:
                    remainingComponents = subComponent + '_' + remainingComponents
        widget = self.__componentInfo[component][0]
        if remainingComponents is None:
            return widget
        else:
            return widget.component(remainingComponents)

    def components(self):
        if False:
            return 10
        return sorted(self.__componentInfo)

    def hascomponent(self, component):
        if False:
            print('Hello World!')
        return component in self.__componentInfo

    def destroycomponent(self, name):
        if False:
            i = 10
            return i + 15
        self.__componentInfo[name][0].destroy()
        del self.__componentInfo[name]

    def destroy(self):
        if False:
            for i in range(10):
                print('nop')
        self.ignoreAll()
        del self._optionInfo
        del self.__componentInfo
        del self.postInitialiseFuncList

    def bind(self, event, command, extraArgs=[]):
        if False:
            while True:
                i = 10
        '\n        Bind the command (which should expect one arg) to the specified\n        event (such as ENTER, EXIT, B1PRESS, B1CLICK, etc.)\n        See DirectGuiGlobals for possible events\n        '
        gEvent = event + self.guiId
        if ConfigVariableBool('debug-directgui-msgs', False):
            from direct.showbase.PythonUtil import StackTrace
            print(gEvent)
            print(StackTrace())
        self.accept(gEvent, command, extraArgs=extraArgs)

    def unbind(self, event):
        if False:
            while True:
                i = 10
        '\n        Unbind the specified event\n        '
        gEvent = event + self.guiId
        self.ignore(gEvent)

def toggleGuiGridSnap():
    if False:
        while True:
            i = 10
    DirectGuiWidget.snapToGrid = 1 - DirectGuiWidget.snapToGrid

def setGuiGridSpacing(spacing):
    if False:
        i = 10
        return i + 15
    DirectGuiWidget.gridSpacing = spacing

class DirectGuiWidget(DirectGuiBase, NodePath):
    snapToGrid = 0
    gridSpacing = 0.05
    guiEdit = ConfigVariableBool('direct-gui-edit', False)
    if guiEdit:
        inactiveInitState = DGG.NORMAL
    else:
        inactiveInitState = DGG.DISABLED
    guiDict: dict[str, DirectGuiWidget] = {}

    def __init__(self, parent=None, **kw):
        if False:
            i = 10
            return i + 15
        optiondefs = (('pgFunc', PGItem, None), ('numStates', 1, None), ('invertedFrames', (), None), ('sortOrder', 0, None), ('state', DGG.NORMAL, self.setState), ('relief', DGG.FLAT, self.setRelief), ('borderWidth', (0.1, 0.1), self.setBorderWidth), ('borderUvWidth', (0.1, 0.1), self.setBorderUvWidth), ('frameSize', None, self.setFrameSize), ('frameColor', (0.8, 0.8, 0.8, 1), self.setFrameColor), ('frameTexture', None, self.setFrameTexture), ('frameVisibleScale', (1, 1), self.setFrameVisibleScale), ('pad', (0, 0), self.resetFrameSize), ('guiId', None, DGG.INITOPT), ('pos', None, DGG.INITOPT), ('hpr', None, DGG.INITOPT), ('scale', None, DGG.INITOPT), ('color', None, DGG.INITOPT), ('suppressMouse', 1, DGG.INITOPT), ('suppressKeys', 0, DGG.INITOPT), ('enableEdit', 1, DGG.INITOPT))
        self.defineoptions(kw, optiondefs)
        DirectGuiBase.__init__(self)
        NodePath.__init__(self)
        self.guiItem = self['pgFunc']('')
        if self['guiId']:
            self.guiItem.setId(self['guiId'])
        self.guiId = self.guiItem.getId()
        if ShowBaseGlobal.__dev__:
            if _track_gui_items:
                if not hasattr(ShowBase, 'guiItems'):
                    ShowBase.guiItems = {}
                if self.guiId in ShowBase.guiItems:
                    ShowBase.notify.warning('duplicate guiId: %s (%s stomping %s)' % (self.guiId, self, ShowBase.guiItems[self.guiId]))
                ShowBase.guiItems[self.guiId] = self
        if parent is None:
            parent = ShowBaseGlobal.aspect2d
        self.assign(parent.attachNewNode(self.guiItem, self['sortOrder']))
        if self['pos']:
            self.setPos(self['pos'])
        if self['hpr']:
            self.setHpr(self['hpr'])
        if self['scale']:
            self.setScale(self['scale'])
        if self['color']:
            self.setColor(self['color'])
        self.setName('%s-%s' % (self.__class__.__name__, self.guiId))
        self.stateNodePath = []
        for i in range(self['numStates']):
            self.stateNodePath.append(NodePath(self.guiItem.getStateDef(i)))
        self.frameStyle = []
        for i in range(self['numStates']):
            self.frameStyle.append(PGFrameStyle())
        self.ll = Point3(0)
        self.ur = Point3(0)
        if self['enableEdit'] and self.guiEdit:
            self.enableEdit()
        suppressFlags = 0
        if self['suppressMouse']:
            suppressFlags |= MouseWatcherRegion.SFMouseButton
            suppressFlags |= MouseWatcherRegion.SFMousePosition
        if self['suppressKeys']:
            suppressFlags |= MouseWatcherRegion.SFOtherButton
        self.guiItem.setSuppressFlags(suppressFlags)
        self.guiDict[self.guiId] = self
        self.postInitialiseFuncList.append(self.frameInitialiseFunc)
        self.initialiseoptions(DirectGuiWidget)

    def frameInitialiseFunc(self):
        if False:
            print('Hello World!')
        self.updateFrameStyle()
        if not self['frameSize']:
            self.resetFrameSize()

    def enableEdit(self):
        if False:
            i = 10
            return i + 15
        self.bind(DGG.B2PRESS, self.editStart)
        self.bind(DGG.B2RELEASE, self.editStop)
        self.bind(DGG.PRINT, self.printConfig)

    def disableEdit(self):
        if False:
            return 10
        self.unbind(DGG.B2PRESS)
        self.unbind(DGG.B2RELEASE)
        self.unbind(DGG.PRINT)

    def editStart(self, event):
        if False:
            while True:
                i = 10
        taskMgr.remove('guiEditTask')
        vWidget2render2d = self.getPos(ShowBaseGlobal.render2d)
        vMouse2render2d = Point3(event.getMouse()[0], 0, event.getMouse()[1])
        editVec = Vec3(vWidget2render2d - vMouse2render2d)
        if base.mouseWatcherNode.getModifierButtons().isDown(KeyboardButton.control()):
            t = taskMgr.add(self.guiScaleTask, 'guiEditTask')
            t.refPos = vWidget2render2d
            t.editVecLen = editVec.length()
            t.initScale = self.getScale()
        else:
            t = taskMgr.add(self.guiDragTask, 'guiEditTask')
            t.editVec = editVec

    def guiScaleTask(self, state):
        if False:
            print('Hello World!')
        mwn = base.mouseWatcherNode
        if mwn.hasMouse():
            vMouse2render2d = Point3(mwn.getMouse()[0], 0, mwn.getMouse()[1])
            newEditVecLen = Vec3(state.refPos - vMouse2render2d).length()
            self.setScale(state.initScale * (newEditVecLen / state.editVecLen))
        return Task.cont

    def guiDragTask(self, state):
        if False:
            for i in range(10):
                print('nop')
        mwn = base.mouseWatcherNode
        if mwn.hasMouse():
            vMouse2render2d = Point3(mwn.getMouse()[0], 0, mwn.getMouse()[1])
            newPos = vMouse2render2d + state.editVec
            self.setPos(ShowBaseGlobal.render2d, newPos)
            if DirectGuiWidget.snapToGrid:
                newPos = self.getPos()
                newPos.set(ROUND_TO(newPos[0], DirectGuiWidget.gridSpacing), ROUND_TO(newPos[1], DirectGuiWidget.gridSpacing), ROUND_TO(newPos[2], DirectGuiWidget.gridSpacing))
                self.setPos(newPos)
        return Task.cont

    def editStop(self, event):
        if False:
            print('Hello World!')
        taskMgr.remove('guiEditTask')

    def setState(self):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self['state'], int):
            self.guiItem.setActive(self['state'])
        elif self['state'] == DGG.NORMAL or self['state'] == 'normal':
            self.guiItem.setActive(1)
        else:
            self.guiItem.setActive(0)

    def resetFrameSize(self):
        if False:
            print('Hello World!')
        if not self.fInit:
            self.setFrameSize(fClearFrame=1)

    def setFrameSize(self, fClearFrame=0):
        if False:
            i = 10
            return i + 15
        frameType = self.getFrameType()
        if self['frameSize']:
            self.bounds = self['frameSize']
            bw = (0, 0)
        else:
            if fClearFrame and frameType != PGFrameStyle.TNone:
                self.frameStyle[0].setType(PGFrameStyle.TNone)
                self.guiItem.setFrameStyle(0, self.frameStyle[0])
                self.guiItem.getStateDef(0)
            self.getBounds()
            if frameType != PGFrameStyle.TNone:
                self.frameStyle[0].setType(frameType)
                self.guiItem.setFrameStyle(0, self.frameStyle[0])
            if frameType != PGFrameStyle.TNone and frameType != PGFrameStyle.TFlat:
                bw = self['borderWidth']
            else:
                bw = (0, 0)
        self.guiItem.setFrame(self.bounds[0] - bw[0], self.bounds[1] + bw[0], self.bounds[2] - bw[1], self.bounds[3] + bw[1])

    def getBounds(self, state=0):
        if False:
            i = 10
            return i + 15
        self.stateNodePath[state].calcTightBounds(self.ll, self.ur)
        vec_right = Vec3.right()
        vec_up = Vec3.up()
        left = vec_right[0] * self.ll[0] + vec_right[1] * self.ll[1] + vec_right[2] * self.ll[2]
        right = vec_right[0] * self.ur[0] + vec_right[1] * self.ur[1] + vec_right[2] * self.ur[2]
        bottom = vec_up[0] * self.ll[0] + vec_up[1] * self.ll[1] + vec_up[2] * self.ll[2]
        top = vec_up[0] * self.ur[0] + vec_up[1] * self.ur[1] + vec_up[2] * self.ur[2]
        self.ll = Point3(left, 0.0, bottom)
        self.ur = Point3(right, 0.0, top)
        self.bounds = [self.ll[0] - self['pad'][0], self.ur[0] + self['pad'][0], self.ll[2] - self['pad'][1], self.ur[2] + self['pad'][1]]
        return self.bounds

    def getWidth(self):
        if False:
            i = 10
            return i + 15
        return self.bounds[1] - self.bounds[0]

    def getHeight(self):
        if False:
            while True:
                i = 10
        return self.bounds[3] - self.bounds[2]

    def getCenter(self):
        if False:
            for i in range(10):
                print('nop')
        x = self.bounds[0] + (self.bounds[1] - self.bounds[0]) / 2.0
        y = self.bounds[2] + (self.bounds[3] - self.bounds[2]) / 2.0
        return (x, y)

    def getFrameType(self, state=0):
        if False:
            print('Hello World!')
        return self.frameStyle[state].getType()

    def updateFrameStyle(self):
        if False:
            while True:
                i = 10
        if not self.fInit:
            for i in range(self['numStates']):
                self.guiItem.setFrameStyle(i, self.frameStyle[i])

    def setRelief(self, fSetStyle=1):
        if False:
            return 10
        relief = self['relief']
        if relief is None:
            relief = PGFrameStyle.TNone
        elif isinstance(relief, str):
            relief = DGG.FrameStyleDict[relief]
        if relief == DGG.RAISED:
            for i in range(self['numStates']):
                if i in self['invertedFrames']:
                    self.frameStyle[1].setType(DGG.SUNKEN)
                else:
                    self.frameStyle[i].setType(DGG.RAISED)
        elif relief == DGG.SUNKEN:
            for i in range(self['numStates']):
                if i in self['invertedFrames']:
                    self.frameStyle[1].setType(DGG.RAISED)
                else:
                    self.frameStyle[i].setType(DGG.SUNKEN)
        else:
            for i in range(self['numStates']):
                self.frameStyle[i].setType(relief)
        self.updateFrameStyle()

    def setFrameColor(self):
        if False:
            while True:
                i = 10
        colors = self['frameColor']
        if isinstance(colors[0], (int, float)):
            colors = (colors,)
        for i in range(self['numStates']):
            if i >= len(colors):
                color = colors[-1]
            else:
                color = colors[i]
            self.frameStyle[i].setColor(color[0], color[1], color[2], color[3])
        self.updateFrameStyle()

    def setFrameTexture(self):
        if False:
            print('Hello World!')
        textures = self['frameTexture']
        if textures is None or isinstance(textures, (Texture, str)):
            textures = (textures,) * self['numStates']
        for i in range(self['numStates']):
            if i >= len(textures):
                texture = textures[-1]
            else:
                texture = textures[i]
            if isinstance(texture, str):
                texture = base.loader.loadTexture(texture)
            if texture:
                self.frameStyle[i].setTexture(texture)
            else:
                self.frameStyle[i].clearTexture()
        self.updateFrameStyle()

    def setFrameVisibleScale(self):
        if False:
            return 10
        scale = self['frameVisibleScale']
        for i in range(self['numStates']):
            self.frameStyle[i].setVisibleScale(scale[0], scale[1])
        self.updateFrameStyle()

    def setBorderWidth(self):
        if False:
            print('Hello World!')
        width = self['borderWidth']
        for i in range(self['numStates']):
            self.frameStyle[i].setWidth(width[0], width[1])
        self.updateFrameStyle()

    def setBorderUvWidth(self):
        if False:
            return 10
        uvWidth = self['borderUvWidth']
        for i in range(self['numStates']):
            self.frameStyle[i].setUvWidth(uvWidth[0], uvWidth[1])
        self.updateFrameStyle()

    def destroy(self):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, 'frameStyle'):
            if ShowBaseGlobal.__dev__:
                if hasattr(ShowBase, 'guiItems'):
                    ShowBase.guiItems.pop(self.guiId, None)
            for child in self.getChildren():
                childGui = self.guiDict.get(child.getName())
                if childGui:
                    childGui.destroy()
                else:
                    parts = child.getName().split('-')
                    simpleChildGui = self.guiDict.get(parts[-1])
                    if simpleChildGui:
                        simpleChildGui.destroy()
            del self.guiDict[self.guiId]
            del self.frameStyle
            self.removeNode()
            for nodePath in self.stateNodePath:
                nodePath.removeNode()
            del self.stateNodePath
            del self.guiItem
            DirectGuiBase.destroy(self)

    def printConfig(self, indent=0):
        if False:
            i = 10
            return i + 15
        space = ' ' * indent
        print('%s%s - %s' % (space, self.guiId, self.__class__.__name__))
        print('%sPos:   %s' % (space, tuple(self.getPos())))
        print('%sScale: %s' % (space, tuple(self.getScale())))
        for child in self.getChildren():
            messenger.send(DGG.PRINT + child.getName(), [indent + 2])

    def copyOptions(self, other):
        if False:
            return 10
        "\n        Copy other's options into our self so we look and feel like other\n        "
        for (key, value) in other._optionInfo.items():
            self[key] = value[1]

    def taskName(self, idString):
        if False:
            i = 10
            return i + 15
        return idString + '-' + str(self.guiId)

    def uniqueName(self, idString):
        if False:
            for i in range(10):
                print('nop')
        return idString + '-' + str(self.guiId)

    def setProp(self, propString, value):
        if False:
            while True:
                i = 10
        "\n        Allows you to set a property like frame['text'] = 'Joe' in\n        a function instead of an assignment.\n        This is useful for setting properties inside function intervals\n        where must input a function and extraArgs, not an assignment.\n        "
        self[propString] = value