import re
import weakref
from collections import OrderedDict
from .. import functions as fn
from ..Qt import QtCore
from .ParameterItem import ParameterItem
PARAM_TYPES = {}
PARAM_NAMES = {}
_PARAM_ITEM_TYPES = {}

def registerParameterItemType(name, itemCls, parameterCls=None, override=False):
    if False:
        return 10
    '\n    Similar to :func:`registerParameterType`, but works on ParameterItems. This is useful for Parameters where the\n    `itemClass` does all the heavy lifting, and a redundant Parameter class must be defined just to house `itemClass`.\n    Instead, use `registerParameterItemType`. If this should belong to a subclass of `Parameter`, specify which one\n    in `parameterCls`.\n    '
    global _PARAM_ITEM_TYPES
    if name in _PARAM_ITEM_TYPES and (not override):
        raise Exception("Parameter item type '%s' already exists (use override=True to replace)" % name)
    parameterCls = parameterCls or Parameter
    _PARAM_ITEM_TYPES[name] = itemCls
    registerParameterType(name, parameterCls, override)

def registerParameterType(name, cls, override=False):
    if False:
        for i in range(10):
            print('nop')
    'Register a parameter type in the parametertree system.\n\n    This enables construction of custom Parameter classes by name in\n    :meth:`~pyqtgraph.parametertree.Parameter.create`.\n    '
    global PARAM_TYPES
    if name in PARAM_TYPES and (not override):
        raise Exception("Parameter type '%s' already exists (use override=True to replace)" % name)
    PARAM_TYPES[name] = cls
    PARAM_NAMES[cls] = name

def __reload__(old):
    if False:
        for i in range(10):
            print('nop')
    PARAM_TYPES.update(old.get('PARAM_TYPES', {}))
    PARAM_NAMES.update(old.get('PARAM_NAMES', {}))

class Parameter(QtCore.QObject):
    """
    A Parameter is the basic unit of data in a parameter tree. Each parameter has
    a name, a type, a value, and several other properties that modify the behavior of the 
    Parameter. Parameters may have parent / child / sibling relationships to construct
    organized hierarchies. Parameters generally do not have any inherent GUI or visual
    interpretation; instead they manage ParameterItem instances which take care of
    display and user interaction.
    
    Note: It is fairly uncommon to use the Parameter class directly; mostly you 
    will use subclasses which provide specialized type and data handling. The static
    pethod Parameter.create(...) is an easy way to generate instances of these subclasses.
       
    For more Parameter types, see ParameterTree.parameterTypes module.
    
    ===================================  =========================================================
    **Signals:**
    sigStateChanged(self, change, info)  Emitted when anything changes about this parameter at 
                                         all.
                                         The second argument is a string indicating what changed 
                                         ('value', 'childAdded', etc..)
                                         The third argument can be any extra information about 
                                         the change
    sigTreeStateChanged(self, changes)   Emitted when any child in the tree changes state
                                         (but only if monitorChildren() is called)
                                         the format of *changes* is [(param, change, info), ...]
    sigValueChanged(self, value)         Emitted when value is finished changing
    sigValueChanging(self, value)        Emitted immediately for all value changes, 
                                         including during editing.
    sigChildAdded(self, child, index)    Emitted when a child is added
    sigChildRemoved(self, child)         Emitted when a child is removed
    sigRemoved(self)                     Emitted when this parameter is removed
    sigParentChanged(self, parent)       Emitted when this parameter's parent has changed
    sigLimitsChanged(self, limits)       Emitted when this parameter's limits have changed
    sigDefaultChanged(self, default)     Emitted when this parameter's default value has changed
    sigNameChanged(self, name)           Emitted when this parameter's name has changed
    sigOptionsChanged(self, opts)        Emitted when any of this parameter's options have changed
    sigContextMenu(self, name)           Emitted when a context menu was clicked
    ===================================  =========================================================
    """
    itemClass = None
    sigValueChanged = QtCore.Signal(object, object)
    sigValueChanging = QtCore.Signal(object, object)
    sigChildAdded = QtCore.Signal(object, object, object)
    sigChildRemoved = QtCore.Signal(object, object)
    sigRemoved = QtCore.Signal(object)
    sigParentChanged = QtCore.Signal(object, object)
    sigLimitsChanged = QtCore.Signal(object, object)
    sigDefaultChanged = QtCore.Signal(object, object)
    sigNameChanged = QtCore.Signal(object, object)
    sigOptionsChanged = QtCore.Signal(object, object)
    sigStateChanged = QtCore.Signal(object, object, object)
    sigTreeStateChanged = QtCore.Signal(object, object)
    sigContextMenu = QtCore.Signal(object, object)

    @staticmethod
    def create(**opts):
        if False:
            for i in range(10):
                print('nop')
        "\n        Static method that creates a new Parameter (or subclass) instance using \n        opts['type'] to select the appropriate class.\n        \n        All options are passed directly to the new Parameter's __init__ method.\n        Use registerParameterType() to add new class types.\n        "
        typ = opts.get('type', None)
        if typ is None:
            cls = Parameter
        else:
            cls = PARAM_TYPES[opts['type']]
        return cls(**opts)

    def __init__(self, **opts):
        if False:
            return 10
        "\n        Initialize a Parameter object. Although it is rare to directly create a\n        Parameter instance, the options available to this method are also allowed\n        by most Parameter subclasses.\n        \n        =======================      =========================================================\n        **Keyword Arguments:**\n        name                         The name to give this Parameter. This is the name that\n                                     will appear in the left-most column of a ParameterTree\n                                     for this Parameter.\n        value                        The value to initially assign to this Parameter.\n        default                      The default value for this Parameter (most Parameters\n                                     provide an option to 'reset to default').\n        children                     A list of children for this Parameter. Children\n                                     may be given either as a Parameter instance or as a\n                                     dictionary to pass to Parameter.create(). In this way,\n                                     it is possible to specify complex hierarchies of\n                                     Parameters from a single nested data structure.\n        readonly                     If True, the user will not be allowed to edit this\n                                     Parameter. (default=False)\n        enabled                      If False, any widget(s) for this parameter will appear\n                                     disabled. (default=True)\n        visible                      If False, the Parameter will not appear when displayed\n                                     in a ParameterTree. (default=True)\n        renamable                    If True, the user may rename this Parameter.\n                                     (default=False)\n        removable                    If True, the user may remove this Parameter.\n                                     (default=False)\n        expanded                     If True, the Parameter will initially be expanded in\n                                     ParameterTrees: Its children will be visible.\n                                     (default=True)\n        syncExpanded                 If True, the `expanded` state of this Parameter is\n                                     synchronized with all ParameterTrees it is displayed in.\n                                     (default=False)\n        title                        (str or None) If specified, then the parameter will be \n                                     displayed to the user using this string as its name. \n                                     However, the parameter will still be referred to \n                                     internally using the *name* specified above. Note that\n                                     this option is not compatible with renamable=True.\n                                     (default=None; added in version 0.9.9)\n        =======================      =========================================================\n        "
        QtCore.QObject.__init__(self)
        self.opts = {'type': None, 'readonly': False, 'visible': True, 'enabled': True, 'renamable': False, 'removable': False, 'strictNaming': False, 'expanded': True, 'syncExpanded': False, 'title': None}
        value = opts.get('value', None)
        name = opts.get('name', None)
        self.opts.update(opts)
        self.opts['value'] = None
        self.opts['name'] = None
        self.childs = []
        self.names = {}
        self.items = weakref.WeakKeyDictionary()
        self._parent = None
        self.treeStateChanges = []
        self.blockTreeChangeEmit = 0
        if not isinstance(name, str):
            raise Exception('Parameter must have a string name specified in opts.')
        self.setName(name)
        self.addChildren(self.opts.pop('children', []))
        if value is not None:
            self.setValue(value)
        if 'default' not in self.opts:
            self.opts['default'] = None
            self.setDefault(self.opts['value'])
        self.sigValueChanged.connect(self._emitValueChanged)
        self.sigChildAdded.connect(self._emitChildAddedChanged)
        self.sigChildRemoved.connect(self._emitChildRemovedChanged)
        self.sigParentChanged.connect(self._emitParentChanged)
        self.sigLimitsChanged.connect(self._emitLimitsChanged)
        self.sigDefaultChanged.connect(self._emitDefaultChanged)
        self.sigNameChanged.connect(self._emitNameChanged)
        self.sigOptionsChanged.connect(self._emitOptionsChanged)
        self.sigContextMenu.connect(self._emitContextMenuChanged)

    def name(self):
        if False:
            print('Hello World!')
        'Return the name of this Parameter.'
        return self.opts['name']

    def title(self):
        if False:
            while True:
                i = 10
        'Return the title of this Parameter.\n        \n        By default, the title is the same as the name unless it has been explicitly specified\n        otherwise.'
        title = self.opts.get('title', None)
        if title is None:
            title = self.name()
        return title

    def contextMenu(self, name):
        if False:
            print('Hello World!')
        '"A context menu entry was clicked'
        self.sigContextMenu.emit(self, name)

    def setName(self, name):
        if False:
            print('Hello World!')
        'Attempt to change the name of this parameter; return the actual name. \n        (The parameter may reject the name change or automatically pick a different name)'
        if self.opts['strictNaming']:
            if len(name) < 1 or re.search('\\W', name) or re.match('\\d', name[0]):
                raise Exception("Parameter name '%s' is invalid. (Must contain only alphanumeric and underscore characters and may not start with a number)" % name)
        parent = self.parent()
        if parent is not None:
            name = parent._renameChild(self, name)
        if self.opts['name'] != name:
            self.opts['name'] = name
            self.sigNameChanged.emit(self, name)
        return name

    def type(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the type string for this Parameter.'
        return self.opts['type']

    def isType(self, typ):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return True if this parameter type matches the name *typ*.\n        This can occur either of two ways:\n        \n          - If self.type() == *typ*\n          - If this parameter's class is registered with the name *typ*\n        "
        if self.type() == typ:
            return True
        global PARAM_TYPES
        cls = PARAM_TYPES.get(typ, None)
        if cls is None:
            raise Exception("Type name '%s' is not registered." % str(typ))
        return self.__class__ is cls

    def childPath(self, child):
        if False:
            return 10
        '\n        Return the path of parameter names from self to child.\n        If child is not a (grand)child of self, return None.\n        '
        path = []
        while child is not self:
            path.insert(0, child.name())
            child = child.parent()
            if child is None:
                return None
        return path

    def setValue(self, value, blockSignal=None):
        if False:
            return 10
        '\n        Set the value of this Parameter; return the actual value that was set.\n        (this may be different from the value that was requested)\n        '
        try:
            if blockSignal is not None:
                self.sigValueChanged.disconnect(blockSignal)
            value = self._interpretValue(value)
            if fn.eq(self.opts['value'], value):
                return value
            self.opts['value'] = value
            self.sigValueChanged.emit(self, value)
        finally:
            if blockSignal is not None:
                self.sigValueChanged.connect(blockSignal)
        return self.opts['value']

    def _interpretValue(self, v):
        if False:
            i = 10
            return i + 15
        return v

    def value(self):
        if False:
            while True:
                i = 10
        '\n        Return the value of this Parameter.\n        '
        return self.opts['value']

    def getValues(self):
        if False:
            while True:
                i = 10
        'Return a tree of all values that are children of this parameter'
        vals = OrderedDict()
        for ch in self:
            vals[ch.name()] = (ch.value(), ch.getValues())
        return vals

    def saveState(self, filter=None):
        if False:
            while True:
                i = 10
        "\n        Return a structure representing the entire state of the parameter tree.\n        The tree state may be restored from this structure using restoreState().\n\n        If *filter* is set to 'user', then only user-settable data will be included in the\n        returned state.\n        "
        if filter is None:
            state = self.opts.copy()
            if state['type'] is None:
                global PARAM_NAMES
                state['type'] = PARAM_NAMES.get(type(self), None)
        elif filter == 'user':
            state = {'value': self.value()}
        else:
            raise ValueError("Unrecognized filter argument: '%s'" % filter)
        ch = OrderedDict([(ch.name(), ch.saveState(filter=filter)) for ch in self])
        if len(ch) > 0:
            state['children'] = ch
        return state

    def restoreState(self, state, recursive=True, addChildren=True, removeChildren=True, blockSignals=True):
        if False:
            i = 10
            return i + 15
        '\n        Restore the state of this parameter and its children from a structure generated using saveState()\n        If recursive is True, then attempt to restore the state of child parameters as well.\n        If addChildren is True, then any children which are referenced in the state object will be\n        created if they do not already exist.\n        If removeChildren is True, then any children which are not referenced in the state object will \n        be removed.\n        If blockSignals is True, no signals will be emitted until the tree has been completely restored. \n        This prevents signal handlers from responding to a partially-rebuilt network.\n        '
        state = state.copy()
        childState = state.pop('children', [])
        if isinstance(childState, dict):
            cs = []
            for (k, v) in childState.items():
                cs.append(v.copy())
                cs[-1].setdefault('name', k)
            childState = cs
        if blockSignals:
            self.blockTreeChangeSignal()
        try:
            self.setOpts(**state)
            if not recursive:
                return
            ptr = 0
            foundChilds = set()
            for ch in childState:
                name = ch['name']
                gotChild = False
                for (i, ch2) in enumerate(self.childs[ptr:]):
                    if ch2.name() != name:
                        continue
                    gotChild = True
                    if i != 0:
                        self.insertChild(ptr, ch2)
                    ch2.restoreState(ch, recursive=recursive, addChildren=addChildren, removeChildren=removeChildren)
                    foundChilds.add(ch2)
                    break
                if not gotChild:
                    if not addChildren:
                        continue
                    ch2 = Parameter.create(**ch)
                    self.insertChild(ptr, ch2)
                    foundChilds.add(ch2)
                ptr += 1
            if removeChildren:
                for ch in self.childs[:]:
                    if ch not in foundChilds:
                        self.removeChild(ch)
        finally:
            if blockSignals:
                self.unblockTreeChangeSignal()

    def defaultValue(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the default value for this parameter.'
        return self.opts['default']

    def setDefault(self, val):
        if False:
            return 10
        'Set the default value for this parameter.'
        if self.opts['default'] == val:
            return
        self.opts['default'] = val
        self.sigDefaultChanged.emit(self, val)

    def setToDefault(self):
        if False:
            i = 10
            return i + 15
        "Set this parameter's value to the default."
        if self.hasDefault():
            self.setValue(self.defaultValue())

    def hasDefault(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns True if this parameter has a default value.'
        return self.opts['default'] is not None

    def valueIsDefault(self):
        if False:
            while True:
                i = 10
        "Returns True if this parameter's value is equal to the default value."
        return fn.eq(self.value(), self.defaultValue())

    def setLimits(self, limits):
        if False:
            for i in range(10):
                print('nop')
        'Set limits on the acceptable values for this parameter. \n        The format of limits depends on the type of the parameter and\n        some parameters do not make use of limits at all.'
        if 'limits' in self.opts and fn.eq(self.opts['limits'], limits):
            return
        self.opts['limits'] = limits
        self.sigLimitsChanged.emit(self, limits)
        return limits

    def writable(self):
        if False:
            i = 10
            return i + 15
        "\n        Returns True if this parameter's value can be changed by the user.\n        Note that the value of the parameter can *always* be changed by\n        calling setValue().\n        "
        return not self.readonly()

    def setWritable(self, writable=True):
        if False:
            return 10
        'Set whether this Parameter should be editable by the user. (This is \n        exactly the opposite of setReadonly).'
        self.setOpts(readonly=not writable)

    def readonly(self):
        if False:
            i = 10
            return i + 15
        '\n        Return True if this parameter is read-only. (this is the opposite of writable())\n        '
        return self.opts.get('readonly', False)

    def setReadonly(self, readonly=True):
        if False:
            for i in range(10):
                print('nop')
        "Set whether this Parameter's value may be edited by the user\n        (this is the opposite of setWritable())."
        self.setOpts(readonly=readonly)

    def setOpts(self, **opts):
        if False:
            print('Hello World!')
        '\n        Set any arbitrary options on this parameter.\n        The exact behavior of this function will depend on the parameter type, but\n        most parameters will accept a common set of options: value, name, limits,\n        default, readonly, removable, renamable, visible, enabled, expanded and syncExpanded.\n        \n        See :func:`Parameter.__init__ <pyqtgraph.parametertree.Parameter.__init__>`\n        for more information on default options.\n        '
        changed = OrderedDict()
        for k in opts:
            if k == 'value':
                self.setValue(opts[k])
            elif k == 'name':
                self.setName(opts[k])
            elif k == 'limits':
                self.setLimits(opts[k])
            elif k == 'default':
                self.setDefault(opts[k])
            elif k not in self.opts or not fn.eq(self.opts[k], opts[k]):
                self.opts[k] = opts[k]
                changed[k] = opts[k]
        if len(changed) > 0:
            self.sigOptionsChanged.emit(self, changed)

    def emitStateChanged(self, changeDesc, data):
        if False:
            while True:
                i = 10
        self.sigStateChanged.emit(self, changeDesc, data)
        self.treeStateChanges.append((self, changeDesc, data))
        self.emitTreeChanges()

    def _emitValueChanged(self, param, data):
        if False:
            print('Hello World!')
        self.emitStateChanged('value', data)

    def _emitChildAddedChanged(self, param, *data):
        if False:
            i = 10
            return i + 15
        self.emitStateChanged('childAdded', data)

    def _emitChildRemovedChanged(self, param, data):
        if False:
            for i in range(10):
                print('nop')
        self.emitStateChanged('childRemoved', data)

    def _emitParentChanged(self, param, data):
        if False:
            i = 10
            return i + 15
        self.emitStateChanged('parent', data)

    def _emitLimitsChanged(self, param, data):
        if False:
            return 10
        self.emitStateChanged('limits', data)

    def _emitDefaultChanged(self, param, data):
        if False:
            return 10
        self.emitStateChanged('default', data)

    def _emitNameChanged(self, param, data):
        if False:
            for i in range(10):
                print('nop')
        self.emitStateChanged('name', data)

    def _emitOptionsChanged(self, param, data):
        if False:
            for i in range(10):
                print('nop')
        self.emitStateChanged('options', data)

    def _emitContextMenuChanged(self, param, data):
        if False:
            return 10
        self.emitStateChanged('contextMenu', data)

    def makeTreeItem(self, depth):
        if False:
            print('Hello World!')
        '\n        Return a TreeWidgetItem suitable for displaying/controlling the content of \n        this parameter. This is called automatically when a ParameterTree attempts\n        to display this Parameter.\n        Most subclasses will want to override this function.\n        '
        itemClass = self.itemClass or _PARAM_ITEM_TYPES.get(self.opts['type'], ParameterItem)
        return itemClass(self, depth)

    def addChild(self, child, autoIncrementName=None, existOk=False):
        if False:
            i = 10
            return i + 15
        "\n        Add another parameter to the end of this parameter's child list.\n        \n        See insertChild() for a description of the *autoIncrementName* and *existOk*\n        arguments.\n        "
        return self.insertChild(len(self.childs), child, autoIncrementName=autoIncrementName, existOk=existOk)

    def addChildren(self, children):
        if False:
            print('Hello World!')
        '\n        Add a list or dict of children to this parameter. This method calls\n        addChild once for each value in *children*.\n        '
        if isinstance(children, dict):
            ch2 = []
            for (name, opts) in children.items():
                if isinstance(opts, dict) and 'name' not in opts:
                    opts = opts.copy()
                    opts['name'] = name
                ch2.append(opts)
            children = ch2
        for chOpts in children:
            self.addChild(chOpts)

    def insertChild(self, pos, child, autoIncrementName=None, existOk=False):
        if False:
            return 10
        "\n        Insert a new child at pos.\n        If pos is a Parameter, then insert at the position of that Parameter.\n        If child is a dict, then a parameter is constructed using\n        :func:`Parameter.create <pyqtgraph.parametertree.Parameter.create>`.\n        \n        By default, the child's 'autoIncrementName' option determines whether\n        the name will be adjusted to avoid prior name collisions. This \n        behavior may be overridden by specifying the *autoIncrementName* \n        argument. This argument was added in version 0.9.9.\n\n        If 'autoIncrementName' is *False*, an error is raised when the inserted child already exists. However, if\n        'existOk' is *True*, the existing child will be returned instead, and this child will *not* be inserted.\n        "
        if isinstance(child, dict):
            child = Parameter.create(**child)
        name = child.name()
        if name in self.names and child is not self.names[name]:
            if autoIncrementName is True or (autoIncrementName is None and child.opts.get('autoIncrementName', False)):
                name = self.incrementName(name)
                child.setName(name)
            elif existOk:
                return self.names[name]
            else:
                raise ValueError('Already have child named %s' % str(name))
        if isinstance(pos, Parameter):
            pos = self.childs.index(pos)
        with self.treeChangeBlocker():
            if child.parent() is not None:
                child.remove()
            self.names[name] = child
            self.childs.insert(pos, child)
            child.parentChanged(self)
            child.sigTreeStateChanged.connect(self.treeStateChanged)
            self.sigChildAdded.emit(self, child, pos)
        return child

    def removeChild(self, child):
        if False:
            i = 10
            return i + 15
        'Remove a child parameter.'
        name = child.name()
        if name not in self.names or self.names[name] is not child:
            raise Exception("Parameter %s is not my child; can't remove." % str(child))
        del self.names[name]
        self.childs.pop(self.childs.index(child))
        child.parentChanged(None)
        try:
            child.sigTreeStateChanged.disconnect(self.treeStateChanged)
        except (TypeError, RuntimeError):
            pass
        self.sigChildRemoved.emit(self, child)

    def clearChildren(self):
        if False:
            while True:
                i = 10
        'Remove all child parameters.'
        for ch in self.childs[:]:
            self.removeChild(ch)

    def children(self):
        if False:
            i = 10
            return i + 15
        "Return a list of this parameter's children.\n        Warning: this overrides QObject.children\n        "
        return self.childs[:]

    def hasChildren(self):
        if False:
            return 10
        'Return True if this Parameter has children.'
        return len(self.childs) > 0

    def parentChanged(self, parent):
        if False:
            return 10
        "This method is called when the parameter's parent has changed.\n        It may be useful to extend this method in subclasses."
        self._parent = parent
        self.sigParentChanged.emit(self, parent)

    def parent(self):
        if False:
            i = 10
            return i + 15
        'Return the parent of this parameter.'
        return self._parent

    def remove(self):
        if False:
            return 10
        "Remove this parameter from its parent's child list"
        parent = self.parent()
        if parent is None:
            raise Exception('Cannot remove; no parent.')
        parent.removeChild(self)
        self.sigRemoved.emit(self)

    def incrementName(self, name):
        if False:
            i = 10
            return i + 15
        (base, num) = re.match('([^\\d]*)(\\d*)', name).groups()
        numLen = len(num)
        if numLen == 0:
            num = 2
            numLen = 1
        else:
            num = int(num)
        while True:
            newName = base + '%%0%dd' % numLen % num
            if newName not in self.names:
                return newName
            num += 1

    def __iter__(self):
        if False:
            while True:
                i = 10
        for ch in self.childs:
            yield ch

    def __getitem__(self, names):
        if False:
            while True:
                i = 10
        "Get the value of a child parameter. The name may also be a tuple giving\n        the path to a sub-parameter::\n        \n            value = param[('child', 'grandchild')]\n        "
        if not isinstance(names, tuple):
            names = (names,)
        return self.param(*names).value()

    def __setitem__(self, names, value):
        if False:
            for i in range(10):
                print('nop')
        "Set the value of a child parameter. The name may also be a tuple giving\n        the path to a sub-parameter::\n        \n            param[('child', 'grandchild')] = value\n        "
        if isinstance(names, str):
            names = (names,)
        return self.param(*names).setValue(value)

    def keys(self):
        if False:
            while True:
                i = 10
        return self.names

    def child(self, *names):
        if False:
            while True:
                i = 10
        "Return a child parameter. \n        Accepts the name of the child or a tuple (path, to, child)\n\n        Added in version 0.9.9. Earlier versions used the 'param' method, which is still\n        implemented for backward compatibility.\n        "
        try:
            param = self.names[names[0]]
        except KeyError:
            raise KeyError('Parameter %s has no child named %s' % (self.name(), names[0]))
        if len(names) > 1:
            return param.child(*names[1:])
        else:
            return param

    def param(self, *names):
        if False:
            for i in range(10):
                print('nop')
        return self.child(*names)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return "<%s '%s' at 0x%x>" % (self.__class__.__name__, self.name(), id(self))

    def _renameChild(self, child, name):
        if False:
            return 10
        if name in self.names:
            return child.name()
        self.names[name] = child
        del self.names[child.name()]
        return name

    def registerItem(self, item):
        if False:
            for i in range(10):
                print('nop')
        self.items[item] = None

    def hide(self):
        if False:
            while True:
                i = 10
        'Hide this parameter. It and its children will no longer be visible in any ParameterTree\n        widgets it is connected to.'
        self.show(False)

    def show(self, s=True):
        if False:
            print('Hello World!')
        'Show this parameter. '
        self.opts['visible'] = s
        self.sigOptionsChanged.emit(self, {'visible': s})

    def treeChangeBlocker(self):
        if False:
            print('Hello World!')
        '\n        Return an object that can be used to temporarily block and accumulate\n        sigTreeStateChanged signals. This is meant to be used when numerous changes are \n        about to be made to the tree and only one change signal should be\n        emitted at the end.\n        \n        Example::\n\n            with param.treeChangeBlocker():\n                param.addChild(...)\n                param.removeChild(...)\n                param.setValue(...)\n        '
        return SignalBlocker(self.blockTreeChangeSignal, self.unblockTreeChangeSignal)

    def blockTreeChangeSignal(self):
        if False:
            return 10
        '\n        Used to temporarily block and accumulate tree change signals.\n        *You must remember to unblock*, so it is advisable to use treeChangeBlocker() instead.\n        '
        self.blockTreeChangeEmit += 1

    def unblockTreeChangeSignal(self):
        if False:
            i = 10
            return i + 15
        'Unblocks enission of sigTreeStateChanged and flushes the changes out through a single signal.'
        self.blockTreeChangeEmit -= 1
        self.emitTreeChanges()

    def treeStateChanged(self, param, changes):
        if False:
            while True:
                i = 10
        '\n        Called when the state of any sub-parameter has changed. \n        \n        ==============  ================================================================\n        **Arguments:**\n        param           The immediate child whose tree state has changed.\n                        note that the change may have originated from a grandchild.\n        changes         List of tuples describing all changes that have been made\n                        in this event: (param, changeDescr, data)\n        ==============  ================================================================\n                     \n        This function can be extended to react to tree state changes.\n        '
        self.treeStateChanges.extend(changes)
        self.emitTreeChanges()

    def emitTreeChanges(self):
        if False:
            while True:
                i = 10
        if self.blockTreeChangeEmit == 0:
            changes = self.treeStateChanges
            self.treeStateChanges = []
            if len(changes) > 0:
                self.sigTreeStateChanged.emit(self, changes)

class SignalBlocker(object):

    def __init__(self, enterFn, exitFn):
        if False:
            print('Hello World!')
        self.enterFn = enterFn
        self.exitFn = exitFn

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        self.enterFn()

    def __exit__(self, exc_type, exc_value, tb):
        if False:
            i = 10
            return i + 15
        self.exitFn()