__all__ = ['ParameterSystem', 'SystemSolver']
from .. import functions as fn
from .parameterTypes import GroupParameter
from .SystemSolver import SystemSolver

class ParameterSystem(GroupParameter):
    """
    ParameterSystem is a subclass of GroupParameter that manages a tree of 
    sub-parameters with a set of interdependencies--changing any one parameter
    may affect other parameters in the system.
    
    See parametertree/SystemSolver for more information.
    
    NOTE: This API is experimental and may change substantially across minor 
    version numbers. 
    """

    def __init__(self, *args, **kwds):
        if False:
            print('Hello World!')
        GroupParameter.__init__(self, *args, **kwds)
        self._system = None
        self._fixParams = []
        sys = kwds.pop('system', None)
        if sys is not None:
            self.setSystem(sys)
        self._ignoreChange = []
        self.sigTreeStateChanged.connect(self.updateSystem)

    def setSystem(self, sys):
        if False:
            i = 10
            return i + 15
        self._system = sys
        defaults = {}
        vals = {}
        for param in self:
            name = param.name()
            constraints = ''
            if hasattr(sys, '_' + name):
                constraints += 'n'
            if not param.readonly():
                constraints += 'f'
                if 'n' in constraints:
                    ch = param.addChild(dict(name='fixed', type='bool', value=False))
                    self._fixParams.append(ch)
                    param.setReadonly(True)
                    param.setOpts(expanded=False)
                else:
                    vals[name] = param.value()
                    ch = param.addChild(dict(name='fixed', type='bool', value=True, readonly=True))
            defaults[name] = [None, param.type(), None, constraints]
        sys.defaultState.update(defaults)
        sys.reset()
        for (name, value) in vals.items():
            setattr(sys, name, value)
        self.updateAllParams()

    def updateSystem(self, param, changes):
        if False:
            for i in range(10):
                print('nop')
        changes = [ch for ch in changes if ch[0] not in self._ignoreChange]
        sets = [ch[0] for ch in changes if ch[1] == 'value']
        for param in sets:
            if param in self._fixParams:
                parent = param.parent()
                if param.value():
                    setattr(self._system, parent.name(), parent.value())
                else:
                    setattr(self._system, parent.name(), None)
            else:
                setattr(self._system, param.name(), param.value())
        self.updateAllParams()

    def updateAllParams(self):
        if False:
            i = 10
            return i + 15
        try:
            self.sigTreeStateChanged.disconnect(self.updateSystem)
            for (name, state) in self._system._vars.items():
                param = self.child(name)
                try:
                    v = getattr(self._system, name)
                    if self._system._vars[name][2] is None:
                        self.updateParamState(self.child(name), 'autoSet')
                        param.setValue(v)
                    else:
                        self.updateParamState(self.child(name), 'fixed')
                except RuntimeError:
                    self.updateParamState(param, 'autoUnset')
        finally:
            self.sigTreeStateChanged.connect(self.updateSystem)

    def updateParamState(self, param, state):
        if False:
            for i in range(10):
                print('nop')
        if state == 'autoSet':
            bg = fn.mkBrush((200, 255, 200, 255))
            bold = False
            readonly = True
        elif state == 'autoUnset':
            bg = fn.mkBrush(None)
            bold = False
            readonly = False
        elif state == 'fixed':
            bg = fn.mkBrush('y')
            bold = True
            readonly = False
        else:
            raise ValueError("'state' must be one of 'autoSet', 'autoUnset', or 'fixed'")
        param.setReadonly(readonly)