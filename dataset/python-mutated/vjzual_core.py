from abc import abstractproperty, ABCMeta, abstractmethod
__author__ = 'tekt'
import json

def argToOp(arg):
    if False:
        print('Hello World!')
    if not arg:
        return None
    if isinstance(arg, str):
        o = op(arg)
        if not o:
            raise Exception('operator not found: ' + arg)
        return o
    return arg

def argToPath(arg):
    if False:
        while True:
            i = 10
    if not arg:
        return ''
    if isinstance(arg, str):
        return arg
    if hasattr(arg, 'path'):
        return arg.path
    return arg

def toggleCooking(path, delayFrames=1):
    if False:
        print('Hello World!')
    path = argToPath(path)
    op(path).allowCooking = False
    run('op("' + path + '").allowCooking = True', delayFrames=delayFrames)

def toggleExport(path, delayFrames=1):
    if False:
        while True:
            i = 10
    path = argToPath(path)
    op(path).export = False
    run('op("' + path + '").export = True', delayFrames=delayFrames)

def toggleBypass(path, delayFrames=1):
    if False:
        print('Hello World!')
    path = argToPath(path)
    op(path).bypass = True
    run('op("' + path + '").bypass = False', delayFrames=delayFrames)

def nameToAbbr(name):
    if False:
        i = 10
        return i + 15
    if ':' in name:
        (dev, ctl) = name.split(':')
        return dev[0] + ':' + ctl
    return name

def _midiAbbrToName(comp, abbr):
    if False:
        while True:
            i = 10
    ctrlmap = comp.op(comp.var('midictrlabbrmap'))
    n = ctrlmap[abbr, 'name']
    return n.val if n else None

def updateTableRow(tbl, rowKey, vals, addMissing=False, ignoreMissingCols=False):
    if False:
        for i in range(10):
            print('nop')
    tbl = argToOp(tbl)
    if not tbl:
        return
    if not tbl[rowKey, 0]:
        if not addMissing:
            raise Exception('row ' + rowKey + ' not found in table ' + tbl)
        else:
            tbl.appendRow([rowKey])
    for colKey in vals:
        v = vals[colKey]
        if ignoreMissingCols and tbl[rowKey, colKey] is None:
            continue
        tbl[rowKey, colKey] = v if v is not None else ''

def overrideRows(tbl, overrides):
    if False:
        return 10
    tbl = argToOp(tbl)
    if not tbl:
        return
    for key in overrides:
        tbl[key, 1] = overrides[key]

def prepFilterList(filterstr):
    if False:
        print('Hello World!')
    if isinstance(filterstr, Cell):
        filterstr = filterstr.val
    if not filterstr:
        return None
    if '|' in filterstr:
        return filterstr.split('|')
    return [filterstr]

def rowsToDictList(tbl):
    if False:
        return 10
    tbl = argToOp(tbl)
    if not tbl:
        return []
    allObjs = []
    cols = [c.val for c in tbl.row(0)]
    for i in range(1, tbl.numRows):
        obj = {c: tbl[i, c].val for c in cols if tbl[i, c] != ''}
        allObjs.append(obj)
    return allObjs

def fillTableFromDicts(tbl, objs, cols=None, keyCol='name'):
    if False:
        i = 10
        return i + 15
    tbl = argToOp(tbl)
    tbl.clear()
    tbl.appendRow(cols)
    if not len(objs):
        return
    for obj in objs:
        if not len(obj):
            continue
        updateTableRow(tbl, obj[keyCol], withoutDictEmptyStrings(obj), addMissing=True, ignoreMissingCols=False)

def buildModuleDefDicts(moduletbl, paramtbl):
    if False:
        i = 10
        return i + 15
    (moduletbl, paramtbl) = (argToOp(moduletbl), argToOp(paramtbl))
    mDicts = rowsToDictList(moduletbl)
    if not mDicts:
        return None
    mDicts = [withoutDictEmptyStrings(m) for m in mDicts]
    pDicts = rowsToDictList(paramtbl)
    pDicts = [withoutDictEmptyStrings(p) for p in pDicts]
    for p in pDicts:
        modname = p['module']
        if modname:
            for m in mDicts:
                if m['name'] == modname:
                    if not 'paramdefs' in m:
                        m['paramdefs'] = []
                    m['paramdefs'].append(p)
                    break
    return mDicts

def extractModuleTblFromDicts(moduleDicts, moduletbl):
    if False:
        return 10
    moduletbl = argToOp(moduletbl)
    moduletbl.setSize(1, moduletbl.numCols)
    for mDict in moduleDicts:
        updateTableRow(moduletbl, mDict['name'], mDict, addMissing=True, ignoreMissingCols=True)

def extractParamTableFromDicts(moduleDicts, paramtbl):
    if False:
        while True:
            i = 10
    paramtbl = argToOp(paramtbl)
    paramtbl.setSize(1, paramtbl.numCols)
    for mDict in moduleDicts:
        if not 'paramdefs' in mDict:
            continue
        for pDict in mDict['paramdefs']:
            updateTableRow(paramtbl, pDict['name'], pDict, addMissing=True, ignoreMissingCols=True)

def withoutDictEmptyStrings(d):
    if False:
        print('Hello World!')
    return {k: d[k] for k in d if d[k] != ''}

def DEBUGLOG(s):
    if False:
        print('Hello World!')
    pass

def getOrAddParamPage(comp, name):
    if False:
        while True:
            i = 10
    comp = argToOp(comp)
    for page in comp.customPages:
        if page.name == name:
            return page
    return comp.appendCustomPage(name)

def _logDeprecatedCall(self, methodName):
    if False:
        while True:
            i = 10
    if hasattr(self, '_comp') and self._comp:
        selfName = self._comp.path
    else:
        selfName = str(self)
    print('deprecated extension method "' + methodName + '" called on [' + type(self).__name__ + '] ' + selfName)

def deprecatedMethod(origFn):
    if False:
        return 10

    def newFn(*args, **kwargs):
        if False:
            return 10
        _logDeprecatedCall(args[0], origFn.__name__)
        return origFn(*args, **kwargs)
    return newFn

def _safeTestForAttr(comp, name):
    if False:
        i = 10
        return i + 15
    try:
        return hasattr(comp, name)
    except:
        return False

def toggleTag(comp, name, enable):
    if False:
        return 10
    comp = argToOp(comp)
    if enable:
        comp.tags.add(name)
    elif name in comp.tags:
        comp.tags.remove(name)

def notImplemented(*unused_args):
    if False:
        return 10
    raise NotImplementedError()

def make_getterNotImplemented():
    if False:
        i = 10
        return i + 15
    return lambda self: notImplemented()

def make_setterNotImplemented():
    if False:
        for i in range(10):
            print('nop')
    return lambda self, value: notImplemented()

def _override(func):
    if False:
        i = 10
        return i + 15
    return func

class VjzParamBase:
    __metaclass__ = ABCMeta

    def GetParamDefProperty(self, name):
        if False:
            for i in range(10):
                print('nop')
        ptbl = VJZ.ParamTable
        if not ptbl:
            return None
        cell = ptbl[self.ParamName, name]
        if cell is not None:
            return cell.val
    ParamValue_getter = make_getterNotImplemented()
    ParamValue_setter = make_setterNotImplemented()
    ParamValue = abstractproperty(ParamValue_getter, ParamValue_setter)

    @abstractproperty
    def ParamName(self):
        if False:
            while True:
                i = 10
        return notImplemented(self)
    ParamMidiName_getter = make_getterNotImplemented()
    ParamMidiName_setter = make_setterNotImplemented()
    ParamMidiName = abstractproperty(ParamMidiName_getter, ParamMidiName_setter)

    def UpdateParamTableEntry(self, vals):
        if False:
            print('Hello World!')
        updateTableRow(VJZ.GetSysOp('editableparamtbl'), self.ParamName, vals)

    def SaveParamMidiMapping(self):
        if False:
            for i in range(10):
                print('nop')
        notImplemented(self)

    def LoadParamMidiMapping(self):
        if False:
            for i in range(10):
                print('nop')
        (dev, ctl) = (self.GetParamDefProperty('mididev'), self.GetParamDefProperty('midictl'))
        if not dev or not ctl:
            self.ParamMidiName = None
        else:
            self.ParamMidiName = dev[0] + ':' + ctl

    def ResetParamToDefault(self):
        if False:
            for i in range(10):
                print('nop')
        val = self.GetParamDefProperty('default')
        if val is None:
            raise Exception('Parameter {0} does not have a default value and cannot be reset'.format(self.ParamName))
        self.ParamValue = val

    def SaveParamValue(self, tbl):
        if False:
            while True:
                i = 10
        val = round(self.ParamValue, 4)
        updateTableRow(tbl, self.ParamName, {'value': val}, addMissing=True)

    def LoadParamValue(self, tbl):
        if False:
            for i in range(10):
                print('nop')
        val = tbl[self.ParamName, 1]
        if val is not None:
            self.ParamValue = float(val)

class OpParamWrapper(VjzParamBase):

    def __init__(self, name, op, parName):
        if False:
            print('Hello World!')
        self._name = name
        self._op = op
        self._parName = parName

    @property
    @_override
    def ParamName(self):
        if False:
            i = 10
            return i + 15
        return self._name

    @_override
    def ParamValue_getter(self):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self._op.par, self._parName).eval()

    @_override
    def ParamValue_setter(self, val):
        if False:
            for i in range(10):
                print('nop')
        setattr(self._op.par, self._parName, val)
    ParamValue = property(ParamValue_getter, ParamValue_setter)

    @_override
    def ParamMidiName_getter(self):
        if False:
            return 10
        return None

    @_override
    def ParamMidiName_setter(self, val):
        if False:
            while True:
                i = 10
        notImplemented(self, val)
    ParamMidiName = property(ParamMidiName_getter, ParamMidiName_setter)

class VjzParam(VjzParamBase):

    def __init__(self, comp):
        if False:
            i = 10
            return i + 15
        self._comp = comp
        page = comp.appendCustomPage('Vjzparam')
        page.appendStr('Paramname', label='Parameter Name')
        page.appendToggle('Scale', label='Scale Parameter Value')
        minmax = page.appendFloat('Scalerange', label='Scale Min/Max', size=2)
        minmax[0].default = 0
        minmax[1].default = 1

    @staticmethod
    def get(comp):
        if False:
            print('Hello World!')
        if _safeTestForAttr(comp, 'ParamDef'):
            return comp
        if _safeTestForAttr(comp, 'paramDef'):
            print('could only find VjzParam extension for comp ' + comp.path + ' using old attribute name')
            return comp
        if comp.ext and hasattr(comp.ext, 'VjzParam'):
            print('could only find VjzParam extension for comp ' + comp.path + ' using ext.VjzParam')
            return comp.ext.VjzParam
        if comp.extensions:
            for e in comp.extensions:
                if isinstance(e, VjzParam):
                    print('could only find VjzParam extension for comp ' + comp.path + ' by looping through extension list')
                    return e
        print('unable to find VjzParam extension for comp: ' + comp.path)
        return None

    def PVar(self, name):
        if False:
            return 10
        return self._comp.var(name)

    @property
    @_override
    def ParamName(self):
        if False:
            i = 10
            return i + 15
        return self.PVar('pname')

    @property
    def ParamDef(self):
        if False:
            return 10
        d = self._comp.op(self.PVar('pdef'))
        return d if d.numRows == 2 else None

    @_override
    def ParamValue_getter(self):
        if False:
            while True:
                i = 10
        return self._comp.op('value')[0][0]

    @_override
    def ParamValue_setter(self, val):
        if False:
            i = 10
            return i + 15
        self._comp.op('slider').panel.u = val
    ParamValue = property(ParamValue_getter, ParamValue_setter)

    @property
    def ParamMidiMapping(self):
        if False:
            while True:
                i = 10
        mapping = self._comp.op('mapping')
        return mapping if mapping.numRows == 2 else None

    @_override
    def ParamMidiName_getter(self):
        if False:
            return 10
        m = self.ParamMidiMapping
        return m[1, 'name'].val if m else None

    @_override
    def ParamMidiName_setter(self, name):
        if False:
            for i in range(10):
                print('nop')
        if not name or name == '-':
            abbr = '-'
            i = 0
        else:
            ctrlmap = self._comp.op(self.PVar('midictrlabbrmap'))
            abbr = nameToAbbr(name)
            n = ctrlmap[abbr, 'abbr']
            if n:
                i = n.row + 1
            else:
                abbr = '-'
                i = 0
        self._comp.op('midictllist/set').run(i, abbr)
    ParamMidiName = property(ParamMidiName_getter, ParamMidiName_setter)

    @_override
    def SaveParamMidiMapping(self):
        if False:
            print('Hello World!')
        mapping = self.ParamMidiMapping
        if not mapping:
            (dev, ctl) = ('', '')
        else:
            (dev, ctl) = (mapping[1, 'mididev'].val, mapping[1, 'midictl'].val)
        self.UpdateParamTableEntry({'mididev': dev, 'midictl': ctl})

class VjzModule:

    def __init__(self, comp):
        if False:
            i = 10
            return i + 15
        self._comp = comp
        page = comp.appendCustomPage('Vjzmodule')
        page.appendStr('Modname', label='Module name')
        callbacks = self._comp.op('callbacks')
        if callbacks and callbacks.isDAT:
            self._callbacks = mod(callbacks)
        else:
            self._callbacks = None
        self._specialParams = []
        toggleTag(self._comp, 'vjzmodule', self.MVar('modfake') != '1')

    @staticmethod
    def get(comp):
        if False:
            i = 10
            return i + 15
        if _safeTestForAttr(comp, 'ModName'):
            return comp
        if _safeTestForAttr(comp, 'modName'):
            print('could only find VjzModule extension for comp ' + comp.path + ' using old attribute name')
            return comp
        if comp.ext and hasattr(comp.ext, 'VjzModule'):
            print('could only find VjzModule extension for comp ' + comp.path + ' using ext.VjzModule')
            return comp.ext.VjzModule
        if comp.extensions:
            for e in comp.extensions:
                if isinstance(e, VjzModule):
                    print('could only find VjzModule extension for comp ' + comp.path + ' by looping through extension list')
                    return e
        print('unable to find VjzModule extension for comp: ' + comp.path)
        return None

    def _InvokeCallback(self, name, *args):
        if False:
            i = 10
            return i + 15
        if not self._callbacks or not hasattr(self._callbacks, name):
            return None
        return getattr(self._callbacks, name)(self, *args)

    def MVar(self, name):
        if False:
            print('Hello World!')
        return self._comp.var(name)

    def GetModOp(self, name):
        if False:
            while True:
                i = 10
        return self._comp.op(self.MVar(name))

    @property
    def ModName(self):
        if False:
            while True:
                i = 10
        return self.MVar('modname')

    @property
    def ModPath(self):
        if False:
            print('Hello World!')
        return self._comp.path

    @property
    def ModState(self):
        if False:
            i = 10
            return i + 15
        return self.GetModOp('modstate')

    @property
    def ModParamTable(self):
        if False:
            i = 10
            return i + 15
        return self.GetModOp('modparamtbl')

    @property
    def ModParamNames(self):
        if False:
            while True:
                i = 10
        return [c.val for c in self.ModParamTable.col('name')[1:]]

    @property
    def ModParamLocalNames(self):
        if False:
            return 10
        return [c.val for c in self.ModParamTable.col('localname')[1:]]

    def GetModParamCompObjects(self):
        if False:
            while True:
                i = 10
        pnames = self.ModParamLocalNames
        params = []
        for p in pnames:
            pop = self.ModParam(p)
            if pop:
                params.append(pop)
        return params

    def GetSpecialParamObjects(self):
        if False:
            while True:
                i = 10
        params = self._InvokeCallback('GetSpecialParamObjects')
        specials = params if params else []
        return specials + self._specialParams

    @property
    def ModParamObjects(self):
        if False:
            for i in range(10):
                print('nop')
        compParams = self.GetModParamCompObjects()
        specialParams = self.GetSpecialParamObjects()
        return compParams + specialParams

    def ModParam(self, name):
        if False:
            while True:
                i = 10
        pop = self._InvokeCallback('GetModParam', name)
        if pop:
            return pop
        pop = self._comp.op(name + '_param')
        if pop:
            return VjzParam.get(pop)

    def SaveParamValues(self, tbl):
        if False:
            print('Hello World!')
        tbl = argToOp(tbl)
        print('saving module ' + self.ModName + ' to ' + tbl.path)
        pnames = self.ModParamLocalNames
        for pname in pnames:
            self._SaveParamValue(tbl, pname)

    def _SaveParamValue(self, tbl, pname):
        if False:
            return 10
        if self._InvokeCallback('SaveParamValue', tbl, pname) is True:
            return
        pop = self.ModParam(pname)
        if pop:
            pop.SaveParamValue(tbl)
            return
        pvals = self._comp.op(self.MVar('modparamsout'))
        if pvals:
            c = pvals.chan(pname)
            if c is not None:
                updateTableRow(tbl, pname, {'value': c[0]})
                return
        print('cannot save parameter ' + pname)

    def LoadParamValues(self, tbl):
        if False:
            return 10
        tbl = argToOp(tbl)
        pnames = self.ModParamLocalNames
        for pname in pnames:
            self._LoadParamValue(tbl, pname)

    def _LoadParamValue(self, tbl, pname):
        if False:
            return 10
        if self._InvokeCallback('LoadParamValue', tbl, pname) is True:
            return
        pop = self.ModParam(pname)
        if pop:
            pop.LoadParamValue(tbl)
        else:
            print('cannot load parameter ' + self.ModPath + ' : ' + pname)

    def ResetParamsToDefaults(self):
        if False:
            print('Hello World!')
        for p in self.ModParamObjects:
            p.ResetParamToDefault()

class VjzSystem:

    def __init__(self, root):
        if False:
            return 10
        self._root = root

    def SVar(self, name):
        if False:
            print('Hello World!')
        return self._root.var(name)

    @property
    def ModuleTable(self):
        if False:
            return 10
        return self.GetSysOp('moduletbl')

    @property
    def ParamTable(self):
        if False:
            i = 10
            return i + 15
        return self.GetSysOp('paramtbl')

    @property
    def ParamStateTable(self):
        if False:
            while True:
                i = 10
        return self.GetSysOp('paramstatetbl')

    def GetSysOp(self, name):
        if False:
            for i in range(10):
                print('nop')
        return self._root.op(self.SVar(name))

    def GetModules(self, fakes=False):
        if False:
            i = 10
            return i + 15
        modtbl = self.ModuleTable
        mods = []
        for mname in modtbl.col('name')[1:]:
            if not fakes and modtbl[mname, 'fake'] == '1':
                continue
            mop = self._root.op(modtbl[mname, 'path'])
            if mop:
                m = VjzModule.get(mop)
                if m:
                    mods.append(m)
        return mods

    def GetModule(self, name):
        if False:
            return 10
        m = self.ModuleTable[name, 'path']
        m = op(m) if m else None
        if m is None:
            raise Exception('module not found: "' + name + '"')
        return VjzModule.get(m)

    def SaveParamValues(self):
        if False:
            print('Hello World!')
        tbl = self.ParamStateTable
        for m in self.GetModules():
            m.SaveParamValues(tbl)
        tbl.save(tbl.par.file.val)

    def LoadParamValues(self):
        if False:
            print('Hello World!')
        tbl = self.ParamStateTable
        for m in self.GetModules():
            print('loading param values in: ', m.ModPath)
            m.LoadParamValues(tbl)

    def SaveParamTableJson(self):
        if False:
            print('Hello World!')
        tbl = self.GetSysOp('editableparamtbl')
        objs = rowsToDictList(tbl)
        j = json.dumps(objs, indent=2)
        jdat = self.GetSysOp('editableparamtbljson')
        jdat.text = j
        jdat.par.write.pulse(1)
VJZ = VjzSystem(op('/_'))