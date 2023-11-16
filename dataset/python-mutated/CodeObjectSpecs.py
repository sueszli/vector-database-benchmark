""" Code object specifications.

For code objects that will be attached to module, function, and generator
objects, as well as tracebacks. They might be shared.

"""
from nuitka.utils.InstanceCounters import counted_del, counted_init, isCountingInstances

class CodeObjectSpec(object):
    __slots__ = ('co_name', 'co_qualname', 'co_kind', 'co_varnames', 'co_argcount', 'co_freevars', 'co_posonlyargcount', 'co_kwonlyargcount', 'co_has_starlist', 'co_has_stardict', 'filename', 'line_number', 'future_spec', 'new_locals', 'is_optimized')

    @counted_init
    def __init__(self, co_name, co_qualname, co_kind, co_varnames, co_freevars, co_argcount, co_posonlyargcount, co_kwonlyargcount, co_has_starlist, co_has_stardict, co_filename, co_lineno, future_spec, co_new_locals=None, co_is_optimized=None):
        if False:
            print('Hello World!')
        self.co_name = co_name
        self.co_qualname = co_qualname
        self.co_kind = co_kind
        self.future_spec = future_spec
        assert future_spec
        if type(co_varnames) is str:
            if co_varnames == '':
                co_varnames = ()
            else:
                co_varnames = co_varnames.split(',')
        if type(co_freevars) is str:
            if co_freevars == '':
                co_freevars = ()
            else:
                co_freevars = co_freevars.split(',')
        if type(co_has_starlist) is not bool:
            co_has_starlist = co_has_starlist != 'False'
        if type(co_has_stardict) is not bool:
            co_has_stardict = co_has_stardict != 'False'
        self.co_varnames = tuple(co_varnames)
        self.co_freevars = tuple(co_freevars)
        self.co_argcount = int(co_argcount)
        self.co_posonlyargcount = int(co_posonlyargcount)
        self.co_kwonlyargcount = int(co_kwonlyargcount)
        self.co_has_starlist = co_has_starlist
        self.co_has_stardict = co_has_stardict
        self.filename = co_filename
        self.line_number = int(co_lineno)
        if type(co_has_starlist) is not bool:
            co_new_locals = co_new_locals != 'False'
        if type(co_has_starlist) is not bool:
            co_is_optimized = co_is_optimized != 'False'
        self.new_locals = co_new_locals
        self.is_optimized = co_is_optimized
    if isCountingInstances():
        __del__ = counted_del()

    def __repr__(self):
        if False:
            return 10
        return "<CodeObjectSpec %(co_kind)s '%(co_name)s' with %(co_varnames)r>" % self.getDetails()

    def getDetails(self):
        if False:
            i = 10
            return i + 15
        return {'co_name': self.co_name, 'co_kind': self.co_kind, 'co_varnames': ','.join(self.co_varnames), 'co_freevars': ','.join(self.co_freevars), 'co_argcount': self.co_argcount, 'co_posonlyargcount': self.co_posonlyargcount, 'co_kwonlyargcount': self.co_kwonlyargcount, 'co_has_starlist': self.co_has_starlist, 'co_has_stardict': self.co_has_stardict, 'co_filename': self.filename, 'co_lineno': self.line_number, 'co_new_locals': self.new_locals, 'co_is_optimized': self.is_optimized, 'code_flags': ','.join(self.future_spec.asFlags())}

    def getCodeObjectKind(self):
        if False:
            i = 10
            return i + 15
        return self.co_kind

    def updateLocalNames(self, local_names, freevar_names):
        if False:
            for i in range(10):
                print('nop')
        'Move detected local variables after closure has been decided.'
        self.co_varnames += tuple((local_name for local_name in local_names if local_name not in self.co_varnames))
        self.co_freevars = tuple(freevar_names)

    def removeFreeVarname(self, freevar_name):
        if False:
            i = 10
            return i + 15
        self.co_freevars = tuple((var_name for var_name in self.co_freevars if var_name != freevar_name))

    def setFlagIsOptimizedValue(self, value):
        if False:
            print('Hello World!')
        self.is_optimized = value

    def getFlagIsOptimizedValue(self):
        if False:
            i = 10
            return i + 15
        return self.is_optimized

    def setFlagNewLocalsValue(self, value):
        if False:
            while True:
                i = 10
        self.new_locals = value

    def getFlagNewLocalsValue(self):
        if False:
            while True:
                i = 10
        return self.new_locals

    def getFutureSpec(self):
        if False:
            while True:
                i = 10
        return self.future_spec

    def getVarNames(self):
        if False:
            print('Hello World!')
        return self.co_varnames

    def getFreeVarNames(self):
        if False:
            while True:
                i = 10
        return self.co_freevars

    def getArgumentCount(self):
        if False:
            print('Hello World!')
        return self.co_argcount

    def getPosOnlyParameterCount(self):
        if False:
            for i in range(10):
                print('nop')
        return self.co_posonlyargcount

    def getKwOnlyParameterCount(self):
        if False:
            return 10
        return self.co_kwonlyargcount

    def getCodeObjectName(self):
        if False:
            i = 10
            return i + 15
        return self.co_name

    def getCodeObjectQualname(self):
        if False:
            return 10
        return self.co_name

    def hasStarListArg(self):
        if False:
            i = 10
            return i + 15
        return self.co_has_starlist

    def hasStarDictArg(self):
        if False:
            return 10
        return self.co_has_stardict

    def getFilename(self):
        if False:
            while True:
                i = 10
        return self.filename

    def getLineNumber(self):
        if False:
            for i in range(10):
                print('nop')
        return self.line_number