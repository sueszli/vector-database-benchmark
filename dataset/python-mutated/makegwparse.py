"""Utilities for makegw - Parse a header file to build an interface

 This module contains the core code for parsing a header file describing a
 COM interface, and building it into an "Interface" structure.

 Each Interface has methods, and each method has arguments.

 Each argument knows how to use Py_BuildValue or Py_ParseTuple to
 exchange itself with Python.
 
 See the @win32com.makegw@ module for information in building a COM
 interface
"""
import re
import traceback

class error_not_found(Exception):

    def __init__(self, msg='The requested item could not be found'):
        if False:
            i = 10
            return i + 15
        super().__init__(msg)

class error_not_supported(Exception):

    def __init__(self, msg='The required functionality is not supported'):
        if False:
            return 10
        super().__init__(msg)
VERBOSE = 0
DEBUG = 0

class ArgFormatter:
    """An instance for a specific type of argument.	 Knows how to convert itself"""

    def __init__(self, arg, builtinIndirection, declaredIndirection=0):
        if False:
            print('Hello World!')
        self.arg = arg
        self.builtinIndirection = builtinIndirection
        self.declaredIndirection = declaredIndirection
        self.gatewayMode = 0

    def _IndirectPrefix(self, indirectionFrom, indirectionTo):
        if False:
            for i in range(10):
                print('nop')
        'Given the indirection level I was declared at (0=Normal, 1=*, 2=**)\n        return a string prefix so I can pass to a function with the\n        required indirection (where the default is the indirection of the method\'s param.\n\n        eg, assuming my arg has indirection level of 2, if this function was passed 1\n        it would return "&", so that a variable declared with indirection of 1\n        can be prefixed with this to turn it into the indirection level required of 2\n        '
        dif = indirectionFrom - indirectionTo
        if dif == 0:
            return ''
        elif dif == -1:
            return '&'
        elif dif == 1:
            return '*'
        else:
            return '?? (%d)' % (dif,)
            raise error_not_supported("Can't indirect this far - please fix me :-)")

    def GetIndirectedArgName(self, indirectFrom, indirectionTo):
        if False:
            return 10
        if indirectFrom is None:
            indirectFrom = self._GetDeclaredIndirection() + self.builtinIndirection
        return self._IndirectPrefix(indirectFrom, indirectionTo) + self.arg.name

    def GetBuildValueArg(self):
        if False:
            i = 10
            return i + 15
        'Get the argument to be passes to Py_BuildValue'
        return self.arg.name

    def GetParseTupleArg(self):
        if False:
            print('Hello World!')
        'Get the argument to be passed to PyArg_ParseTuple'
        if self.gatewayMode:
            return self.GetIndirectedArgName(None, 1)
        return self.GetIndirectedArgName(self.builtinIndirection, 1)

    def GetInterfaceCppObjectInfo(self):
        if False:
            i = 10
            return i + 15
        'Provide information about the C++ object used.\n\n        Simple variables (such as integers) can declare their type (eg an integer)\n        and use it as the target of both PyArg_ParseTuple and the COM function itself.\n\n        More complex types require a PyObject * declared as the target of PyArg_ParseTuple,\n        then some conversion routine to the C++ object which is actually passed to COM.\n\n        This method provides the name, and optionally the type of that C++ variable.\n        If the type if provided, the caller will likely generate a variable declaration.\n        The name must always be returned.\n\n        Result is a tuple of (variableName, [DeclareType|None|""])\n        '
        return (self.GetIndirectedArgName(self.builtinIndirection, self.arg.indirectionLevel + self.builtinIndirection), f'{self.GetUnconstType()} {self.arg.name}')

    def GetInterfaceArgCleanup(self):
        if False:
            while True:
                i = 10
        'Return cleanup code for C++ args passed to the interface method.'
        if DEBUG:
            return '/* GetInterfaceArgCleanup output goes here: %s */\n' % self.arg.name
        else:
            return ''

    def GetInterfaceArgCleanupGIL(self):
        if False:
            for i in range(10):
                print('nop')
        'Return cleanup code for C++ args passed to the interface\n        method that must be executed with the GIL held'
        if DEBUG:
            return '/* GetInterfaceArgCleanup (GIL held) output goes here: %s */\n' % self.arg.name
        else:
            return ''

    def GetUnconstType(self):
        if False:
            for i in range(10):
                print('nop')
        return self.arg.unc_type

    def SetGatewayMode(self):
        if False:
            i = 10
            return i + 15
        self.gatewayMode = 1

    def _GetDeclaredIndirection(self):
        if False:
            i = 10
            return i + 15
        return self.arg.indirectionLevel
        print('declared:', self.arg.name, self.gatewayMode)
        if self.gatewayMode:
            return self.arg.indirectionLevel
        else:
            return self.declaredIndirection

    def DeclareParseArgTupleInputConverter(self):
        if False:
            return 10
        'Declare the variable used as the PyArg_ParseTuple param for a gateway'
        if DEBUG:
            return '/* Declare ParseArgTupleInputConverter goes here: %s */\n' % self.arg.name
        else:
            return ''

    def GetParsePostCode(self):
        if False:
            for i in range(10):
                print('nop')
        'Get a string of C++ code to be executed after (ie, to finalise) the PyArg_ParseTuple conversion'
        if DEBUG:
            return '/* GetParsePostCode code goes here: %s */\n' % self.arg.name
        else:
            return ''

    def GetBuildForInterfacePreCode(self):
        if False:
            return 10
        'Get a string of C++ code to be executed before (ie, to initialise) the Py_BuildValue conversion for Interfaces'
        if DEBUG:
            return '/* GetBuildForInterfacePreCode goes here: %s */\n' % self.arg.name
        else:
            return ''

    def GetBuildForGatewayPreCode(self):
        if False:
            i = 10
            return i + 15
        'Get a string of C++ code to be executed before (ie, to initialise) the Py_BuildValue conversion for Gateways'
        s = self.GetBuildForInterfacePreCode()
        if DEBUG:
            if s[:4] == '/* G':
                s = '/* GetBuildForGatewayPreCode goes here: %s */\n' % self.arg.name
        return s

    def GetBuildForInterfacePostCode(self):
        if False:
            for i in range(10):
                print('nop')
        'Get a string of C++ code to be executed after (ie, to finalise) the Py_BuildValue conversion for Interfaces'
        if DEBUG:
            return '/* GetBuildForInterfacePostCode goes here: %s */\n' % self.arg.name
        return ''

    def GetBuildForGatewayPostCode(self):
        if False:
            return 10
        'Get a string of C++ code to be executed after (ie, to finalise) the Py_BuildValue conversion for Gateways'
        s = self.GetBuildForInterfacePostCode()
        if DEBUG:
            if s[:4] == '/* G':
                s = '/* GetBuildForGatewayPostCode goes here: %s */\n' % self.arg.name
        return s

    def GetAutoduckString(self):
        if False:
            for i in range(10):
                print('nop')
        return '// @pyparm {}|{}||Description for {}'.format(self._GetPythonTypeDesc(), self.arg.name, self.arg.name)

    def _GetPythonTypeDesc(self):
        if False:
            print('Hello World!')
        'Returns a string with the description of the type.\t Used for doco purposes'
        return None

    def NeedUSES_CONVERSION(self):
        if False:
            for i in range(10):
                print('nop')
        'Determines if this arg forces a USES_CONVERSION macro'
        return 0

class ArgFormatterFloat(ArgFormatter):

    def GetFormatChar(self):
        if False:
            i = 10
            return i + 15
        return 'f'

    def DeclareParseArgTupleInputConverter(self):
        if False:
            for i in range(10):
                print('nop')
        return '\tdouble dbl%s;\n' % self.arg.name

    def GetParseTupleArg(self):
        if False:
            while True:
                i = 10
        return '&dbl' + self.arg.name

    def _GetPythonTypeDesc(self):
        if False:
            for i in range(10):
                print('nop')
        return 'float'

    def GetBuildValueArg(self):
        if False:
            while True:
                i = 10
        return '&dbl' + self.arg.name

    def GetBuildForInterfacePreCode(self):
        if False:
            while True:
                i = 10
        return '\tdbl' + self.arg.name + ' = ' + self.arg.name + ';\n'

    def GetBuildForGatewayPreCode(self):
        if False:
            return 10
        return '\tdbl%s = ' % self.arg.name + self._IndirectPrefix(self._GetDeclaredIndirection(), 0) + self.arg.name + ';\n'

    def GetParsePostCode(self):
        if False:
            for i in range(10):
                print('nop')
        s = '\t'
        if self.gatewayMode:
            s = s + self._IndirectPrefix(self._GetDeclaredIndirection(), 0)
        s = s + self.arg.name
        s = s + ' = (float)dbl%s;\n' % self.arg.name
        return s

class ArgFormatterShort(ArgFormatter):

    def GetFormatChar(self):
        if False:
            while True:
                i = 10
        return 'i'

    def DeclareParseArgTupleInputConverter(self):
        if False:
            for i in range(10):
                print('nop')
        return '\tINT i%s;\n' % self.arg.name

    def GetParseTupleArg(self):
        if False:
            while True:
                i = 10
        return '&i' + self.arg.name

    def _GetPythonTypeDesc(self):
        if False:
            i = 10
            return i + 15
        return 'int'

    def GetBuildValueArg(self):
        if False:
            i = 10
            return i + 15
        return '&i' + self.arg.name

    def GetBuildForInterfacePreCode(self):
        if False:
            return 10
        return '\ti' + self.arg.name + ' = ' + self.arg.name + ';\n'

    def GetBuildForGatewayPreCode(self):
        if False:
            for i in range(10):
                print('nop')
        return '\ti%s = ' % self.arg.name + self._IndirectPrefix(self._GetDeclaredIndirection(), 0) + self.arg.name + ';\n'

    def GetParsePostCode(self):
        if False:
            print('Hello World!')
        s = '\t'
        if self.gatewayMode:
            s = s + self._IndirectPrefix(self._GetDeclaredIndirection(), 0)
        s = s + self.arg.name
        s = s + ' = i%s;\n' % self.arg.name
        return s

class ArgFormatterLONG_PTR(ArgFormatter):

    def GetFormatChar(self):
        if False:
            while True:
                i = 10
        return 'O'

    def DeclareParseArgTupleInputConverter(self):
        if False:
            while True:
                i = 10
        return '\tPyObject *ob%s;\n' % self.arg.name

    def GetParseTupleArg(self):
        if False:
            for i in range(10):
                print('nop')
        return '&ob' + self.arg.name

    def _GetPythonTypeDesc(self):
        if False:
            i = 10
            return i + 15
        return 'int/long'

    def GetBuildValueArg(self):
        if False:
            for i in range(10):
                print('nop')
        return 'ob' + self.arg.name

    def GetBuildForInterfacePostCode(self):
        if False:
            for i in range(10):
                print('nop')
        return '\tPy_XDECREF(ob%s);\n' % self.arg.name

    def GetParsePostCode(self):
        if False:
            i = 10
            return i + 15
        return '\tif (bPythonIsHappy && !PyWinLong_AsULONG_PTR(ob{}, (ULONG_PTR *){})) bPythonIsHappy = FALSE;\n'.format(self.arg.name, self.GetIndirectedArgName(None, 2))

    def GetBuildForInterfacePreCode(self):
        if False:
            print('Hello World!')
        notdirected = self.GetIndirectedArgName(None, 1)
        return f'\tob{self.arg.name} = PyWinObject_FromULONG_PTR({notdirected});\n'

    def GetBuildForGatewayPostCode(self):
        if False:
            print('Hello World!')
        return '\tPy_XDECREF(ob%s);\n' % self.arg.name

class ArgFormatterPythonCOM(ArgFormatter):
    """An arg formatter for types exposed in the PythonCOM module"""

    def GetFormatChar(self):
        if False:
            while True:
                i = 10
        return 'O'

    def DeclareParseArgTupleInputConverter(self):
        if False:
            return 10
        return '\tPyObject *ob%s;\n' % self.arg.name

    def GetParseTupleArg(self):
        if False:
            print('Hello World!')
        return '&ob' + self.arg.name

    def _GetPythonTypeDesc(self):
        if False:
            return 10
        return '<o Py%s>' % self.arg.type

    def GetBuildValueArg(self):
        if False:
            print('Hello World!')
        return 'ob' + self.arg.name

    def GetBuildForInterfacePostCode(self):
        if False:
            return 10
        return '\tPy_XDECREF(ob%s);\n' % self.arg.name

class ArgFormatterBSTR(ArgFormatterPythonCOM):

    def _GetPythonTypeDesc(self):
        if False:
            return 10
        return '<o unicode>'

    def GetParsePostCode(self):
        if False:
            while True:
                i = 10
        return '\tif (bPythonIsHappy && !PyWinObject_AsBstr(ob{}, {})) bPythonIsHappy = FALSE;\n'.format(self.arg.name, self.GetIndirectedArgName(None, 2))

    def GetBuildForInterfacePreCode(self):
        if False:
            print('Hello World!')
        notdirected = self.GetIndirectedArgName(None, 1)
        return f'\tob{self.arg.name} = MakeBstrToObj({notdirected});\n'

    def GetBuildForInterfacePostCode(self):
        if False:
            return 10
        return f'\tSysFreeString({self.arg.name});\n' + ArgFormatterPythonCOM.GetBuildForInterfacePostCode(self)

    def GetBuildForGatewayPostCode(self):
        if False:
            i = 10
            return i + 15
        return '\tPy_XDECREF(ob%s);\n' % self.arg.name

class ArgFormatterOLECHAR(ArgFormatterPythonCOM):

    def _GetPythonTypeDesc(self):
        if False:
            i = 10
            return i + 15
        return '<o unicode>'

    def GetUnconstType(self):
        if False:
            return 10
        if self.arg.type[:3] == 'LPC':
            return self.arg.type[:2] + self.arg.type[3:]
        else:
            return self.arg.unc_type

    def GetParsePostCode(self):
        if False:
            while True:
                i = 10
        return '\tif (bPythonIsHappy && !PyWinObject_AsBstr(ob{}, {})) bPythonIsHappy = FALSE;\n'.format(self.arg.name, self.GetIndirectedArgName(None, 2))

    def GetInterfaceArgCleanup(self):
        if False:
            print('Hello World!')
        return '\tSysFreeString(%s);\n' % self.GetIndirectedArgName(None, 1)

    def GetBuildForInterfacePreCode(self):
        if False:
            while True:
                i = 10
        notdirected = self.GetIndirectedArgName(self.builtinIndirection, 1)
        return f'\tob{self.arg.name} = MakeOLECHARToObj({notdirected});\n'

    def GetBuildForInterfacePostCode(self):
        if False:
            i = 10
            return i + 15
        return f'\tCoTaskMemFree({self.arg.name});\n' + ArgFormatterPythonCOM.GetBuildForInterfacePostCode(self)

    def GetBuildForGatewayPostCode(self):
        if False:
            print('Hello World!')
        return '\tPy_XDECREF(ob%s);\n' % self.arg.name

class ArgFormatterTCHAR(ArgFormatterPythonCOM):

    def _GetPythonTypeDesc(self):
        if False:
            i = 10
            return i + 15
        return 'string/<o unicode>'

    def GetUnconstType(self):
        if False:
            for i in range(10):
                print('nop')
        if self.arg.type[:3] == 'LPC':
            return self.arg.type[:2] + self.arg.type[3:]
        else:
            return self.arg.unc_type

    def GetParsePostCode(self):
        if False:
            print('Hello World!')
        return '\tif (bPythonIsHappy && !PyWinObject_AsTCHAR(ob{}, {})) bPythonIsHappy = FALSE;\n'.format(self.arg.name, self.GetIndirectedArgName(None, 2))

    def GetInterfaceArgCleanup(self):
        if False:
            while True:
                i = 10
        return '\tPyWinObject_FreeTCHAR(%s);\n' % self.GetIndirectedArgName(None, 1)

    def GetBuildForInterfacePreCode(self):
        if False:
            i = 10
            return i + 15
        notdirected = self.GetIndirectedArgName(self.builtinIndirection, 1)
        return f'\tob{self.arg.name} = PyWinObject_FromTCHAR({notdirected});\n'

    def GetBuildForInterfacePostCode(self):
        if False:
            return 10
        return '// ??? - TCHAR post code\n'

    def GetBuildForGatewayPostCode(self):
        if False:
            i = 10
            return i + 15
        return '\tPy_XDECREF(ob%s);\n' % self.arg.name

class ArgFormatterIID(ArgFormatterPythonCOM):

    def _GetPythonTypeDesc(self):
        if False:
            print('Hello World!')
        return '<o PyIID>'

    def GetParsePostCode(self):
        if False:
            while True:
                i = 10
        return '\tif (!PyWinObject_AsIID(ob{}, &{})) bPythonIsHappy = FALSE;\n'.format(self.arg.name, self.arg.name)

    def GetBuildForInterfacePreCode(self):
        if False:
            i = 10
            return i + 15
        notdirected = self.GetIndirectedArgName(None, 0)
        return f'\tob{self.arg.name} = PyWinObject_FromIID({notdirected});\n'

    def GetInterfaceCppObjectInfo(self):
        if False:
            i = 10
            return i + 15
        return (self.arg.name, 'IID %s' % self.arg.name)

class ArgFormatterTime(ArgFormatterPythonCOM):

    def __init__(self, arg, builtinIndirection, declaredIndirection=0):
        if False:
            while True:
                i = 10
        if arg.indirectionLevel == 0 and arg.unc_type[:2] == 'LP':
            arg.unc_type = arg.unc_type[2:]
            arg.indirectionLevel = arg.indirectionLevel + 1
            builtinIndirection = 0
        ArgFormatterPythonCOM.__init__(self, arg, builtinIndirection, declaredIndirection)

    def _GetPythonTypeDesc(self):
        if False:
            print('Hello World!')
        return '<o PyDateTime>'

    def GetParsePostCode(self):
        if False:
            i = 10
            return i + 15
        return '\tif (!PyTime_Check(ob{})) {{\n\t\tPyErr_SetString(PyExc_TypeError, "The argument must be a PyTime object");\n\t\tbPythonIsHappy = FALSE;\n\t}}\n\tif (!((PyTime *)ob{})->GetTime({})) bPythonIsHappy = FALSE;\n'.format(self.arg.name, self.arg.name, self.GetIndirectedArgName(self.builtinIndirection, 1))

    def GetBuildForInterfacePreCode(self):
        if False:
            i = 10
            return i + 15
        notdirected = self.GetIndirectedArgName(self.builtinIndirection, 0)
        return f'\tob{self.arg.name} = new PyTime({notdirected});\n'

    def GetBuildForInterfacePostCode(self):
        if False:
            i = 10
            return i + 15
        ret = ''
        if self.builtinIndirection + self.arg.indirectionLevel > 1:
            ret = '\tCoTaskMemFree(%s);\n' % self.arg.name
        return ret + ArgFormatterPythonCOM.GetBuildForInterfacePostCode(self)

class ArgFormatterSTATSTG(ArgFormatterPythonCOM):

    def _GetPythonTypeDesc(self):
        if False:
            print('Hello World!')
        return '<o STATSTG>'

    def GetParsePostCode(self):
        if False:
            return 10
        return '\tif (!PyCom_PyObjectAsSTATSTG(ob{}, {}, 0/*flags*/)) bPythonIsHappy = FALSE;\n'.format(self.arg.name, self.GetIndirectedArgName(None, 1))

    def GetBuildForInterfacePreCode(self):
        if False:
            for i in range(10):
                print('nop')
        notdirected = self.GetIndirectedArgName(None, 1)
        return '\tob{} = PyCom_PyObjectFromSTATSTG({});\n\t// STATSTG doco says our responsibility to free\n\tif (({}).pwcsName) CoTaskMemFree(({}).pwcsName);\n'.format(self.arg.name, self.GetIndirectedArgName(None, 1), notdirected, notdirected)

class ArgFormatterGeneric(ArgFormatterPythonCOM):

    def _GetPythonTypeDesc(self):
        if False:
            while True:
                i = 10
        return '<o %s>' % self.arg.type

    def GetParsePostCode(self):
        if False:
            for i in range(10):
                print('nop')
        return '\tif (!PyObject_As{}(ob{}, &{}) bPythonIsHappy = FALSE;\n'.format(self.arg.type, self.arg.name, self.GetIndirectedArgName(None, 1))

    def GetInterfaceArgCleanup(self):
        if False:
            print('Hello World!')
        return f'\tPyObject_Free{self.arg.type}({self.arg.name});\n'

    def GetBuildForInterfacePreCode(self):
        if False:
            print('Hello World!')
        notdirected = self.GetIndirectedArgName(None, 1)
        return '\tob{} = PyObject_From{}({});\n'.format(self.arg.name, self.arg.type, self.GetIndirectedArgName(None, 1))

class ArgFormatterIDLIST(ArgFormatterPythonCOM):

    def _GetPythonTypeDesc(self):
        if False:
            print('Hello World!')
        return '<o PyIDL>'

    def GetParsePostCode(self):
        if False:
            print('Hello World!')
        return '\tif (bPythonIsHappy && !PyObject_AsPIDL(ob{}, &{})) bPythonIsHappy = FALSE;\n'.format(self.arg.name, self.GetIndirectedArgName(None, 1))

    def GetInterfaceArgCleanup(self):
        if False:
            print('Hello World!')
        return f'\tPyObject_FreePIDL({self.arg.name});\n'

    def GetBuildForInterfacePreCode(self):
        if False:
            for i in range(10):
                print('nop')
        notdirected = self.GetIndirectedArgName(None, 1)
        return '\tob{} = PyObject_FromPIDL({});\n'.format(self.arg.name, self.GetIndirectedArgName(None, 1))

class ArgFormatterHANDLE(ArgFormatterPythonCOM):

    def _GetPythonTypeDesc(self):
        if False:
            while True:
                i = 10
        return '<o PyHANDLE>'

    def GetParsePostCode(self):
        if False:
            i = 10
            return i + 15
        return '\tif (!PyWinObject_AsHANDLE(ob{}, &{}, FALSE) bPythonIsHappy = FALSE;\n'.format(self.arg.name, self.GetIndirectedArgName(None, 1))

    def GetBuildForInterfacePreCode(self):
        if False:
            for i in range(10):
                print('nop')
        notdirected = self.GetIndirectedArgName(None, 1)
        return '\tob{} = PyWinObject_FromHANDLE({});\n'.format(self.arg.name, self.GetIndirectedArgName(None, 0))

class ArgFormatterLARGE_INTEGER(ArgFormatterPythonCOM):

    def GetKeyName(self):
        if False:
            i = 10
            return i + 15
        return 'LARGE_INTEGER'

    def _GetPythonTypeDesc(self):
        if False:
            for i in range(10):
                print('nop')
        return '<o %s>' % self.GetKeyName()

    def GetParsePostCode(self):
        if False:
            i = 10
            return i + 15
        return '\tif (!PyWinObject_As{}(ob{}, {})) bPythonIsHappy = FALSE;\n'.format(self.GetKeyName(), self.arg.name, self.GetIndirectedArgName(None, 1))

    def GetBuildForInterfacePreCode(self):
        if False:
            return 10
        notdirected = self.GetIndirectedArgName(None, 0)
        return '\tob{} = PyWinObject_From{}({});\n'.format(self.arg.name, self.GetKeyName(), notdirected)

class ArgFormatterULARGE_INTEGER(ArgFormatterLARGE_INTEGER):

    def GetKeyName(self):
        if False:
            for i in range(10):
                print('nop')
        return 'ULARGE_INTEGER'

class ArgFormatterInterface(ArgFormatterPythonCOM):

    def GetInterfaceCppObjectInfo(self):
        if False:
            while True:
                i = 10
        return (self.GetIndirectedArgName(1, self.arg.indirectionLevel), '{} * {}'.format(self.GetUnconstType(), self.arg.name))

    def GetParsePostCode(self):
        if False:
            print('Hello World!')
        if self.gatewayMode:
            sArg = self.GetIndirectedArgName(None, 2)
        else:
            sArg = self.GetIndirectedArgName(1, 2)
        return '\tif (bPythonIsHappy && !PyCom_InterfaceFromPyInstanceOrObject(ob{}, IID_{}, (void **){}, TRUE /* bNoneOK */))\n\t\t bPythonIsHappy = FALSE;\n'.format(self.arg.name, self.arg.type, sArg)

    def GetBuildForInterfacePreCode(self):
        if False:
            return 10
        return '\tob{} = PyCom_PyObjectFromIUnknown({}, IID_{}, FALSE);\n'.format(self.arg.name, self.arg.name, self.arg.type)

    def GetBuildForGatewayPreCode(self):
        if False:
            for i in range(10):
                print('nop')
        sPrefix = self._IndirectPrefix(self._GetDeclaredIndirection(), 1)
        return '\tob{} = PyCom_PyObjectFromIUnknown({}{}, IID_{}, TRUE);\n'.format(self.arg.name, sPrefix, self.arg.name, self.arg.type)

    def GetInterfaceArgCleanup(self):
        if False:
            for i in range(10):
                print('nop')
        return f'\tif ({self.arg.name}) {self.arg.name}->Release();\n'

class ArgFormatterVARIANT(ArgFormatterPythonCOM):

    def GetParsePostCode(self):
        if False:
            return 10
        return '\tif ( !PyCom_VariantFromPyObject(ob{}, {}) )\n\t\tbPythonIsHappy = FALSE;\n'.format(self.arg.name, self.GetIndirectedArgName(None, 1))

    def GetBuildForGatewayPreCode(self):
        if False:
            for i in range(10):
                print('nop')
        notdirected = self.GetIndirectedArgName(None, 1)
        return f'\tob{self.arg.name} = PyCom_PyObjectFromVariant({notdirected});\n'

    def GetBuildForGatewayPostCode(self):
        if False:
            i = 10
            return i + 15
        return '\tPy_XDECREF(ob%s);\n' % self.arg.name
ConvertSimpleTypes = {'BOOL': ('BOOL', 'int', 'i'), 'UINT': ('UINT', 'int', 'i'), 'BYTE': ('BYTE', 'int', 'i'), 'INT': ('INT', 'int', 'i'), 'DWORD': ('DWORD', 'int', 'l'), 'HRESULT': ('HRESULT', 'int', 'l'), 'ULONG': ('ULONG', 'int', 'l'), 'LONG': ('LONG', 'int', 'l'), 'int': ('int', 'int', 'i'), 'long': ('long', 'int', 'l'), 'DISPID': ('DISPID', 'long', 'l'), 'APPBREAKFLAGS': ('int', 'int', 'i'), 'BREAKRESUMEACTION': ('int', 'int', 'i'), 'ERRORRESUMEACTION': ('int', 'int', 'i'), 'BREAKREASON': ('int', 'int', 'i'), 'BREAKPOINT_STATE': ('int', 'int', 'i'), 'BREAKRESUME_ACTION': ('int', 'int', 'i'), 'SOURCE_TEXT_ATTR': ('int', 'int', 'i'), 'TEXT_DOC_ATTR': ('int', 'int', 'i'), 'QUERYOPTION': ('int', 'int', 'i'), 'PARSEACTION': ('int', 'int', 'i')}

class ArgFormatterSimple(ArgFormatter):
    """An arg formatter for simple integer etc types"""

    def GetFormatChar(self):
        if False:
            print('Hello World!')
        return ConvertSimpleTypes[self.arg.type][2]

    def _GetPythonTypeDesc(self):
        if False:
            return 10
        return ConvertSimpleTypes[self.arg.type][1]
AllConverters = {'const OLECHAR': (ArgFormatterOLECHAR, 0, 1), 'WCHAR': (ArgFormatterOLECHAR, 0, 1), 'OLECHAR': (ArgFormatterOLECHAR, 0, 1), 'LPCOLESTR': (ArgFormatterOLECHAR, 1, 1), 'LPOLESTR': (ArgFormatterOLECHAR, 1, 1), 'LPCWSTR': (ArgFormatterOLECHAR, 1, 1), 'LPWSTR': (ArgFormatterOLECHAR, 1, 1), 'LPCSTR': (ArgFormatterOLECHAR, 1, 1), 'LPTSTR': (ArgFormatterTCHAR, 1, 1), 'LPCTSTR': (ArgFormatterTCHAR, 1, 1), 'HANDLE': (ArgFormatterHANDLE, 0), 'BSTR': (ArgFormatterBSTR, 1, 0), 'const IID': (ArgFormatterIID, 0), 'CLSID': (ArgFormatterIID, 0), 'IID': (ArgFormatterIID, 0), 'GUID': (ArgFormatterIID, 0), 'const GUID': (ArgFormatterIID, 0), 'const IID': (ArgFormatterIID, 0), 'REFCLSID': (ArgFormatterIID, 0), 'REFIID': (ArgFormatterIID, 0), 'REFGUID': (ArgFormatterIID, 0), 'const FILETIME': (ArgFormatterTime, 0), 'const SYSTEMTIME': (ArgFormatterTime, 0), 'const LPSYSTEMTIME': (ArgFormatterTime, 1, 1), 'LPSYSTEMTIME': (ArgFormatterTime, 1, 1), 'FILETIME': (ArgFormatterTime, 0), 'SYSTEMTIME': (ArgFormatterTime, 0), 'STATSTG': (ArgFormatterSTATSTG, 0), 'LARGE_INTEGER': (ArgFormatterLARGE_INTEGER, 0), 'ULARGE_INTEGER': (ArgFormatterULARGE_INTEGER, 0), 'VARIANT': (ArgFormatterVARIANT, 0), 'float': (ArgFormatterFloat, 0), 'single': (ArgFormatterFloat, 0), 'short': (ArgFormatterShort, 0), 'WORD': (ArgFormatterShort, 0), 'VARIANT_BOOL': (ArgFormatterShort, 0), 'HWND': (ArgFormatterLONG_PTR, 1), 'HMENU': (ArgFormatterLONG_PTR, 1), 'HOLEMENU': (ArgFormatterLONG_PTR, 1), 'HICON': (ArgFormatterLONG_PTR, 1), 'HDC': (ArgFormatterLONG_PTR, 1), 'LPARAM': (ArgFormatterLONG_PTR, 1), 'WPARAM': (ArgFormatterLONG_PTR, 1), 'LRESULT': (ArgFormatterLONG_PTR, 1), 'UINT': (ArgFormatterShort, 0), 'SVSIF': (ArgFormatterShort, 0), 'Control': (ArgFormatterInterface, 0, 1), 'DataObject': (ArgFormatterInterface, 0, 1), '_PropertyBag': (ArgFormatterInterface, 0, 1), 'AsyncProp': (ArgFormatterInterface, 0, 1), 'DataSource': (ArgFormatterInterface, 0, 1), 'DataFormat': (ArgFormatterInterface, 0, 1), 'void **': (ArgFormatterInterface, 2, 2), 'ITEMIDLIST': (ArgFormatterIDLIST, 0, 0), 'LPITEMIDLIST': (ArgFormatterIDLIST, 0, 1), 'LPCITEMIDLIST': (ArgFormatterIDLIST, 0, 1), 'const ITEMIDLIST': (ArgFormatterIDLIST, 0, 1)}
for key in ConvertSimpleTypes.keys():
    AllConverters[key] = (ArgFormatterSimple, 0)

def make_arg_converter(arg):
    if False:
        print('Hello World!')
    try:
        clz = AllConverters[arg.type][0]
        bin = AllConverters[arg.type][1]
        decl = 0
        if len(AllConverters[arg.type]) > 2:
            decl = AllConverters[arg.type][2]
        return clz(arg, bin, decl)
    except KeyError:
        if arg.type[0] == 'I':
            return ArgFormatterInterface(arg, 0, 1)
        raise error_not_supported(f"The type '{arg.type}' ({arg.name}) is unknown.")

class Argument:
    """A representation of an argument to a COM method

    This class contains information about a specific argument to a method.
    In addition, methods exist so that an argument knows how to convert itself
    to/from Python arguments.
    """
    regex = re.compile('/\\* \\[([^\\]]*.*?)] \\*/[ \\t](.*[* ]+)(\\w+)(\\[ *])?[\\),]')

    def __init__(self, good_interface_names):
        if False:
            print('Hello World!')
        self.good_interface_names = good_interface_names
        self.inout = self.name = self.type = None
        self.const = 0
        self.arrayDecl = 0

    def BuildFromFile(self, file):
        if False:
            while True:
                i = 10
        'Parse and build my data from a file\n\n        Reads the next line in the file, and matches it as an argument\n        description.  If not a valid argument line, an error_not_found exception\n        is raised.\n        '
        line = file.readline()
        mo = self.regex.search(line)
        if not mo:
            raise error_not_found
        self.name = mo.group(3)
        self.inout = mo.group(1).split('][')
        typ = mo.group(2).strip()
        self.raw_type = typ
        self.indirectionLevel = 0
        if mo.group(4):
            self.arrayDecl = 1
            try:
                pos = typ.rindex('__RPC_FAR')
                self.indirectionLevel = self.indirectionLevel + 1
                typ = typ[:pos].strip()
            except ValueError:
                pass
        typ = typ.replace('__RPC_FAR', '')
        while 1:
            try:
                pos = typ.rindex('*')
                self.indirectionLevel = self.indirectionLevel + 1
                typ = typ[:pos].strip()
            except ValueError:
                break
        self.type = typ
        if self.type[:6] == 'const ':
            self.unc_type = self.type[6:]
        else:
            self.unc_type = self.type
        if VERBOSE:
            print('\t   Arg {} of type {}{} ({})'.format(self.name, self.type, '*' * self.indirectionLevel, self.inout))

    def HasAttribute(self, typ):
        if False:
            i = 10
            return i + 15
        'Determines if the argument has the specific attribute.\n\n        Argument attributes are specified in the header file, such as\n        "[in][out][retval]" etc.  You can pass a specific string (eg "out")\n        to find if this attribute was specified for the argument\n        '
        return typ in self.inout

    def GetRawDeclaration(self):
        if False:
            return 10
        ret = f'{self.raw_type} {self.name}'
        if self.arrayDecl:
            ret = ret + '[]'
        return ret

class Method:
    """A representation of a C++ method on a COM interface

    This class contains information about a specific method, as well as
    a list of all @Argument@s
    """
    regex = re.compile('virtual (/\\*.*?\\*/ )?(.*?) (.*?) (.*?)\\(\\w?')

    def __init__(self, good_interface_names):
        if False:
            for i in range(10):
                print('nop')
        self.good_interface_names = good_interface_names
        self.name = self.result = self.callconv = None
        self.args = []

    def BuildFromFile(self, file):
        if False:
            i = 10
            return i + 15
        'Parse and build my data from a file\n\n        Reads the next line in the file, and matches it as a method\n        description.  If not a valid method line, an error_not_found exception\n        is raised.\n        '
        line = file.readline()
        mo = self.regex.search(line)
        if not mo:
            raise error_not_found
        self.name = mo.group(4)
        self.result = mo.group(2)
        if self.result != 'HRESULT':
            if self.result == 'DWORD':
                print('Warning: Old style interface detected - compilation errors likely!')
            else:
                print('Method %s - Only HRESULT return types are supported.' % self.name)
            print(f'\t Method {self.result} {self.name}(')
        while 1:
            arg = Argument(self.good_interface_names)
            try:
                arg.BuildFromFile(file)
                self.args.append(arg)
            except error_not_found:
                break

class Interface:
    """A representation of a C++ COM Interface

    This class contains information about a specific interface, as well as
    a list of all @Method@s
    """
    regex = re.compile('(interface|) ([^ ]*) : public (.*)$')

    def __init__(self, mo):
        if False:
            print('Hello World!')
        self.methods = []
        self.name = mo.group(2)
        self.base = mo.group(3)
        if VERBOSE:
            print(f'Interface {self.name} : public {self.base}')

    def BuildMethods(self, file):
        if False:
            for i in range(10):
                print('nop')
        'Build all sub-methods for this interface'
        file.readline()
        file.readline()
        while 1:
            try:
                method = Method([self.name])
                method.BuildFromFile(file)
                self.methods.append(method)
            except error_not_found:
                break

def find_interface(interfaceName, file):
    if False:
        return 10
    'Find and return an interface in a file\n\n    Given an interface name and file, search for the specified interface.\n\n    Upon return, the interface itself has been built,\n    but not the methods.\n    '
    interface = None
    line = file.readline()
    while line:
        mo = Interface.regex.search(line)
        if mo:
            name = mo.group(2)
            print(name)
            AllConverters[name] = (ArgFormatterInterface, 0, 1)
            if name == interfaceName:
                interface = Interface(mo)
                interface.BuildMethods(file)
        line = file.readline()
    if interface:
        return interface
    raise error_not_found

def parse_interface_info(interfaceName, file):
    if False:
        print('Hello World!')
    'Find, parse and return an interface in a file\n\n    Given an interface name and file, search for the specified interface.\n\n    Upon return, the interface itself is fully built,\n    '
    try:
        return find_interface(interfaceName, file)
    except re.error:
        traceback.print_exc()
        print('The interface could not be built, as the regular expression failed!')

def test():
    if False:
        for i in range(10):
            print('nop')
    f = open('d:\\msdev\\include\\objidl.h')
    try:
        parse_interface_info('IPersistStream', f)
    finally:
        f.close()

def test_regex(r, text):
    if False:
        i = 10
        return i + 15
    res = r.search(text, 0)
    if res == -1:
        print('** Not found')
    else:
        print('%d\n%s\n%s\n%s\n%s' % (res, r.group(1), r.group(2), r.group(3), r.group(4)))