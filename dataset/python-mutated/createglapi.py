"""
Scipt to generate code for our gl API, including all its backends.

The files involved in the code generation process are:

  * gl2.h - the C header file for the GL ES 2.0 API
  * vispy_ext.h - C header file with a subset of GL3+ constants defined
  * webgl.idl - the interface definition language for WebGL
  * annotations.py - manual annotations for non-trivial functions
  * headerparser.py - parses .h and .idl files 
  * createglapi.py - uses all of the above to generate the actual code

Rationale
---------

The GL ES 2.0 API is relatively small. Therefore a fully automated build
process like PyOpenGL now has is not really necessary. I tried to
automate what we can and simply do the rest with manual annotations.
Some groups of functions (like glUniform and friends) are handled in
*this* file.

Even though the API is quite small, we want to generate several
implementations, such as gl2 (desktop), es2 (angle on Windows), a generic
proxy, a mock backend and possibly more. Therefore automation
is crucial.

Further notes
-------------

This file is pretty big and even though I tried to make the code as clear
as possible, it's not always that easy to read. In effect this code is
not so easy to maintain. I hope it is at least clear enough so it can be
used to maintain the GL API itself.

This function must be run using Python3.
"""
from __future__ import annotations
import os
import sys
import headerparser
from annotations import parse_anotations
from OpenGL import GL
THISDIR = os.path.abspath(os.path.dirname(__file__))
GLDIR = os.path.join(THISDIR, '..', 'vispy', 'gloo', 'gl')
PREAMBLE = '"""GL definitions converted to Python by codegen/createglapi.py.\n\nTHIS CODE IS AUTO-GENERATED. DO NOT EDIT.\n\n%s\n\n"""\n'

def create_parsers():
    if False:
        return 10
    parser1 = headerparser.Parser(os.path.join(THISDIR, 'headers', 'gl2.h'))
    parser2 = headerparser.Parser(os.path.join(THISDIR, 'headers', 'webgl.idl'))
    names1 = set(parser1.constant_names)
    names2 = set(parser2.constant_names)
    if names1 == names2:
        print('Constants in gl2 and webgl are equal')
    else:
        print('===== Extra names in gl2 =====')
        print(', '.join(names1.difference(names2)))
        print('===== Extra names in webgl =====')
        print(', '.join(names2.difference(names1)))
        print('===========')
    superset = names1.intersection(names2)
    constants = {}
    for c1 in parser1.constants.values():
        if c1.shortname in superset:
            constants[c1.shortname] = c1.value
    assert len(constants) == len(superset)
    for c2 in parser2.constants.values():
        if c2.shortname in constants:
            assert c2.value == constants[c2.shortname]
    print('Hooray! All constants that occur in both namespaces have equal values.')
    return (parser1, parser2)
DEFINE_ENUM = '\nclass Enum(int):\n    """Enum (integer) with a meaningfull repr."""\n    \n    def __new__(cls, name, value):\n        base = int.__new__(cls, value)\n        base.name = name\n        return base\n        \n    def __repr__(self):\n        return self.name\n'
DEFINE_CONST_MAP = "\nENUM_MAP = {}\nfor var_name, ob in list(globals().items()):\n    if var_name.startswith('GL_'):\n        ENUM_MAP[int(ob)] = ob\ndel ob, var_name\n"

def create_constants_module(constant_definitions: list[headerparser.ConstantDefinition], extension: bool=False) -> None:
    if False:
        while True:
            i = 10
    lines = [PREAMBLE % 'Constants for OpenGL ES 2.0.', DEFINE_ENUM, '']
    if extension:
        renamed_defs = []
        for c in constant_definitions:
            if 'OES' in c.oname:
                c.oname = c.oname.replace('OES', '')
                c.oname = c.oname.replace('__', '_').strip('_')
                renamed_defs.append(c)
        constant_definitions = renamed_defs
    for c in sorted(constant_definitions, key=lambda x: x.oname):
        if isinstance(c.value, int):
            lines.append('%s = Enum(%r, %r)' % (c.oname, c.oname, c.value))
        else:
            lines.append('%s = %r' % (c.oname, c.value))
    lines.append('')
    lines.append(DEFINE_CONST_MAP)
    fname = '_constants_ext.py' if extension else '_constants.py'
    with open(os.path.join(GLDIR, fname), 'wb') as f:
        f.write('\n'.join(lines).encode('utf-8'))
    print('wrote %s' % fname)
IGNORE_FUNCTIONS = ['releaseShaderCompiler', 'shaderBinary']
WEBGL_EQUIVALENTS = {'genBuffers': 'createBuffer', 'genFramebuffers': 'createFramebuffer', 'genRenderbuffers': 'createRenderbuffer', 'genTextures': 'createTexture', 'deleteBuffers': 'deleteBuffer', 'deleteFramebuffers': 'deleteFramebuffer', 'deleteRenderbuffers': 'deleteRenderbuffer', 'deleteTextures': 'deleteTexture', 'clearDepthf': 'clearDepth', 'depthRangef': 'depthRange', 'getBufferParameteriv': 'getBufferParameter', 'getRenderbufferParameteriv': 'getRenderbufferParameter', 'getFramebufferAttachmentParameteriv': 'getFramebufferAttachmentParameter', 'getVertexAttribPointerv': 'getVertexAttribOffset', 'getProgramiv': 'getProgramParameter', 'getShaderiv': 'getShaderParameter', 'getBooleanv': 'getParameter', 'getFloatv': 'getParameter', 'getIntegerv': 'getParameter', 'getString': 'getParameter'}
EASY_TYPES = {'void': (type(None), 'c_voidp'), 'GLenum': (int, 'c_uint'), 'GLboolean': (bool, 'c_bool'), 'GLuint': (int, 'c_uint'), 'GLint': (int, 'c_int'), 'GLbitfield': (int, 'c_uint'), 'GLsizei': (int, 'c_int'), 'GLfloat': (float, 'c_float'), 'GLclampf': (float, 'c_float')}
HARDER_TYPES = {'GLenum*': ('', 'POINTER(ctypes.c_uint)'), 'GLboolean*': ('', 'POINTER(ctypes.c_bool)'), 'GLuint*': ('', 'POINTER(ctypes.c_uint)'), 'GLint*': ('', 'POINTER(ctypes.c_int)'), 'GLsizei*': ('', 'POINTER(ctypes.c_int)'), 'GLfloat*': ('', 'POINTER(ctypes.c_float)'), 'GLubyte*': ('', 'c_char_p'), 'GLchar*': ('', 'c_char_p'), 'GLchar**': ('', 'POINTER(ctypes.c_char_p)'), 'GLvoid*': ('', 'c_void_p'), 'GLvoid**': ('', 'POINTER(ctypes.c_void_p)'), 'GLintptr': ('', 'c_ssize_t'), 'GLsizeiptr': ('', 'c_ssize_t')}
KNOWN_TYPES = EASY_TYPES.copy()
KNOWN_TYPES.update(HARDER_TYPES)

def apiname(funcname):
    if False:
        print('Hello World!')
    'Define what name the API uses, the short or the gl version.'
    if funcname.startswith('gl'):
        return funcname
    elif funcname.startswith('_'):
        return '_gl' + funcname[1].upper() + funcname[2:]
    else:
        return 'gl' + funcname[0].upper() + funcname[1:]

class FunctionDescription:

    def __init__(self, name, es2func, wglfunc, annfunc):
        if False:
            print('Hello World!')
        self.name = name
        self.apiname = apiname(name)
        self.es2 = es2func
        self.wgl = wglfunc
        self.ann = annfunc
        self.args = []

class ApiGenerator:
    """Base API generator class. We derive several subclasses to implement
    the different backends.
    """
    backend_name = None
    DESCRIPTION = 'GL API X'
    PREAMBLE = ''

    def __init__(self):
        if False:
            print('Hello World!')
        self.lines = []
        self.functions_auto = set()
        self.functions_anno = set()
        self.functions_todo = set()

    def save(self):
        if False:
            print('Hello World!')
        text = '\n'.join(self.lines) + '\n'
        for i in range(10):
            text = text.replace('\n\n\n\n', '\n\n\n')
        with open(self.filename, 'wb') as f:
            f.write((PREAMBLE % self.DESCRIPTION).encode('utf-8'))
            for line in self.PREAMBLE.splitlines():
                f.write(line[4:].encode('utf-8') + b'\n')
            f.write(b'\n')
            f.write(text.encode('utf-8'))

    def add_functions(self, all_functions):
        if False:
            while True:
                i = 10
        for func_description in all_functions:
            self.add_function(func_description)

    def add_function(self, des: FunctionDescription):
        if False:
            while True:
                i = 10
        if des.es2.group:
            if des.name.startswith('get'):
                assert len(des.es2.group) == 2
                des.es2 = des.es2.group[0]
                self._add_function(des)
            else:
                self._add_function_group(des)
        else:
            self._add_function(des)
        self.lines.append('\n')

    def _add_function_group(self, des: FunctionDescription) -> tuple[set, set, set]:
        if False:
            print('Hello World!')
        lines = self.lines
        handled = True
        es2funcs = {}
        for f in des.es2.group:
            cname = f.shortname
            es2funcs[cname] = f
        if des.name == 'uniform':
            for t in ('float', 'int'):
                for i in (1, 2, 3, 4):
                    args = ', '.join(['v%i' % j for j in range(1, i + 1)])
                    cname = 'uniform%i%s' % (i, t[0])
                    sig = '%s(location, %s)' % (apiname(cname), args)
                    self._add_group_function(des, sig, es2funcs[cname])
            for t in ('float', 'int'):
                for i in (1, 2, 3, 4):
                    cname = 'uniform%i%sv' % (i, t[0])
                    sig = '%s(location, count, values)' % apiname(cname)
                    self._add_group_function(des, sig, es2funcs[cname])
        elif des.name == 'uniformMatrix':
            for i in (2, 3, 4):
                cname = 'uniformMatrix%ifv' % i
                sig = '%s(location, count, transpose, values)' % apiname(cname)
                self._add_group_function(des, sig, es2funcs[cname])
        elif des.name == 'vertexAttrib':
            for i in (1, 2, 3, 4):
                args = ', '.join(['v%i' % j for j in range(1, i + 1)])
                cname = 'vertexAttrib%if' % i
                sig = '%s(index, %s)' % (apiname(cname), args)
                self._add_group_function(des, sig, es2funcs[cname])
        elif des.name == 'texParameter':
            for t in ('float', 'int'):
                cname = 'texParameter%s' % t[0]
                sig = '%s(target, pname, param)' % apiname(cname)
                self._add_group_function(des, sig, es2funcs[cname])
        else:
            handled = False
        if handled:
            self.functions_auto.add(des.name)
        else:
            self.functions_todo.add(des.name)
            lines.append('# todo: Dont know group %s' % des.name)

    def _add_function(self, des):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def _add_group_function(self, des, sig, es2func):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

class ProxyApiGenerator(ApiGenerator):
    """Generator for the general proxy class that will be loaded into gloo.gl."""
    filename = os.path.join(GLDIR, '_proxy.py')
    DESCRIPTION = 'Base proxy API for GL ES 2.0.'
    PREAMBLE = '\n    class BaseGLProxy(object):\n        """ Base proxy class for the GL ES 2.0 API. Subclasses should\n        implement __call__ to process the API calls.\n        """\n       \n        def __call__(self, funcname, returns, *args):\n            raise NotImplementedError()\n    '

    def _returns(self, des):
        if False:
            return 10
        shortame = des.name
        for prefix in ('get', 'is', 'check', 'create', 'read'):
            if shortame.startswith(prefix):
                return True
        else:
            return False

    def _add_function(self, des):
        if False:
            i = 10
            return i + 15
        ret = self._returns(des)
        prefix = 'return ' if ret else ''
        argstr = ', '.join(des.args)
        self.lines.append('    def %s(self, %s):' % (des.apiname, argstr))
        self.lines.append('        %sself("%s", %r, %s)' % (prefix, apiname(des.name), ret, argstr))

    def _add_group_function(self, des, sig, es2func):
        if False:
            print('Hello World!')
        ret = self._returns(des)
        prefix = 'return ' if ret else ''
        funcname = apiname(sig.split('(')[0])
        args = sig.split('(', 1)[1].split(')')[0]
        self.lines.append('    def %s(self, %s):' % (funcname, args))
        self.lines.append('        %sself("%s", %r, %s)' % (prefix, funcname, ret, args))

class Gl2ApiGenerator(ApiGenerator):
    """Generator for the gl2 (desktop) backend."""
    filename = os.path.join(GLDIR, '_gl2.py')
    write_c_sig = True
    define_argtypes_in_module = False
    backend_name = 'gl'
    DESCRIPTION = 'Subset of desktop GL API compatible with GL ES 2.0'
    PREAMBLE = '\n    import ctypes\n    from .gl2 import _lib, _get_gl_func\n    '

    def _get_argtype_str(self, es2func):
        if False:
            i = 10
            return i + 15
        ct_arg_types = [KNOWN_TYPES.get(arg.ctype, None) for arg in es2func.args]
        if None in ct_arg_types:
            argstr = 'UNKNOWN_ARGTYPES'
        elif es2func.group:
            argstr = 'UNKNOWN_ARGTYPES'
        else:
            argstr = ', '.join(['ctypes.%s' % t[1] for t in ct_arg_types[1:]])
            argstr = '()' if not argstr else '(%s,)' % argstr
        if ct_arg_types[0][0] != type(None):
            resstr = 'ctypes.%s' % ct_arg_types[0][1]
        else:
            resstr = 'None'
        return (resstr, argstr)

    def _write_argtypes(self, es2func):
        if False:
            for i in range(10):
                print('nop')
        lines = self.lines
        ct_arg_types = [KNOWN_TYPES.get(arg.ctype, None) for arg in es2func.args]
        if None in ct_arg_types:
            lines.append('# todo: unknown argtypes')
        elif es2func.group:
            lines.append('# todo: oops, dont set argtypes for group!')
        elif ct_arg_types[1:]:
            argstr = ', '.join(['ctypes.%s' % t[1] for t in ct_arg_types[1:]])
            lines.append('_lib.%s.argtypes = %s,' % (es2func.glname, argstr))
        else:
            lines.append('_lib.%s.argtypes = ()' % es2func.glname)
        if ct_arg_types[0][0] != type(None):
            lines.append('_lib.%s.restype = ctypes.%s' % (es2func.glname, ct_arg_types[0][1]))

    def _native_call_line(self, name, es2func, cargstr=None, prefix='', indent=4):
        if False:
            return 10
        (resstr, argstr) = self._get_argtype_str(es2func)
        if cargstr is None:
            cargs = [arg.name for arg in es2func.args[1:]]
            cargstr = ', '.join(cargs)
        lines = 'try:\n'
        lines += '    nativefunc = %s._native\n' % apiname(name)
        lines += 'except AttributeError:\n'
        lines += '    nativefunc = %s._native = _get_gl_func("%s", %s, %s)\n' % (apiname(name), es2func.glname, resstr, argstr)
        lines += '%snativefunc(%s)\n' % (prefix, cargstr)
        lines = [' ' * indent + line for line in lines.splitlines()]
        return '\n'.join(lines)

    def _add_function(self, des):
        if False:
            print('Hello World!')
        lines = self.lines
        es2func = des.es2
        if self.define_argtypes_in_module:
            self._write_argtypes(es2func)
        ce_arg_types = [arg.ctype for arg in es2func.args[1:]]
        ce_arg_names = [arg.name for arg in es2func.args[1:]]
        ct_arg_types = [KNOWN_TYPES.get(arg.ctype, None) for arg in es2func.args]
        ct_arg_types_easy = [EASY_TYPES.get(arg.ctype, None) for arg in es2func.args]
        if self.write_c_sig:
            argnamesstr = ', '.join([c_type + ' ' + c_name for (c_type, c_name) in zip(ce_arg_types, ce_arg_names)])
            lines.append('# %s = %s(%s)' % (es2func.args[0].ctype, es2func.oname, argnamesstr))
        lines.append('def %s(%s):' % (des.apiname, ', '.join(des.args)))
        cargs = [arg.name for arg in des.es2.args[1:]]
        if des.ann:
            prefix = 'res = '
            self.functions_anno.add(des.name)
            callline = self._native_call_line(des.name, es2func, prefix=prefix)
            lines.extend(des.ann.get_lines(callline, self.backend_name))
        elif es2func.group:
            self.functions_todo.add(des.name)
            lines.append('    pass  # todo: Oops. this is a group!')
        elif None in ct_arg_types_easy:
            self.functions_todo.add(des.name)
            lines.append('    pass  # todo: Not all easy types!')
        elif des.args != [arg.name for arg in des.wgl.args[1:]]:
            self.functions_todo.add(des.name)
            lines.append('    pass  # todo: ES 2.0 and WebGL args do not match!')
        else:
            self.functions_auto.add(des.name)
            prefix = ''
            if ct_arg_types[0][0] != type(None):
                prefix = 'return '
            elif des.es2.shortname.startswith('get'):
                raise RuntimeError('Get func returns void?')
            callline = self._native_call_line(des.name, des.es2, prefix=prefix)
            lines.append(callline)
        if 'gl2' in self.__class__.__name__.lower():
            if es2func.oname in ('glDepthRangef', 'glClearDepthf'):
                for i in range(1, 10):
                    line = lines[-i]
                    if not line.strip() or line.startswith('#'):
                        break
                    line = line.replace('c_float', 'c_double')
                    line = line.replace('glDepthRangef', 'glDepthRange')
                    line = line.replace('glClearDepthf', 'glClearDepth')
                    lines[-i] = line

    def _add_group_function(self, des, sig, es2func):
        if False:
            for i in range(10):
                print('nop')
        lines = self.lines
        call_line = self._native_call_line
        if self.define_argtypes_in_module:
            self._write_argtypes(es2func)
        funcname = sig.split('(', 1)[0]
        args = sig.split('(', 1)[1].split(')')[0]
        if des.name == 'uniform':
            if funcname[-1] != 'v':
                lines.append('def %s:' % sig)
                lines.append(call_line(funcname, es2func, args))
            else:
                t = {'f': 'float', 'i': 'int'}[funcname[-2]]
                lines.append('def %s:' % sig)
                lines.append('    values = [%s(val) for val in values]' % t)
                lines.append('    values = (ctypes.c_%s*len(values))(*values)' % t)
                lines.append(call_line(funcname, es2func, 'location, count, values'))
        elif des.name == 'uniformMatrix':
            lines.append('def %s:' % sig)
            lines.append('    if not values.flags["C_CONTIGUOUS"]:')
            lines.append('        values = values.copy()')
            lines.append('    assert values.dtype.name == "float32"')
            lines.append('    values_ = values')
            lines.append('    values = values_.ctypes.data_as(ctypes.POINTER(ctypes.c_float))')
            lines.append(call_line(funcname, es2func, 'location, count, transpose, values'))
        elif des.name == 'vertexAttrib':
            lines.append('def %s:' % sig)
            lines.append(call_line(funcname, es2func, args))
        elif des.name == 'texParameter':
            lines.append('def %s:' % sig)
            lines.append(call_line(funcname, es2func, args))
        else:
            raise ValueError('unknown group func')

class Es2ApiGenerator(Gl2ApiGenerator):
    """Generator for the es2 backend (i.e. Angle on Windows). Very
    similar to the gl2 API, but we do not need that deferred loading
    of GL functions here.
    """
    filename = os.path.join(GLDIR, '_es2.py')
    write_c_sig = True
    define_argtypes_in_module = True
    backend_name = 'es'
    DESCRIPTION = 'GL ES 2.0 API (via Angle/DirectX on Windows)'
    PREAMBLE = '\n    import ctypes\n    from .es2 import _lib\n    '

    def _native_call_line(self, name, es2func, cargstr=None, prefix='', indent=4):
        if False:
            while True:
                i = 10
        (resstr, argstr) = self._get_argtype_str(es2func)
        if cargstr is None:
            cargs = [arg.name for arg in es2func.args[1:]]
            cargstr = ', '.join(cargs)
        return ' ' * indent + '%s_lib.%s(%s)' % (prefix, es2func.glname, cargstr)

class PyOpenGL2ApiGenerator(ApiGenerator):
    """Generator for a fallback pyopengl backend."""
    filename = os.path.join(GLDIR, '_pyopengl2.py')
    backend_name = 'pyopengl'
    DESCRIPTION = 'Proxy API for GL ES 2.0 subset, via the PyOpenGL library.'
    PREAMBLE = '\n    import ctypes\n    from OpenGL import GL\n    import OpenGL.GL.framebufferobjects as FBO\n    '

    def __init__(self):
        if False:
            print('Hello World!')
        ApiGenerator.__init__(self)
        self._functions_to_import = []
        self._used_functions = []

    def _add_function(self, des):
        if False:
            print('Hello World!')
        mod = 'GL'
        if 'renderbuffer' in des.name.lower() or 'framebuffer' in des.name.lower():
            mod = 'FBO'
        argstr = ', '.join(des.args)
        call_line = '    return %s.%s(%s)' % (mod, des.es2.glname, argstr)
        ann_lines = []
        if des.ann is not None:
            ann_lines = des.ann.get_lines(call_line, self.backend_name)
        if ann_lines:
            self.lines.append('def %s(%s):' % (des.apiname, argstr))
            self.lines.extend(ann_lines)
            self._used_functions.append(des.es2.glname)
        else:
            self._functions_to_import.append((des.es2.glname, des.apiname))

    def _add_group_function(self, des, sig, es2func):
        if False:
            i = 10
            return i + 15
        funcname = apiname(sig.split('(')[0])
        self._functions_to_import.append((funcname, funcname))

    def save(self):
        if False:
            while True:
                i = 10
        self.lines.append('# List of functions that we should import from OpenGL.GL')
        self.lines.append('_functions_to_import = [')
        for (name1, name2) in self._functions_to_import:
            self.lines.append('    ("%s", "%s"),' % (name1, name2))
        self.lines.append('    ]')
        self.lines.append('')
        self.lines.append('# List of functions in OpenGL.GL that we use')
        self.lines.append('_used_functions = [')
        for name in self._used_functions:
            self.lines.append('    "%s",' % name)
        self.lines.append('    ]')
        ApiGenerator.save(self)

class FunctionCollector:

    def __init__(self, annotations, parser1, parser2):
        if False:
            for i in range(10):
                print('nop')
        self.annotations = annotations
        self.parser1 = parser1
        self.parser2 = parser2
        self.used_webgl_names = set()
        self.functions_auto = set()
        self.functions_anno = set()
        self.functions_todo = set()
        self.all_functions = []

    def collect_function_definitions(self):
        if False:
            return 10
        'Process function definitions of ES 2.0, WebGL and annotations.\n\n        We try to combine information from these three sources to find the\n        arguments for the Python API. In this "merging" process we also\n        check for inconsistencies between the API definitions.\n\n        '
        for name in self.parser1.function_names:
            if name in IGNORE_FUNCTIONS:
                continue
            es2func = self.parser1.functions[name]
            lookupname = WEBGL_EQUIVALENTS.get(es2func.shortname, es2func.shortname)
            wglfunc = self.parser2.functions.get(lookupname, None)
            if wglfunc:
                self.used_webgl_names.add(lookupname)
            else:
                print('WARNING: %s not available in WebGL' % es2func.shortname)
            name = WEBGL_EQUIVALENTS.get(name, name)
            if name == 'getParameter':
                if es2func.shortname != 'getString':
                    name = '_' + es2func.shortname
            annfunc = self.annotations.get(name, None)
            des = FunctionDescription(name, es2func, wglfunc, annfunc)
            self.all_functions.append(des)
            argnames_es2 = [arg.name for arg in es2func.args[1:]]
            if wglfunc:
                argnames_wgl = [arg.name for arg in wglfunc.args[1:]]
            if annfunc:
                argnames_ann = annfunc.args
                argnames_ann = [arg.split('=')[0] for arg in argnames_ann]
            if wglfunc and argnames_es2 == argnames_wgl:
                if annfunc and argnames_ann != argnames_es2:
                    des.args = argnames_ann
                    print('WARNING: %s: Annotation overload even though webgl and es2 match.' % name)
                else:
                    des.args = argnames_es2
            elif wglfunc:
                if annfunc and argnames_ann != argnames_wgl:
                    des.args = argnames_ann
                    print('WARNING: %s: Annotation overload webgl args.' % name)
                else:
                    des.args = argnames_wgl
            else:
                print('WARNING: %s: Could not determine args!!' % name)
            for func in [es2func, wglfunc]:
                if func is None:
                    continue
                group = func.group or [func]
                for f in group:
                    if f.oname.startswith('gl') and (not hasattr(GL, f.glname)):
                        print('WARNING: %s seems not available in PyOpenGL' % f.glname)

    def check_unused_webgl_funcs(self):
        if False:
            for i in range(10):
                print('nop')
        'Check which WebGL functions we did not find/use.'
        for name in set(self.parser2.function_names).difference(self.used_webgl_names):
            print('WARNING: WebGL function %s not in Desktop' % name)

    def report_status(self):
        if False:
            print('Hello World!')
        print('Could generate %i functions automatically, and %i with annotations' % (len(self.functions_auto), len(self.functions_anno)))
        print('Need more info for %i functions.' % len(self.functions_todo))
        if not self.functions_todo:
            print('Hooray! All %i functions are covered!' % len(self.all_functions))

def main():
    if False:
        return 10
    annotations = parse_anotations()
    (gl2es2_parser, webgl_parser) = create_parsers()
    vispy_ext_parser = headerparser.Parser(os.path.join(THISDIR, 'headers', 'vispy_ext.h'))
    constant_definitions = list(gl2es2_parser.constants.values()) + list(vispy_ext_parser.constants.values())
    create_constants_module(constant_definitions)
    func_collector = FunctionCollector(annotations, gl2es2_parser, webgl_parser)
    func_collector.collect_function_definitions()
    for Gen in [ProxyApiGenerator, Gl2ApiGenerator, Es2ApiGenerator, PyOpenGL2ApiGenerator]:
        gen = Gen()
        gen.add_functions(func_collector.all_functions)
        func_collector.functions_auto.update(gen.functions_auto)
        func_collector.functions_anno.update(gen.functions_anno)
        func_collector.functions_todo.update(gen.functions_todo)
        gen.save()
    func_collector.check_unused_webgl_funcs()
    func_collector.report_status()
if __name__ == '__main__':
    sys.exit(main())