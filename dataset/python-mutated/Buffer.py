from __future__ import absolute_import
from .Visitor import CythonTransform
from .ModuleNode import ModuleNode
from .Errors import CompileError
from .UtilityCode import CythonUtilityCode
from .Code import UtilityCode, TempitaUtilityCode
from . import Options
from . import Interpreter
from . import PyrexTypes
from . import Naming
from . import Symtab

def dedent(text, reindent=0):
    if False:
        print('Hello World!')
    from textwrap import dedent
    text = dedent(text)
    if reindent > 0:
        indent = ' ' * reindent
        text = '\n'.join([indent + x for x in text.split('\n')])
    return text

class IntroduceBufferAuxiliaryVars(CythonTransform):
    buffers_exists = False
    using_memoryview = False

    def __call__(self, node):
        if False:
            print('Hello World!')
        assert isinstance(node, ModuleNode)
        self.max_ndim = 0
        result = super(IntroduceBufferAuxiliaryVars, self).__call__(node)
        if self.buffers_exists:
            use_bufstruct_declare_code(node.scope)
        return result

    def handle_scope(self, node, scope):
        if False:
            return 10
        scope_items = scope.entries.items()
        bufvars = [entry for (name, entry) in scope_items if entry.type.is_buffer]
        if len(bufvars) > 0:
            bufvars.sort(key=lambda entry: entry.name)
            self.buffers_exists = True
        memviewslicevars = [entry for (name, entry) in scope_items if entry.type.is_memoryviewslice]
        if len(memviewslicevars) > 0:
            self.buffers_exists = True
        for (name, entry) in scope_items:
            if name == 'memoryview' and isinstance(entry.utility_code_definition, CythonUtilityCode):
                self.using_memoryview = True
                break
        del scope_items
        if isinstance(node, ModuleNode) and len(bufvars) > 0:
            raise CompileError(node.pos, 'Buffer vars not allowed in module scope')
        for entry in bufvars:
            if entry.type.dtype.is_ptr:
                raise CompileError(node.pos, 'Buffers with pointer types not yet supported.')
            name = entry.name
            buftype = entry.type
            if buftype.ndim > Options.buffer_max_dims:
                raise CompileError(node.pos, 'Buffer ndims exceeds Options.buffer_max_dims = %d' % Options.buffer_max_dims)
            if buftype.ndim > self.max_ndim:
                self.max_ndim = buftype.ndim

            def decvar(type, prefix):
                if False:
                    print('Hello World!')
                cname = scope.mangle(prefix, name)
                aux_var = scope.declare_var(name=None, cname=cname, type=type, pos=node.pos)
                if entry.is_arg:
                    aux_var.used = True
                return aux_var
            auxvars = ((PyrexTypes.c_pyx_buffer_nd_type, Naming.pybuffernd_prefix), (PyrexTypes.c_pyx_buffer_type, Naming.pybufferstruct_prefix))
            (pybuffernd, rcbuffer) = [decvar(type, prefix) for (type, prefix) in auxvars]
            entry.buffer_aux = Symtab.BufferAux(pybuffernd, rcbuffer)
        scope.buffer_entries = bufvars
        self.scope = scope

    def visit_ModuleNode(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.handle_scope(node, node.scope)
        self.visitchildren(node)
        return node

    def visit_FuncDefNode(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.handle_scope(node, node.local_scope)
        self.visitchildren(node)
        return node
buffer_options = ('dtype', 'ndim', 'mode', 'negative_indices', 'cast')
buffer_defaults = {'ndim': 1, 'mode': 'full', 'negative_indices': True, 'cast': False}
buffer_positional_options_count = 1
ERR_BUF_OPTION_UNKNOWN = '"%s" is not a buffer option'
ERR_BUF_TOO_MANY = 'Too many buffer options'
ERR_BUF_DUP = '"%s" buffer option already supplied'
ERR_BUF_MISSING = '"%s" missing'
ERR_BUF_MODE = 'Only allowed buffer modes are: "c", "fortran", "full", "strided" (as a compile-time string)'
ERR_BUF_NDIM = 'ndim must be a non-negative integer'
ERR_BUF_DTYPE = 'dtype must be "object", numeric type or a struct'
ERR_BUF_BOOL = '"%s" must be a boolean'

def analyse_buffer_options(globalpos, env, posargs, dictargs, defaults=None, need_complete=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Must be called during type analysis, as analyse is called\n    on the dtype argument.\n\n    posargs and dictargs should consist of a list and a dict\n    of tuples (value, pos). Defaults should be a dict of values.\n\n    Returns a dict containing all the options a buffer can have and\n    its value (with the positions stripped).\n    '
    if defaults is None:
        defaults = buffer_defaults
    (posargs, dictargs) = Interpreter.interpret_compiletime_options(posargs, dictargs, type_env=env, type_args=(0, 'dtype'))
    if len(posargs) > buffer_positional_options_count:
        raise CompileError(posargs[-1][1], ERR_BUF_TOO_MANY)
    options = {}
    for (name, (value, pos)) in dictargs.items():
        if name not in buffer_options:
            raise CompileError(pos, ERR_BUF_OPTION_UNKNOWN % name)
        options[name] = value
    for (name, (value, pos)) in zip(buffer_options, posargs):
        if name not in buffer_options:
            raise CompileError(pos, ERR_BUF_OPTION_UNKNOWN % name)
        if name in options:
            raise CompileError(pos, ERR_BUF_DUP % name)
        options[name] = value
    for name in buffer_options:
        if name not in options:
            try:
                options[name] = defaults[name]
            except KeyError:
                if need_complete:
                    raise CompileError(globalpos, ERR_BUF_MISSING % name)
    dtype = options.get('dtype')
    if dtype and dtype.is_extension_type:
        raise CompileError(globalpos, ERR_BUF_DTYPE)
    ndim = options.get('ndim')
    if ndim and (not isinstance(ndim, int) or ndim < 0):
        raise CompileError(globalpos, ERR_BUF_NDIM)
    mode = options.get('mode')
    if mode and (not mode in ('full', 'strided', 'c', 'fortran')):
        raise CompileError(globalpos, ERR_BUF_MODE)

    def assert_bool(name):
        if False:
            for i in range(10):
                print('nop')
        x = options.get(name)
        if not isinstance(x, bool):
            raise CompileError(globalpos, ERR_BUF_BOOL % name)
    assert_bool('negative_indices')
    assert_bool('cast')
    return options

class BufferEntry(object):

    def __init__(self, entry):
        if False:
            print('Hello World!')
        self.entry = entry
        self.type = entry.type
        self.cname = entry.buffer_aux.buflocal_nd_var.cname
        self.buf_ptr = '%s.rcbuffer->pybuffer.buf' % self.cname
        self.buf_ptr_type = entry.type.buffer_ptr_type
        self.init_attributes()

    def init_attributes(self):
        if False:
            return 10
        self.shape = self.get_buf_shapevars()
        self.strides = self.get_buf_stridevars()
        self.suboffsets = self.get_buf_suboffsetvars()

    def get_buf_suboffsetvars(self):
        if False:
            return 10
        return self._for_all_ndim('%s.diminfo[%d].suboffsets')

    def get_buf_stridevars(self):
        if False:
            i = 10
            return i + 15
        return self._for_all_ndim('%s.diminfo[%d].strides')

    def get_buf_shapevars(self):
        if False:
            return 10
        return self._for_all_ndim('%s.diminfo[%d].shape')

    def _for_all_ndim(self, s):
        if False:
            i = 10
            return i + 15
        return [s % (self.cname, i) for i in range(self.type.ndim)]

    def generate_buffer_lookup_code(self, code, index_cnames):
        if False:
            i = 10
            return i + 15
        params = []
        nd = self.type.ndim
        mode = self.type.mode
        if mode == 'full':
            for (i, s, o) in zip(index_cnames, self.get_buf_stridevars(), self.get_buf_suboffsetvars()):
                params.append(i)
                params.append(s)
                params.append(o)
            funcname = '__Pyx_BufPtrFull%dd' % nd
            funcgen = buf_lookup_full_code
        else:
            if mode == 'strided':
                funcname = '__Pyx_BufPtrStrided%dd' % nd
                funcgen = buf_lookup_strided_code
            elif mode == 'c':
                funcname = '__Pyx_BufPtrCContig%dd' % nd
                funcgen = buf_lookup_c_code
            elif mode == 'fortran':
                funcname = '__Pyx_BufPtrFortranContig%dd' % nd
                funcgen = buf_lookup_fortran_code
            else:
                assert False
            for (i, s) in zip(index_cnames, self.get_buf_stridevars()):
                params.append(i)
                params.append(s)
        if funcname not in code.globalstate.utility_codes:
            code.globalstate.utility_codes.add(funcname)
            protocode = code.globalstate['utility_code_proto']
            defcode = code.globalstate['utility_code_def']
            funcgen(protocode, defcode, name=funcname, nd=nd)
        buf_ptr_type_code = self.buf_ptr_type.empty_declaration_code()
        ptrcode = '%s(%s, %s, %s)' % (funcname, buf_ptr_type_code, self.buf_ptr, ', '.join(params))
        return ptrcode

def get_flags(buffer_aux, buffer_type):
    if False:
        return 10
    flags = 'PyBUF_FORMAT'
    mode = buffer_type.mode
    if mode == 'full':
        flags += '| PyBUF_INDIRECT'
    elif mode == 'strided':
        flags += '| PyBUF_STRIDES'
    elif mode == 'c':
        flags += '| PyBUF_C_CONTIGUOUS'
    elif mode == 'fortran':
        flags += '| PyBUF_F_CONTIGUOUS'
    else:
        assert False
    if buffer_aux.writable_needed:
        flags += '| PyBUF_WRITABLE'
    return flags

def used_buffer_aux_vars(entry):
    if False:
        while True:
            i = 10
    buffer_aux = entry.buffer_aux
    buffer_aux.buflocal_nd_var.used = True
    buffer_aux.rcbuf_var.used = True

def put_unpack_buffer_aux_into_scope(buf_entry, code):
    if False:
        while True:
            i = 10
    (buffer_aux, mode) = (buf_entry.buffer_aux, buf_entry.type.mode)
    pybuffernd_struct = buffer_aux.buflocal_nd_var.cname
    fldnames = ['strides', 'shape']
    if mode == 'full':
        fldnames.append('suboffsets')
    ln = []
    for i in range(buf_entry.type.ndim):
        for fldname in fldnames:
            ln.append('%s.diminfo[%d].%s = %s.rcbuffer->pybuffer.%s[%d];' % (pybuffernd_struct, i, fldname, pybuffernd_struct, fldname, i))
    code.putln(' '.join(ln))

def put_init_vars(entry, code):
    if False:
        while True:
            i = 10
    bufaux = entry.buffer_aux
    pybuffernd_struct = bufaux.buflocal_nd_var.cname
    pybuffer_struct = bufaux.rcbuf_var.cname
    code.putln('%s.pybuffer.buf = NULL;' % pybuffer_struct)
    code.putln('%s.refcount = 0;' % pybuffer_struct)
    code.putln('%s.data = NULL;' % pybuffernd_struct)
    code.putln('%s.rcbuffer = &%s;' % (pybuffernd_struct, pybuffer_struct))

def put_acquire_arg_buffer(entry, code, pos):
    if False:
        for i in range(10):
            print('nop')
    buffer_aux = entry.buffer_aux
    getbuffer = get_getbuffer_call(code, entry.cname, buffer_aux, entry.type)
    code.putln('{')
    code.putln('__Pyx_BufFmt_StackElem __pyx_stack[%d];' % entry.type.dtype.struct_nesting_depth())
    code.putln(code.error_goto_if('%s == -1' % getbuffer, pos))
    code.putln('}')
    put_unpack_buffer_aux_into_scope(entry, code)

def put_release_buffer_code(code, entry):
    if False:
        while True:
            i = 10
    code.globalstate.use_utility_code(acquire_utility_code)
    code.putln('__Pyx_SafeReleaseBuffer(&%s.rcbuffer->pybuffer);' % entry.buffer_aux.buflocal_nd_var.cname)

def get_getbuffer_call(code, obj_cname, buffer_aux, buffer_type):
    if False:
        for i in range(10):
            print('nop')
    ndim = buffer_type.ndim
    cast = int(buffer_type.cast)
    flags = get_flags(buffer_aux, buffer_type)
    pybuffernd_struct = buffer_aux.buflocal_nd_var.cname
    dtype_typeinfo = get_type_information_cname(code, buffer_type.dtype)
    code.globalstate.use_utility_code(acquire_utility_code)
    return '__Pyx_GetBufferAndValidate(&%(pybuffernd_struct)s.rcbuffer->pybuffer, (PyObject*)%(obj_cname)s, &%(dtype_typeinfo)s, %(flags)s, %(ndim)d, %(cast)d, __pyx_stack)' % locals()

def put_assign_to_buffer(lhs_cname, rhs_cname, buf_entry, is_initialized, pos, code):
    if False:
        while True:
            i = 10
    '\n    Generate code for reassigning a buffer variables. This only deals with getting\n    the buffer auxiliary structure and variables set up correctly, the assignment\n    itself and refcounting is the responsibility of the caller.\n\n    However, the assignment operation may throw an exception so that the reassignment\n    never happens.\n\n    Depending on the circumstances there are two possible outcomes:\n    - Old buffer released, new acquired, rhs assigned to lhs\n    - Old buffer released, new acquired which fails, reaqcuire old lhs buffer\n      (which may or may not succeed).\n    '
    (buffer_aux, buffer_type) = (buf_entry.buffer_aux, buf_entry.type)
    pybuffernd_struct = buffer_aux.buflocal_nd_var.cname
    flags = get_flags(buffer_aux, buffer_type)
    code.putln('{')
    code.putln('__Pyx_BufFmt_StackElem __pyx_stack[%d];' % buffer_type.dtype.struct_nesting_depth())
    getbuffer = get_getbuffer_call(code, '%s', buffer_aux, buffer_type)
    if is_initialized:
        code.putln('__Pyx_SafeReleaseBuffer(&%s.rcbuffer->pybuffer);' % pybuffernd_struct)
        retcode_cname = code.funcstate.allocate_temp(PyrexTypes.c_int_type, manage_ref=False)
        code.putln('%s = %s;' % (retcode_cname, getbuffer % rhs_cname))
        code.putln('if (%s) {' % code.unlikely('%s < 0' % retcode_cname))
        exc_temps = tuple((code.funcstate.allocate_temp(PyrexTypes.py_object_type, manage_ref=False) for _ in range(3)))
        code.putln('PyErr_Fetch(&%s, &%s, &%s);' % exc_temps)
        code.putln('if (%s) {' % code.unlikely('%s == -1' % (getbuffer % lhs_cname)))
        code.putln('Py_XDECREF(%s); Py_XDECREF(%s); Py_XDECREF(%s);' % exc_temps)
        code.globalstate.use_utility_code(raise_buffer_fallback_code)
        code.putln('__Pyx_RaiseBufferFallbackError();')
        code.putln('} else {')
        code.putln('PyErr_Restore(%s, %s, %s);' % exc_temps)
        code.putln('}')
        code.putln('%s = %s = %s = 0;' % exc_temps)
        for t in exc_temps:
            code.funcstate.release_temp(t)
        code.putln('}')
        put_unpack_buffer_aux_into_scope(buf_entry, code)
        code.putln(code.error_goto_if_neg(retcode_cname, pos))
        code.funcstate.release_temp(retcode_cname)
    else:
        code.putln('if (%s) {' % code.unlikely('%s == -1' % (getbuffer % rhs_cname)))
        code.putln('%s = %s; __Pyx_INCREF(Py_None); %s.rcbuffer->pybuffer.buf = NULL;' % (lhs_cname, PyrexTypes.typecast(buffer_type, PyrexTypes.py_object_type, 'Py_None'), pybuffernd_struct))
        code.putln(code.error_goto(pos))
        code.put('} else {')
        put_unpack_buffer_aux_into_scope(buf_entry, code)
        code.putln('}')
    code.putln('}')

def put_buffer_lookup_code(entry, index_signeds, index_cnames, directives, pos, code, negative_indices, in_nogil_context):
    if False:
        return 10
    '\n    Generates code to process indices and calculate an offset into\n    a buffer. Returns a C string which gives a pointer which can be\n    read from or written to at will (it is an expression so caller should\n    store it in a temporary if it is used more than once).\n\n    As the bounds checking can have any number of combinations of unsigned\n    arguments, smart optimizations etc. we insert it directly in the function\n    body. The lookup however is delegated to a inline function that is instantiated\n    once per ndim (lookup with suboffsets tend to get quite complicated).\n\n    entry is a BufferEntry\n    '
    negative_indices = directives['wraparound'] and negative_indices
    if directives['boundscheck']:
        failed_dim_temp = code.funcstate.allocate_temp(PyrexTypes.c_int_type, manage_ref=False)
        code.putln('%s = -1;' % failed_dim_temp)
        for (dim, (signed, cname, shape)) in enumerate(zip(index_signeds, index_cnames, entry.get_buf_shapevars())):
            if signed != 0:
                code.putln('if (%s < 0) {' % cname)
                if negative_indices:
                    code.putln('%s += %s;' % (cname, shape))
                    code.putln('if (%s) %s = %d;' % (code.unlikely('%s < 0' % cname), failed_dim_temp, dim))
                else:
                    code.putln('%s = %d;' % (failed_dim_temp, dim))
                code.put('} else ')
            if signed != 0:
                cast = ''
            else:
                cast = '(size_t)'
            code.putln('if (%s) %s = %d;' % (code.unlikely('%s >= %s%s' % (cname, cast, shape)), failed_dim_temp, dim))
        if in_nogil_context:
            code.globalstate.use_utility_code(raise_indexerror_nogil)
            func = '__Pyx_RaiseBufferIndexErrorNogil'
        else:
            code.globalstate.use_utility_code(raise_indexerror_code)
            func = '__Pyx_RaiseBufferIndexError'
        code.putln('if (%s) {' % code.unlikely('%s != -1' % failed_dim_temp))
        code.putln('%s(%s);' % (func, failed_dim_temp))
        code.putln(code.error_goto(pos))
        code.putln('}')
        code.funcstate.release_temp(failed_dim_temp)
    elif negative_indices:
        for (signed, cname, shape) in zip(index_signeds, index_cnames, entry.get_buf_shapevars()):
            if signed != 0:
                code.putln('if (%s < 0) %s += %s;' % (cname, cname, shape))
    return entry.generate_buffer_lookup_code(code, index_cnames)

def use_bufstruct_declare_code(env):
    if False:
        print('Hello World!')
    env.use_utility_code(buffer_struct_declare_code)

def buf_lookup_full_code(proto, defin, name, nd):
    if False:
        return 10
    '\n    Generates a buffer lookup function for the right number\n    of dimensions. The function gives back a void* at the right location.\n    '
    macroargs = ', '.join(['i%d, s%d, o%d' % (i, i, i) for i in range(nd)])
    proto.putln('#define %s(type, buf, %s) (type)(%s_imp(buf, %s))' % (name, macroargs, name, macroargs))
    funcargs = ', '.join(['Py_ssize_t i%d, Py_ssize_t s%d, Py_ssize_t o%d' % (i, i, i) for i in range(nd)])
    proto.putln('static CYTHON_INLINE void* %s_imp(void* buf, %s);' % (name, funcargs))
    defin.putln(dedent('\n        static CYTHON_INLINE void* %s_imp(void* buf, %s) {\n          char* ptr = (char*)buf;\n        ') % (name, funcargs) + ''.join([dedent('          ptr += s%d * i%d;\n          if (o%d >= 0) ptr = *((char**)ptr) + o%d;\n        ') % (i, i, i, i) for i in range(nd)]) + '\nreturn ptr;\n}')

def buf_lookup_strided_code(proto, defin, name, nd):
    if False:
        while True:
            i = 10
    '\n    Generates a buffer lookup function for the right number\n    of dimensions. The function gives back a void* at the right location.\n    '
    args = ', '.join(['i%d, s%d' % (i, i) for i in range(nd)])
    offset = ' + '.join(['i%d * s%d' % (i, i) for i in range(nd)])
    proto.putln('#define %s(type, buf, %s) (type)((char*)buf + %s)' % (name, args, offset))

def buf_lookup_c_code(proto, defin, name, nd):
    if False:
        for i in range(10):
            print('nop')
    "\n    Similar to strided lookup, but can assume that the last dimension\n    doesn't need a multiplication as long as.\n    Still we keep the same signature for now.\n    "
    if nd == 1:
        proto.putln('#define %s(type, buf, i0, s0) ((type)buf + i0)' % name)
    else:
        args = ', '.join(['i%d, s%d' % (i, i) for i in range(nd)])
        offset = ' + '.join(['i%d * s%d' % (i, i) for i in range(nd - 1)])
        proto.putln('#define %s(type, buf, %s) ((type)((char*)buf + %s) + i%d)' % (name, args, offset, nd - 1))

def buf_lookup_fortran_code(proto, defin, name, nd):
    if False:
        return 10
    '\n    Like C lookup, but the first index is optimized instead.\n    '
    if nd == 1:
        proto.putln('#define %s(type, buf, i0, s0) ((type)buf + i0)' % name)
    else:
        args = ', '.join(['i%d, s%d' % (i, i) for i in range(nd)])
        offset = ' + '.join(['i%d * s%d' % (i, i) for i in range(1, nd)])
        proto.putln('#define %s(type, buf, %s) ((type)((char*)buf + %s) + i%d)' % (name, args, offset, 0))

def mangle_dtype_name(dtype):
    if False:
        while True:
            i = 10
    if dtype.is_pyobject:
        return 'object'
    elif dtype.is_ptr:
        return 'ptr'
    else:
        if dtype.is_typedef or dtype.is_struct_or_union:
            prefix = 'nn_'
        else:
            prefix = ''
        return prefix + dtype.specialization_name()

def get_type_information_cname(code, dtype, maxdepth=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Output the run-time type information (__Pyx_TypeInfo) for given dtype,\n    and return the name of the type info struct.\n\n    Structs with two floats of the same size are encoded as complex numbers.\n    One can separate between complex numbers declared as struct or with native\n    encoding by inspecting to see if the fields field of the type is\n    filled in.\n    '
    namesuffix = mangle_dtype_name(dtype)
    name = '__Pyx_TypeInfo_%s' % namesuffix
    structinfo_name = '__Pyx_StructFields_%s' % namesuffix
    if dtype.is_error:
        return '<error>'
    if maxdepth is None:
        maxdepth = dtype.struct_nesting_depth()
    if maxdepth <= 0:
        assert False
    if name not in code.globalstate.utility_codes:
        code.globalstate.utility_codes.add(name)
        typecode = code.globalstate['typeinfo']
        arraysizes = []
        if dtype.is_array:
            while dtype.is_array:
                arraysizes.append(dtype.size)
                dtype = dtype.base_type
        complex_possible = dtype.is_struct_or_union and dtype.can_be_complex()
        declcode = dtype.empty_declaration_code()
        if dtype.is_simple_buffer_dtype():
            structinfo_name = 'NULL'
        elif dtype.is_struct:
            struct_scope = dtype.scope
            if dtype.is_cv_qualified:
                struct_scope = struct_scope.base_type_scope
            fields = struct_scope.var_entries
            assert len(fields) > 0
            types = [get_type_information_cname(code, f.type, maxdepth - 1) for f in fields]
            typecode.putln('static __Pyx_StructField %s[] = {' % structinfo_name, safe=True)
            if dtype.is_cv_qualified:
                struct_type = dtype.cv_base_type.empty_declaration_code()
            else:
                struct_type = dtype.empty_declaration_code()
            for (f, typeinfo) in zip(fields, types):
                typecode.putln('  {&%s, "%s", offsetof(%s, %s)},' % (typeinfo, f.name, struct_type, f.cname), safe=True)
            typecode.putln('  {NULL, NULL, 0}', safe=True)
            typecode.putln('};', safe=True)
        else:
            assert False
        rep = str(dtype)
        flags = '0'
        is_unsigned = '0'
        if dtype is PyrexTypes.c_char_type:
            is_unsigned = '__PYX_IS_UNSIGNED(%s)' % declcode
            typegroup = "'H'"
        elif dtype.is_int:
            is_unsigned = '__PYX_IS_UNSIGNED(%s)' % declcode
            typegroup = "%s ? 'U' : 'I'" % is_unsigned
        elif complex_possible or dtype.is_complex:
            typegroup = "'C'"
        elif dtype.is_float:
            typegroup = "'R'"
        elif dtype.is_struct:
            typegroup = "'S'"
            if dtype.packed:
                flags = '__PYX_BUF_FLAGS_PACKED_STRUCT'
        elif dtype.is_pyobject:
            typegroup = "'O'"
        else:
            assert False, dtype
        typeinfo = 'static __Pyx_TypeInfo %s = { "%s", %s, sizeof(%s), { %s }, %s, %s, %s, %s };'
        tup = (name, rep, structinfo_name, declcode, ', '.join([str(x) for x in arraysizes]) or '0', len(arraysizes), typegroup, is_unsigned, flags)
        typecode.putln(typeinfo % tup, safe=True)
    return name

def load_buffer_utility(util_code_name, context=None, **kwargs):
    if False:
        while True:
            i = 10
    if context is None:
        return UtilityCode.load(util_code_name, 'Buffer.c', **kwargs)
    else:
        return TempitaUtilityCode.load(util_code_name, 'Buffer.c', context=context, **kwargs)
context = dict(max_dims=Options.buffer_max_dims)
buffer_struct_declare_code = load_buffer_utility('BufferStructDeclare', context=context)
buffer_formats_declare_code = load_buffer_utility('BufferFormatStructs')
raise_indexerror_code = load_buffer_utility('BufferIndexError')
raise_indexerror_nogil = load_buffer_utility('BufferIndexErrorNogil')
raise_buffer_fallback_code = load_buffer_utility('BufferFallbackError')
acquire_utility_code = load_buffer_utility('BufferGetAndValidate', context=context)
buffer_format_check_code = load_buffer_utility('BufferFormatCheck', context=context)
_typeinfo_to_format_code = load_buffer_utility('TypeInfoToFormat')