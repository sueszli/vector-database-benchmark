"""
This is a direct translation of nvvm.h
"""
import logging
import re
import sys
import warnings
from ctypes import c_void_p, c_int, POINTER, c_char_p, c_size_t, byref, c_char
import threading
from llvmlite import ir
from .error import NvvmError, NvvmSupportError, NvvmWarning
from .libs import get_libdevice, open_libdevice, open_cudalib
from numba.core import cgutils, config
logger = logging.getLogger(__name__)
ADDRSPACE_GENERIC = 0
ADDRSPACE_GLOBAL = 1
ADDRSPACE_SHARED = 3
ADDRSPACE_CONSTANT = 4
ADDRSPACE_LOCAL = 5
nvvm_program = c_void_p
nvvm_result = c_int
RESULT_CODE_NAMES = '\nNVVM_SUCCESS\nNVVM_ERROR_OUT_OF_MEMORY\nNVVM_ERROR_PROGRAM_CREATION_FAILURE\nNVVM_ERROR_IR_VERSION_MISMATCH\nNVVM_ERROR_INVALID_INPUT\nNVVM_ERROR_INVALID_PROGRAM\nNVVM_ERROR_INVALID_IR\nNVVM_ERROR_INVALID_OPTION\nNVVM_ERROR_NO_MODULE_IN_PROGRAM\nNVVM_ERROR_COMPILATION\n'.split()
for (i, k) in enumerate(RESULT_CODE_NAMES):
    setattr(sys.modules[__name__], k, i)
_datalayout_original = 'e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64'
_datalayout_i128 = 'e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64'

def is_available():
    if False:
        while True:
            i = 10
    '\n    Return if libNVVM is available\n    '
    try:
        NVVM()
    except NvvmSupportError:
        return False
    else:
        return True
_nvvm_lock = threading.Lock()

class NVVM(object):
    """Process-wide singleton.
    """
    _PROTOTYPES = {'nvvmVersion': (nvvm_result, POINTER(c_int), POINTER(c_int)), 'nvvmCreateProgram': (nvvm_result, POINTER(nvvm_program)), 'nvvmDestroyProgram': (nvvm_result, POINTER(nvvm_program)), 'nvvmAddModuleToProgram': (nvvm_result, nvvm_program, c_char_p, c_size_t, c_char_p), 'nvvmLazyAddModuleToProgram': (nvvm_result, nvvm_program, c_char_p, c_size_t, c_char_p), 'nvvmCompileProgram': (nvvm_result, nvvm_program, c_int, POINTER(c_char_p)), 'nvvmGetCompiledResultSize': (nvvm_result, nvvm_program, POINTER(c_size_t)), 'nvvmGetCompiledResult': (nvvm_result, nvvm_program, c_char_p), 'nvvmGetProgramLogSize': (nvvm_result, nvvm_program, POINTER(c_size_t)), 'nvvmGetProgramLog': (nvvm_result, nvvm_program, c_char_p), 'nvvmIRVersion': (nvvm_result, POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int)), 'nvvmVerifyProgram': (nvvm_result, nvvm_program, c_int, POINTER(c_char_p))}
    __INSTANCE = None

    def __new__(cls):
        if False:
            return 10
        with _nvvm_lock:
            if cls.__INSTANCE is None:
                cls.__INSTANCE = inst = object.__new__(cls)
                try:
                    inst.driver = open_cudalib('nvvm')
                except OSError as e:
                    cls.__INSTANCE = None
                    errmsg = 'libNVVM cannot be found. Do `conda install cudatoolkit`:\n%s'
                    raise NvvmSupportError(errmsg % e)
                for (name, proto) in inst._PROTOTYPES.items():
                    func = getattr(inst.driver, name)
                    func.restype = proto[0]
                    func.argtypes = proto[1:]
                    setattr(inst, name, func)
        return cls.__INSTANCE

    def __init__(self):
        if False:
            return 10
        ir_versions = self.get_ir_version()
        self._majorIR = ir_versions[0]
        self._minorIR = ir_versions[1]
        self._majorDbg = ir_versions[2]
        self._minorDbg = ir_versions[3]
        self._supported_ccs = get_supported_ccs()

    @property
    def data_layout(self):
        if False:
            return 10
        if (self._majorIR, self._minorIR) < (1, 8):
            return _datalayout_original
        else:
            return _datalayout_i128

    @property
    def supported_ccs(self):
        if False:
            i = 10
            return i + 15
        return self._supported_ccs

    def get_version(self):
        if False:
            i = 10
            return i + 15
        major = c_int()
        minor = c_int()
        err = self.nvvmVersion(byref(major), byref(minor))
        self.check_error(err, 'Failed to get version.')
        return (major.value, minor.value)

    def get_ir_version(self):
        if False:
            for i in range(10):
                print('nop')
        majorIR = c_int()
        minorIR = c_int()
        majorDbg = c_int()
        minorDbg = c_int()
        err = self.nvvmIRVersion(byref(majorIR), byref(minorIR), byref(majorDbg), byref(minorDbg))
        self.check_error(err, 'Failed to get IR version.')
        return (majorIR.value, minorIR.value, majorDbg.value, minorDbg.value)

    def check_error(self, error, msg, exit=False):
        if False:
            while True:
                i = 10
        if error:
            exc = NvvmError(msg, RESULT_CODE_NAMES[error])
            if exit:
                print(exc)
                sys.exit(1)
            else:
                raise exc

class CompilationUnit(object):

    def __init__(self):
        if False:
            return 10
        self.driver = NVVM()
        self._handle = nvvm_program()
        err = self.driver.nvvmCreateProgram(byref(self._handle))
        self.driver.check_error(err, 'Failed to create CU')

    def __del__(self):
        if False:
            print('Hello World!')
        driver = NVVM()
        err = driver.nvvmDestroyProgram(byref(self._handle))
        driver.check_error(err, 'Failed to destroy CU', exit=True)

    def add_module(self, buffer):
        if False:
            i = 10
            return i + 15
        '\n         Add a module level NVVM IR to a compilation unit.\n         - The buffer should contain an NVVM module IR either in the bitcode\n           representation (LLVM3.0) or in the text representation.\n        '
        err = self.driver.nvvmAddModuleToProgram(self._handle, buffer, len(buffer), None)
        self.driver.check_error(err, 'Failed to add module')

    def lazy_add_module(self, buffer):
        if False:
            print('Hello World!')
        '\n        Lazily add an NVVM IR module to a compilation unit.\n        The buffer should contain NVVM module IR either in the bitcode\n        representation or in the text representation.\n        '
        err = self.driver.nvvmLazyAddModuleToProgram(self._handle, buffer, len(buffer), None)
        self.driver.check_error(err, 'Failed to add module')

    def compile(self, **options):
        if False:
            for i in range(10):
                print('nop')
        'Perform Compilation.\n\n        Compilation options are accepted as keyword arguments, with the\n        following considerations:\n\n        - Underscores (`_`) in option names are converted to dashes (`-`), to\n          match NVVM\'s option name format.\n        - Options that take a value will be emitted in the form\n          "-<name>=<value>".\n        - Booleans passed as option values will be converted to integers.\n        - Options which take no value (such as `-gen-lto`) should have a value\n          of `None` passed in and will be emitted in the form "-<name>".\n\n        For documentation on NVVM compilation options, see the CUDA Toolkit\n        Documentation:\n\n        https://docs.nvidia.com/cuda/libnvvm-api/index.html#_CPPv418nvvmCompileProgram11nvvmProgramiPPKc\n        '

        def stringify_option(k, v):
            if False:
                i = 10
                return i + 15
            k = k.replace('_', '-')
            if v is None:
                return f'-{k}'
            if isinstance(v, bool):
                v = int(v)
            return f'-{k}={v}'
        options = [stringify_option(k, v) for (k, v) in options.items()]
        c_opts = (c_char_p * len(options))(*[c_char_p(x.encode('utf8')) for x in options])
        err = self.driver.nvvmVerifyProgram(self._handle, len(options), c_opts)
        self._try_error(err, 'Failed to verify\n')
        err = self.driver.nvvmCompileProgram(self._handle, len(options), c_opts)
        self._try_error(err, 'Failed to compile\n')
        reslen = c_size_t()
        err = self.driver.nvvmGetCompiledResultSize(self._handle, byref(reslen))
        self._try_error(err, 'Failed to get size of compiled result.')
        ptxbuf = (c_char * reslen.value)()
        err = self.driver.nvvmGetCompiledResult(self._handle, ptxbuf)
        self._try_error(err, 'Failed to get compiled result.')
        self.log = self.get_log()
        if self.log:
            warnings.warn(self.log, category=NvvmWarning)
        return ptxbuf[:]

    def _try_error(self, err, msg):
        if False:
            print('Hello World!')
        self.driver.check_error(err, '%s\n%s' % (msg, self.get_log()))

    def get_log(self):
        if False:
            print('Hello World!')
        reslen = c_size_t()
        err = self.driver.nvvmGetProgramLogSize(self._handle, byref(reslen))
        self.driver.check_error(err, 'Failed to get compilation log size.')
        if reslen.value > 1:
            logbuf = (c_char * reslen.value)()
            err = self.driver.nvvmGetProgramLog(self._handle, logbuf)
            self.driver.check_error(err, 'Failed to get compilation log.')
            return logbuf.value.decode('utf8')
        return ''
COMPUTE_CAPABILITIES = ((3, 5), (3, 7), (5, 0), (5, 2), (5, 3), (6, 0), (6, 1), (6, 2), (7, 0), (7, 2), (7, 5), (8, 0), (8, 6), (8, 7), (8, 9), (9, 0))
CTK_SUPPORTED = {(11, 2): ((3, 5), (8, 6)), (11, 3): ((3, 5), (8, 6)), (11, 4): ((3, 5), (8, 7)), (11, 5): ((3, 5), (8, 7)), (11, 6): ((3, 5), (8, 7)), (11, 7): ((3, 5), (8, 7)), (11, 8): ((3, 5), (9, 0)), (12, 0): ((5, 0), (9, 0)), (12, 1): ((5, 0), (9, 0)), (12, 2): ((5, 0), (9, 0))}

def ccs_supported_by_ctk(ctk_version):
    if False:
        i = 10
        return i + 15
    try:
        (min_cc, max_cc) = CTK_SUPPORTED[ctk_version]
        return tuple([cc for cc in COMPUTE_CAPABILITIES if min_cc <= cc <= max_cc])
    except KeyError:
        return tuple([cc for cc in COMPUTE_CAPABILITIES if cc >= config.CUDA_DEFAULT_PTX_CC])

def get_supported_ccs():
    if False:
        for i in range(10):
            print('nop')
    try:
        from numba.cuda.cudadrv.runtime import runtime
        cudart_version = runtime.get_version()
    except:
        _supported_cc = ()
        return _supported_cc
    min_cudart = min(CTK_SUPPORTED)
    if cudart_version < min_cudart:
        _supported_cc = ()
        ctk_ver = f'{cudart_version[0]}.{cudart_version[1]}'
        unsupported_ver = f'CUDA Toolkit {ctk_ver} is unsupported by Numba - {min_cudart[0]}.{min_cudart[1]} is the minimum required version.'
        warnings.warn(unsupported_ver)
        return _supported_cc
    _supported_cc = ccs_supported_by_ctk(cudart_version)
    return _supported_cc

def find_closest_arch(mycc):
    if False:
        return 10
    '\n    Given a compute capability, return the closest compute capability supported\n    by the CUDA toolkit.\n\n    :param mycc: Compute capability as a tuple ``(MAJOR, MINOR)``\n    :return: Closest supported CC as a tuple ``(MAJOR, MINOR)``\n    '
    supported_ccs = NVVM().supported_ccs
    if not supported_ccs:
        msg = 'No supported GPU compute capabilities found. Please check your cudatoolkit version matches your CUDA version.'
        raise NvvmSupportError(msg)
    for (i, cc) in enumerate(supported_ccs):
        if cc == mycc:
            return cc
        elif cc > mycc:
            if i == 0:
                msg = 'GPU compute capability %d.%d is not supported(requires >=%d.%d)' % (mycc + cc)
                raise NvvmSupportError(msg)
            else:
                return supported_ccs[i - 1]
    return supported_ccs[-1]

def get_arch_option(major, minor):
    if False:
        for i in range(10):
            print('nop')
    'Matches with the closest architecture option\n    '
    if config.FORCE_CUDA_CC:
        arch = config.FORCE_CUDA_CC
    else:
        arch = find_closest_arch((major, minor))
    return 'compute_%d%d' % arch
MISSING_LIBDEVICE_FILE_MSG = 'Missing libdevice file.\nPlease ensure you have package cudatoolkit >= 11.0\nInstall package by:\n\n    conda install cudatoolkit\n'

class LibDevice(object):
    _cache_ = None

    def __init__(self):
        if False:
            i = 10
            return i + 15
        if self._cache_ is None:
            if get_libdevice() is None:
                raise RuntimeError(MISSING_LIBDEVICE_FILE_MSG)
            self._cache_ = open_libdevice()
        self.bc = self._cache_

    def get(self):
        if False:
            for i in range(10):
                print('nop')
        return self.bc
cas_nvvm = '\n    %cas_success = cmpxchg volatile {Ti}* %iptr, {Ti} %old, {Ti} %new monotonic monotonic\n    %cas = extractvalue {{ {Ti}, i1 }} %cas_success, 0\n'
ir_numba_atomic_binary_template = '\ndefine internal {T} @___numba_atomic_{T}_{FUNC}({T}* %ptr, {T} %val) alwaysinline {{\nentry:\n    %iptr = bitcast {T}* %ptr to {Ti}*\n    %old2 = load volatile {Ti}, {Ti}* %iptr\n    br label %attempt\n\nattempt:\n    %old = phi {Ti} [ %old2, %entry ], [ %cas, %attempt ]\n    %dold = bitcast {Ti} %old to {T}\n    %dnew = {OP} {T} %dold, %val\n    %new = bitcast {T} %dnew to {Ti}\n    {CAS}\n    %repeat = icmp ne {Ti} %cas, %old\n    br i1 %repeat, label %attempt, label %done\n\ndone:\n    %result = bitcast {Ti} %old to {T}\n    ret {T} %result\n}}\n'
ir_numba_atomic_inc_template = '\ndefine internal {T} @___numba_atomic_{Tu}_inc({T}* %iptr, {T} %val) alwaysinline {{\nentry:\n    %old2 = load volatile {T}, {T}* %iptr\n    br label %attempt\n\nattempt:\n    %old = phi {T} [ %old2, %entry ], [ %cas, %attempt ]\n    %bndchk = icmp ult {T} %old, %val\n    %inc = add {T} %old, 1\n    %new = select i1 %bndchk, {T} %inc, {T} 0\n    {CAS}\n    %repeat = icmp ne {T} %cas, %old\n    br i1 %repeat, label %attempt, label %done\n\ndone:\n    ret {T} %old\n}}\n'
ir_numba_atomic_dec_template = '\ndefine internal {T} @___numba_atomic_{Tu}_dec({T}* %iptr, {T} %val) alwaysinline {{\nentry:\n    %old2 = load volatile {T}, {T}* %iptr\n    br label %attempt\n\nattempt:\n    %old = phi {T} [ %old2, %entry ], [ %cas, %attempt ]\n    %dec = add {T} %old, -1\n    %bndchk = icmp ult {T} %dec, %val\n    %new = select i1 %bndchk, {T} %dec, {T} %val\n    {CAS}\n    %repeat = icmp ne {T} %cas, %old\n    br i1 %repeat, label %attempt, label %done\n\ndone:\n    ret {T} %old\n}}\n'
ir_numba_atomic_minmax_template = '\ndefine internal {T} @___numba_atomic_{T}_{NAN}{FUNC}({T}* %ptr, {T} %val) alwaysinline {{\nentry:\n    %ptrval = load volatile {T}, {T}* %ptr\n    ; Return early when:\n    ; - For nanmin / nanmax when val is a NaN\n    ; - For min / max when val or ptr is a NaN\n    %early_return = fcmp uno {T} %val, %{PTR_OR_VAL}val\n    br i1 %early_return, label %done, label %lt_check\n\nlt_check:\n    %dold = phi {T} [ %ptrval, %entry ], [ %dcas, %attempt ]\n    ; Continue attempts if dold less or greater than val (depending on whether min or max)\n    ; or if dold is NaN (for nanmin / nanmax)\n    %cmp = fcmp {OP} {T} %dold, %val\n    br i1 %cmp, label %attempt, label %done\n\nattempt:\n    ; Attempt to swap in the value\n    %old = bitcast {T} %dold to {Ti}\n    %iptr = bitcast {T}* %ptr to {Ti}*\n    %new = bitcast {T} %val to {Ti}\n    {CAS}\n    %dcas = bitcast {Ti} %cas to {T}\n    br label %lt_check\n\ndone:\n    ret {T} %ptrval\n}}\n'

def ir_cas(Ti):
    if False:
        while True:
            i = 10
    return cas_nvvm.format(Ti=Ti)

def ir_numba_atomic_binary(T, Ti, OP, FUNC):
    if False:
        for i in range(10):
            print('nop')
    params = dict(T=T, Ti=Ti, OP=OP, FUNC=FUNC, CAS=ir_cas(Ti))
    return ir_numba_atomic_binary_template.format(**params)

def ir_numba_atomic_minmax(T, Ti, NAN, OP, PTR_OR_VAL, FUNC):
    if False:
        i = 10
        return i + 15
    params = dict(T=T, Ti=Ti, NAN=NAN, OP=OP, PTR_OR_VAL=PTR_OR_VAL, FUNC=FUNC, CAS=ir_cas(Ti))
    return ir_numba_atomic_minmax_template.format(**params)

def ir_numba_atomic_inc(T, Tu):
    if False:
        return 10
    return ir_numba_atomic_inc_template.format(T=T, Tu=Tu, CAS=ir_cas(T))

def ir_numba_atomic_dec(T, Tu):
    if False:
        print('Hello World!')
    return ir_numba_atomic_dec_template.format(T=T, Tu=Tu, CAS=ir_cas(T))

def llvm_replace(llvmir):
    if False:
        i = 10
        return i + 15
    replacements = [('declare double @"___numba_atomic_double_add"(double* %".1", double %".2")', ir_numba_atomic_binary(T='double', Ti='i64', OP='fadd', FUNC='add')), ('declare float @"___numba_atomic_float_sub"(float* %".1", float %".2")', ir_numba_atomic_binary(T='float', Ti='i32', OP='fsub', FUNC='sub')), ('declare double @"___numba_atomic_double_sub"(double* %".1", double %".2")', ir_numba_atomic_binary(T='double', Ti='i64', OP='fsub', FUNC='sub')), ('declare i64 @"___numba_atomic_u64_inc"(i64* %".1", i64 %".2")', ir_numba_atomic_inc(T='i64', Tu='u64')), ('declare i64 @"___numba_atomic_u64_dec"(i64* %".1", i64 %".2")', ir_numba_atomic_dec(T='i64', Tu='u64')), ('declare float @"___numba_atomic_float_max"(float* %".1", float %".2")', ir_numba_atomic_minmax(T='float', Ti='i32', NAN='', OP='nnan olt', PTR_OR_VAL='ptr', FUNC='max')), ('declare double @"___numba_atomic_double_max"(double* %".1", double %".2")', ir_numba_atomic_minmax(T='double', Ti='i64', NAN='', OP='nnan olt', PTR_OR_VAL='ptr', FUNC='max')), ('declare float @"___numba_atomic_float_min"(float* %".1", float %".2")', ir_numba_atomic_minmax(T='float', Ti='i32', NAN='', OP='nnan ogt', PTR_OR_VAL='ptr', FUNC='min')), ('declare double @"___numba_atomic_double_min"(double* %".1", double %".2")', ir_numba_atomic_minmax(T='double', Ti='i64', NAN='', OP='nnan ogt', PTR_OR_VAL='ptr', FUNC='min')), ('declare float @"___numba_atomic_float_nanmax"(float* %".1", float %".2")', ir_numba_atomic_minmax(T='float', Ti='i32', NAN='nan', OP='ult', PTR_OR_VAL='', FUNC='max')), ('declare double @"___numba_atomic_double_nanmax"(double* %".1", double %".2")', ir_numba_atomic_minmax(T='double', Ti='i64', NAN='nan', OP='ult', PTR_OR_VAL='', FUNC='max')), ('declare float @"___numba_atomic_float_nanmin"(float* %".1", float %".2")', ir_numba_atomic_minmax(T='float', Ti='i32', NAN='nan', OP='ugt', PTR_OR_VAL='', FUNC='min')), ('declare double @"___numba_atomic_double_nanmin"(double* %".1", double %".2")', ir_numba_atomic_minmax(T='double', Ti='i64', NAN='nan', OP='ugt', PTR_OR_VAL='', FUNC='min')), ('immarg', '')]
    for (decl, fn) in replacements:
        llvmir = llvmir.replace(decl, fn)
    llvmir = llvm140_to_70_ir(llvmir)
    return llvmir

def llvm_to_ptx(llvmir, **opts):
    if False:
        print('Hello World!')
    if isinstance(llvmir, str):
        llvmir = [llvmir]
    if opts.pop('fastmath', False):
        opts.update({'ftz': True, 'fma': True, 'prec_div': False, 'prec_sqrt': False})
    cu = CompilationUnit()
    libdevice = LibDevice()
    for mod in llvmir:
        mod = llvm_replace(mod)
        cu.add_module(mod.encode('utf8'))
    cu.lazy_add_module(libdevice.get())
    return cu.compile(**opts)
re_attributes_def = re.compile('^attributes #\\d+ = \\{ ([\\w\\s]+)\\ }')

def llvm140_to_70_ir(ir):
    if False:
        while True:
            i = 10
    '\n    Convert LLVM 14.0 IR for LLVM 7.0.\n    '
    buf = []
    for line in ir.splitlines():
        if line.startswith('attributes #'):
            m = re_attributes_def.match(line)
            attrs = m.group(1).split()
            attrs = ' '.join((a for a in attrs if a != 'willreturn'))
            line = line.replace(m.group(1), attrs)
        buf.append(line)
    return '\n'.join(buf)

def set_cuda_kernel(lfunc):
    if False:
        i = 10
        return i + 15
    mod = lfunc.module
    mdstr = ir.MetaDataString(mod, 'kernel')
    mdvalue = ir.Constant(ir.IntType(32), 1)
    md = mod.add_metadata((lfunc, mdstr, mdvalue))
    nmd = cgutils.get_or_insert_named_metadata(mod, 'nvvm.annotations')
    nmd.add(md)
    lfunc.attributes.discard('noinline')

def add_ir_version(mod):
    if False:
        i = 10
        return i + 15
    'Add NVVM IR version to module'
    i32 = ir.IntType(32)
    ir_versions = [i32(v) for v in NVVM().get_ir_version()]
    md_ver = mod.add_metadata(ir_versions)
    mod.add_named_metadata('nvvmir.version', md_ver)