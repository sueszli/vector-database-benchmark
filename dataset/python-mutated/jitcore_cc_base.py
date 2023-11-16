import glob
import os
import tempfile
import platform
import sysconfig
from distutils.sysconfig import get_python_inc
from miasm.jitter.jitcore import JitCore
from miasm.core.utils import keydefaultdict
is_win = platform.system() == 'Windows'

def gen_core(arch, attrib):
    if False:
        print('Hello World!')
    lib_dir = os.path.dirname(os.path.realpath(__file__))
    txt = ''
    txt += '#include "%s/queue.h"\n' % lib_dir
    txt += '#include "%s/op_semantics.h"\n' % lib_dir
    txt += '#include "%s/vm_mngr.h"\n' % lib_dir
    txt += '#include "%s/bn.h"\n' % lib_dir
    txt += '#include "%s/vm_mngr_py.h"\n' % lib_dir
    txt += '#include "%s/JitCore.h"\n' % lib_dir
    txt += '#include "%s/arch/JitCore_%s.h"\n' % (lib_dir, arch.name)
    txt += '\n#define RAISE(errtype, msg) {PyObject* p; p = PyErr_Format( errtype, msg ); return p;}\n'
    return txt

class myresolver(object):

    def __init__(self, offset):
        if False:
            while True:
                i = 10
        self.offset = offset

    def ret(self):
        if False:
            print('Hello World!')
        return 'return PyLong_FromUnsignedLongLong(0x%X);' % self.offset

class resolver(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.resolvers = keydefaultdict(myresolver)

    def get_resolver(self, offset):
        if False:
            for i in range(10):
                print('nop')
        return self.resolvers[offset]

class JitCore_Cc_Base(JitCore):
    """JiT management, abstract class using a C compiler as backend"""

    def __init__(self, lifter, bin_stream):
        if False:
            while True:
                i = 10
        self.jitted_block_delete_cb = self.deleteCB
        super(JitCore_Cc_Base, self).__init__(lifter, bin_stream)
        self.resolver = resolver()
        self.lifter = lifter
        self.states = {}
        self.tempdir = os.path.join(tempfile.gettempdir(), 'miasm_cache')
        try:
            os.mkdir(self.tempdir, 493)
        except OSError:
            pass
        if not os.access(self.tempdir, os.R_OK | os.W_OK):
            raise RuntimeError('Cannot access cache directory %s ' % self.tempdir)
        self.exec_wrapper = None
        self.libs = None
        self.include_files = None

    def deleteCB(self, offset):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def load(self):
        if False:
            for i in range(10):
                print('nop')
        lib_dir = os.path.dirname(os.path.realpath(__file__))
        ext = sysconfig.get_config_var('EXT_SUFFIX')
        if ext is None:
            ext = '.so' if not is_win else '.lib'
        if is_win:
            ext_files = glob.glob(os.path.join(lib_dir, 'VmMngr.*lib'))
            if len(ext_files) == 1:
                ext = os.path.basename(ext_files[0]).replace('VmMngr', '')
        libs = [os.path.join(lib_dir, 'VmMngr' + ext), os.path.join(lib_dir, 'arch', 'JitCore_%s%s' % (self.lifter.arch.name, ext))]
        include_files = [os.path.dirname(__file__), get_python_inc()]
        self.include_files = include_files
        self.libs = libs

    def init_codegen(self, codegen):
        if False:
            print('Hello World!')
        '\n        Get the code generator @codegen\n        @codegen: an CGen instance\n        '
        self.codegen = codegen

    def gen_c_code(self, block):
        if False:
            print('Hello World!')
        '\n        Return the C code corresponding to the @irblocks\n        @irblocks: list of irblocks\n        '
        f_declaration = '_MIASM_EXPORT int %s(block_id * BlockDst, JitCpu* jitcpu)' % self.FUNCNAME
        out = self.codegen.gen_c(block, log_mn=self.log_mn, log_regs=self.log_regs)
        out = [f_declaration + '{'] + out + ['}\n']
        c_code = out
        return self.gen_C_source(self.lifter, c_code)

    @staticmethod
    def gen_C_source(lifter, func_code):
        if False:
            return 10
        raise NotImplementedError()