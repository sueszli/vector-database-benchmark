from waflib import Task
from waflib.Configure import conf
from waflib.TaskGen import feature, before_method, after_method
LIB_CODE = '\n#ifdef _MSC_VER\n#define testEXPORT __declspec(dllexport)\n#else\n#define testEXPORT\n#endif\ntestEXPORT int lib_func(void) { return 9; }\n'
MAIN_CODE = '\n#ifdef _MSC_VER\n#define testEXPORT __declspec(dllimport)\n#else\n#define testEXPORT\n#endif\ntestEXPORT int lib_func(void);\nint main(int argc, char **argv) {\n\t(void)argc; (void)argv;\n\treturn !(lib_func() == 9);\n}\n'

@feature('link_lib_test')
@before_method('process_source')
def link_lib_test_fun(self):
    if False:
        print('Hello World!')

    def write_test_file(task):
        if False:
            return 10
        task.outputs[0].write(task.generator.code)
    rpath = []
    if getattr(self, 'add_rpath', False):
        rpath = [self.bld.path.get_bld().abspath()]
    mode = self.mode
    m = '%s %s' % (mode, mode)
    ex = self.test_exec and 'test_exec' or ''
    bld = self.bld
    bld(rule=write_test_file, target='test.' + mode, code=LIB_CODE)
    bld(rule=write_test_file, target='main.' + mode, code=MAIN_CODE)
    bld(features='%sshlib' % m, source='test.' + mode, target='test')
    bld(features='%sprogram %s' % (m, ex), source='main.' + mode, target='app', use='test', rpath=rpath)

@conf
def check_library(self, mode=None, test_exec=True):
    if False:
        while True:
            i = 10
    if not mode:
        mode = 'c'
        if self.env.CXX:
            mode = 'cxx'
    self.check(compile_filename=[], features='link_lib_test', msg='Checking for libraries', mode=mode, test_exec=test_exec)
INLINE_CODE = '\ntypedef int foo_t;\nstatic %s foo_t static_foo () {return 0; }\n%s foo_t foo () {\n\treturn 0;\n}\n'
INLINE_VALUES = ['inline', '__inline__', '__inline']

@conf
def check_inline(self, **kw):
    if False:
        for i in range(10):
            print('nop')
    self.start_msg('Checking for inline')
    if not 'define_name' in kw:
        kw['define_name'] = 'INLINE_MACRO'
    if not 'features' in kw:
        if self.env.CXX:
            kw['features'] = ['cxx']
        else:
            kw['features'] = ['c']
    for x in INLINE_VALUES:
        kw['fragment'] = INLINE_CODE % (x, x)
        try:
            self.check(**kw)
        except self.errors.ConfigurationError:
            continue
        else:
            self.end_msg(x)
            if x != 'inline':
                self.define('inline', x, quote=False)
            return x
    self.fatal('could not use inline functions')
LARGE_FRAGMENT = '#include <unistd.h>\nint main(int argc, char **argv) {\n\t(void)argc; (void)argv;\n\treturn !(sizeof(off_t) >= 8);\n}\n'

@conf
def check_large_file(self, **kw):
    if False:
        for i in range(10):
            print('nop')
    if not 'define_name' in kw:
        kw['define_name'] = 'HAVE_LARGEFILE'
    if not 'execute' in kw:
        kw['execute'] = True
    if not 'features' in kw:
        if self.env.CXX:
            kw['features'] = ['cxx', 'cxxprogram']
        else:
            kw['features'] = ['c', 'cprogram']
    kw['fragment'] = LARGE_FRAGMENT
    kw['msg'] = 'Checking for large file support'
    ret = True
    try:
        if self.env.DEST_BINFMT != 'pe':
            ret = self.check(**kw)
    except self.errors.ConfigurationError:
        pass
    else:
        if ret:
            return True
    kw['msg'] = 'Checking for -D_FILE_OFFSET_BITS=64'
    kw['defines'] = ['_FILE_OFFSET_BITS=64']
    try:
        ret = self.check(**kw)
    except self.errors.ConfigurationError:
        pass
    else:
        self.define('_FILE_OFFSET_BITS', 64)
        return ret
    self.fatal('There is no support for large files')
ENDIAN_FRAGMENT = '\n#ifdef _MSC_VER\n#define testshlib_EXPORT __declspec(dllexport)\n#else\n#define testshlib_EXPORT\n#endif\n\nshort int ascii_mm[] = { 0x4249, 0x4765, 0x6E44, 0x6961, 0x6E53, 0x7953, 0 };\nshort int ascii_ii[] = { 0x694C, 0x5454, 0x656C, 0x6E45, 0x6944, 0x6E61, 0 };\nint testshlib_EXPORT use_ascii (int i) {\n\treturn ascii_mm[i] + ascii_ii[i];\n}\nshort int ebcdic_ii[] = { 0x89D3, 0xE3E3, 0x8593, 0x95C5, 0x89C4, 0x9581, 0 };\nshort int ebcdic_mm[] = { 0xC2C9, 0xC785, 0x95C4, 0x8981, 0x95E2, 0xA8E2, 0 };\nint use_ebcdic (int i) {\n\treturn ebcdic_mm[i] + ebcdic_ii[i];\n}\nextern int foo;\n'

class grep_for_endianness(Task.Task):
    color = 'PINK'

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        txt = self.inputs[0].read(flags='rb').decode('latin-1')
        if txt.find('LiTTleEnDian') > -1:
            self.generator.tmp.append('little')
        elif txt.find('BIGenDianSyS') > -1:
            self.generator.tmp.append('big')
        else:
            return -1

@feature('grep_for_endianness')
@after_method('apply_link')
def grep_for_endianness_fun(self):
    if False:
        for i in range(10):
            print('nop')
    self.create_task('grep_for_endianness', self.link_task.outputs[0])

@conf
def check_endianness(self):
    if False:
        while True:
            i = 10
    tmp = []

    def check_msg(self):
        if False:
            for i in range(10):
                print('nop')
        return tmp[0]
    self.check(fragment=ENDIAN_FRAGMENT, features='c cshlib grep_for_endianness', msg='Checking for endianness', define='ENDIANNESS', tmp=tmp, okmsg=check_msg, confcache=None)
    return tmp[0]