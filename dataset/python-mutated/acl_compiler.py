import os
from jittor_utils import env_or_try_find
import jittor_utils
import ctypes
import glob
import jittor.compiler as compiler
has_acl = 0
cc_flags = ''
tikcc_path = env_or_try_find('tikcc_path', 'ccec')
dlopen_flags = os.RTLD_NOW | os.RTLD_GLOBAL
compiler.has_acl = has_acl

def install():
    if False:
        print('Hello World!')
    import jittor.compiler as compiler
    global has_acl, cc_flags
    acl_compiler_home = os.path.dirname(__file__)
    cc_files = sorted(glob.glob(acl_compiler_home + '/**/*.cc', recursive=True))
    cc_files2 = []
    for name in cc_files:
        if 'acl_op_exec' in name:
            compiler.extra_core_files.append(name)
        else:
            cc_files2.append(name)
    cc_files = cc_files2
    cc_flags += f' -DHAS_CUDA -DIS_ACL      -I/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/include/     -L/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/lib64     -I{acl_compiler_home} -lascendcl -lacl_op_compiler '
    ctypes.CDLL('libascendcl.so', dlopen_flags)
    '\n    -ltikc_runtime\n    -I/usr/local/Ascend/driver/include     -L/usr/local/Ascend/compiler/lib64     -L/usr/local/Ascend/runtime/lib64     '
    jittor_utils.LOG.i('ACL detected')
    global mod
    mod = jittor_utils.compile_module('\n#include "common.h"\nnamespace jittor {\n// @pyjt(process)\nstring process_acl(const string& src, const string& name, const map<string,string>& kargs);\n// @pyjt(init_acl_ops)\nvoid init_acl_ops();\n}', compiler.cc_flags + ' ' + ' '.join(cc_files) + cc_flags)
    jittor_utils.process_jittor_source('acl', mod.process)
    has_acl = 1
    os.environ['use_mkl'] = '0'
    compiler.setup_fake_cuda_lib = True

def install_extern():
    if False:
        while True:
            i = 10
    return False

def check():
    if False:
        i = 10
        return i + 15
    import jittor.compiler as compiler
    global has_acl, cc_flags
    if tikcc_path:
        try:
            install()
        except Exception as e:
            jittor_utils.LOG.w(f'load ACL failed, exception: {e}')
            has_acl = 0
    compiler.has_acl = has_acl
    compiler.tikcc_path = tikcc_path
    if not has_acl:
        return False
    compiler.cc_flags += cc_flags
    compiler.nvcc_path = tikcc_path
    compiler.nvcc_flags = compiler.cc_flags.replace('-std=c++14', '')
    return True

def post_process():
    if False:
        i = 10
        return i + 15
    if has_acl:
        from jittor import pool
        pool.pool_use_code_op = False
        import jittor as jt
        jt.flags.use_cuda_host_allocator = 1
        jt.flags.use_parallel_op_compiler = 0
        jt.flags.amp_reg |= 32 + 4
        mod.init_acl_ops()