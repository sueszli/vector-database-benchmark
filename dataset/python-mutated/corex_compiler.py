import os
from jittor_utils import env_or_try_find
import jittor_utils
import ctypes
import glob
import jittor.compiler as compiler
has_corex = 0
cc_flags = ''
compiler.has_corex = has_corex

def install():
    if False:
        i = 10
        return i + 15
    import jittor.compiler as compiler
    global has_corex, cc_flags
    acl_compiler_home = os.path.dirname(__file__)
    cc_files = sorted(glob.glob(acl_compiler_home + '/**/*.cc', recursive=True))
    jittor_utils.LOG.i('COREX detected')
    mod = jittor_utils.compile_module('\n#include "common.h"\n#include "utils/str_utils.h"\n\nnamespace jittor {\n// @pyjt(process)\nstring process_acl(const string& src, const string& name, const map<string,string>& kargs) {\n    auto new_src = src;\n    new_src = replace(new_src, "helper_cuda.h", "../inc/helper_cuda.h");\n    if (name == "string_view_map.h")\n        new_src = replace(new_src, "using std::string_view;", "using string_view = string;");\n    if (name == "nan_checker.cu")\n        new_src = replace(new_src, "__trap()", "assert(0)");\n    if (name == "jit_compiler.cc") {\n        // remove asm tuner\n        new_src = token_replace_all(new_src, "cmd = python_path$1;", "");\n        new_src = token_replace_all(new_src, "JPU(op_compiler($1));", \n        R"(JPU(op_compiler($1));\n            *extra_flags2 = replace(*extra_flags2, "--extended-lambda", "");\n            *extra_flags2 = replace(*extra_flags2, "--expt-relaxed-constexpr", "");\n        )");\n        new_src = token_replace_all(new_src, \n            "if (is_cuda_op && $1 != string::npos)",\n            "if (is_cuda_op)");\n    }\n    if (name == "where_op.cc") {\n        // default where kernel cannot handle 64 warp size, use cub_where instead\n        new_src = token_replace_all(new_src, "if (cub_where$1) {", "if (cub_where) {");\n    }\n    if (name == "loop_var_analyze_pass.cc") {\n        new_src = token_replace_all(new_src, "DEFINE_FLAG($1, para_opt_level,$2,$3);", \n                                             "DEFINE_FLAG($1, para_opt_level, 4,$3);");\n    }\n    return new_src;\n}\n}', compiler.cc_flags + ' ' + ' '.join(cc_files) + cc_flags)
    jittor_utils.process_jittor_source('corex', mod.process)
    has_corex = 1
    compiler.has_corex = has_corex
    corex_home = '/usr/local/corex'
    compiler.nvcc_path = corex_home + '/bin/clang++'
    compiler.cc_path = compiler.nvcc_path
    compiler.cc_flags = compiler.cc_flags.replace('-fopenmp', '')
    compiler.nvcc_flags = compiler.cc_flags + ' -x cu -Ofast -DNO_ATOMIC64 -Wno-c++11-narrowing '
    compiler.convert_nvcc_flags = lambda x: x
    compiler.is_cuda = 0
    os.environ['use_cutt'] = '0'
    compiler.cc_type = 'clang'

def install_extern():
    if False:
        for i in range(10):
            print('nop')
    return False

def check():
    if False:
        return 10
    global has_corex, cc_flags
    if os.path.isdir('/usr/local/corex'):
        try:
            install()
        except Exception as e:
            jittor_utils.LOG.w(f'load COREX failed, exception: {e}')
            has_corex = 0
    if not has_corex:
        return False
    return True

def post_process():
    if False:
        i = 10
        return i + 15
    if not has_corex:
        return
    import jittor.compiler as compiler
    compiler.flags.cc_flags = compiler.flags.cc_flags.replace('-fopenmp', '')