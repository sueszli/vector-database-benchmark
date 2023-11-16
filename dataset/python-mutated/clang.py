from waflib.Tools import ccroot, ar, gcc
from waflib.Configure import conf

@conf
def find_clang(conf):
    if False:
        i = 10
        return i + 15
    cc = conf.find_program('clang', var='CC')
    conf.get_cc_version(cc, clang=True)
    conf.env.CC_NAME = 'clang'

def configure(conf):
    if False:
        while True:
            i = 10
    conf.find_clang()
    conf.find_program(['llvm-ar', 'ar'], var='AR')
    conf.find_ar()
    conf.gcc_common_flags()
    conf.gcc_modifier_platform()
    conf.cc_load_tools()
    conf.cc_add_flags()
    conf.link_add_flags()