import sys
from waflib.Tools import ccroot, ar, gcc
from waflib.Configure import conf

@conf
def find_icc(conf):
    if False:
        return 10
    cc = conf.find_program(['icc', 'ICL'], var='CC')
    conf.get_cc_version(cc, icc=True)
    conf.env.CC_NAME = 'icc'

def configure(conf):
    if False:
        i = 10
        return i + 15
    conf.find_icc()
    conf.find_ar()
    conf.gcc_common_flags()
    conf.gcc_modifier_platform()
    conf.cc_load_tools()
    conf.cc_add_flags()
    conf.link_add_flags()