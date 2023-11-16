import waflib.Tools.asm
from waflib.Tools import ar

def configure(conf):
    if False:
        return 10
    conf.find_program(['gas', 'gcc'], var='AS')
    conf.env.AS_TGT_F = ['-c', '-o']
    conf.env.ASLNK_TGT_F = ['-o']
    conf.find_ar()
    conf.load('asm')
    conf.env.ASM_NAME = 'gas'