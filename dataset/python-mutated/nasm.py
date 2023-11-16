import os
import waflib.Tools.asm
from waflib.TaskGen import feature

@feature('asm')
def apply_nasm_vars(self):
    if False:
        while True:
            i = 10
    self.env.append_value('ASFLAGS', self.to_list(getattr(self, 'nasm_flags', [])))

def configure(conf):
    if False:
        return 10
    conf.find_program(['nasm', 'yasm'], var='AS')
    conf.env.AS_TGT_F = ['-o']
    conf.env.ASLNK_TGT_F = ['-o']
    conf.load('asm')
    conf.env.ASMPATH_ST = '-I%s' + os.sep
    txt = conf.cmd_and_log(conf.env.AS + ['--version'])
    if 'yasm' in txt.lower():
        conf.env.ASM_NAME = 'yasm'
    else:
        conf.env.ASM_NAME = 'nasm'