from waflib.TaskGen import extension
from waflib import Task

@extension('.lua')
def add_lua(self, node):
    if False:
        return 10
    tsk = self.create_task('luac', node, node.change_ext('.luac'))
    inst_to = getattr(self, 'install_path', self.env.LUADIR and '${LUADIR}' or None)
    if inst_to:
        self.add_install_files(install_to=inst_to, install_from=tsk.outputs)
    return tsk

class luac(Task.Task):
    run_str = '${LUAC} -s -o ${TGT} ${SRC}'
    color = 'PINK'

def configure(conf):
    if False:
        return 10
    conf.find_program('luac', var='LUAC')