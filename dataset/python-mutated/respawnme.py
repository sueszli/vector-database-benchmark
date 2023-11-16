from __future__ import annotations
import os
import sys
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.respawn import respawn_module, has_respawned

def main():
    if False:
        i = 10
        return i + 15
    mod = AnsibleModule(argument_spec=dict(mode=dict(required=True, choices=['multi_respawn', 'no_respawn', 'respawn'])))
    if mod.params['mode'] == 'no_respawn':
        mod.exit_json(interpreter_path=sys.executable)
    elif mod.params['mode'] == 'respawn':
        if not has_respawned():
            new_interpreter = os.path.join(mod.tmpdir, 'anotherpython')
            os.symlink(sys.executable, new_interpreter)
            respawn_module(interpreter_path=new_interpreter)
            raise Exception('FAIL, should never reach this line')
        else:
            mod.exit_json(created_interpreter=sys.executable, interpreter_path=sys.executable)
    elif mod.params['mode'] == 'multi_respawn':
        respawn_module(sys.executable)
    raise Exception('FAIL, should never reach this code')
if __name__ == '__main__':
    main()