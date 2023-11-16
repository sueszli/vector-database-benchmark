from __future__ import annotations
import os
from ansible.module_utils.basic import AnsibleModule

def main():
    if False:
        i = 10
        return i + 15
    temp = os.path.abspath('temp')
    os.mkdir(temp)
    os.chdir(temp)
    os.rmdir(temp)
    module = AnsibleModule(argument_spec=dict())
    module.exit_json(before=temp, after=os.getcwd())
if __name__ == '__main__':
    main()