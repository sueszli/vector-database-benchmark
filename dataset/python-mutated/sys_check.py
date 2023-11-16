from __future__ import annotations
import sys
from ansible.module_utils.basic import AnsibleModule

def main():
    if False:
        i = 10
        return i + 15
    module = AnsibleModule({})
    this_module = sys.modules[__name__]
    module.exit_json(failed=not getattr(this_module, 'AnsibleModule', False))
if __name__ == '__main__':
    main()