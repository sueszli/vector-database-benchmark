from __future__ import annotations
import json
import sys
from ansible_collections.testns.testcoll.plugins.module_utils.moved_out_root import importme
from ..module_utils.formerly_testcoll_pkg import thing as movedthing
from ..module_utils.formerly_testcoll_pkg.submod import thing as submodmovedthing

def main():
    if False:
        for i in range(10):
            print('nop')
    mu_result = importme()
    print(json.dumps(dict(changed=False, source='user', mu_result=mu_result, mu_result2=movedthing, mu_result3=submodmovedthing)))
    sys.exit()
if __name__ == '__main__':
    main()