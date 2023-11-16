from __future__ import annotations
import json
import sys
from ansible_collections.testns.testcoll.plugins.module_utils.base import thingtocall

def main():
    if False:
        print('Hello World!')
    mu_result = thingtocall()
    print(json.dumps(dict(changed=False, source='user', mu_result=mu_result)))
    sys.exit()
if __name__ == '__main__':
    main()