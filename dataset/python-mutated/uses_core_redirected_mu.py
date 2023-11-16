from __future__ import annotations
import json
import sys
from ansible.module_utils.formerly_core import thingtocall

def main():
    if False:
        while True:
            i = 10
    mu_result = thingtocall()
    print(json.dumps(dict(changed=False, source='user', mu_result=mu_result)))
    sys.exit()
if __name__ == '__main__':
    main()