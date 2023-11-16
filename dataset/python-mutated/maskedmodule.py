from __future__ import annotations
import json

def main():
    if False:
        for i in range(10):
            print('nop')
    print(json.dumps(dict(changed=False, failed=True, msg='this collection should be masked by testcoll in the user content root')))
if __name__ == '__main__':
    main()