from __future__ import annotations
import json
DOCUMENTATION = '\nmodule: testmodule\ndescription: for testing\nextends_documentation_fragment:\n  - noncollbogusfrag\n  - noncollbogusfrag.bogusvar\n  - bogusns.testcoll.frag\n  - testns.boguscoll.frag\n  - testns.testcoll.bogusfrag\n  - testns.testcoll.frag.bogusvar\n'

def main():
    if False:
        i = 10
        return i + 15
    print(json.dumps(dict(changed=False, source='user')))
if __name__ == '__main__':
    main()