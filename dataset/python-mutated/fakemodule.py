from __future__ import annotations
DOCUMENTATION = '\n    module: fakemodule\n    short_desciption: fake module\n    description:\n        - this is a fake module\n    version_added: 1.0.0\n    options:\n        _notreal:\n            description:  really not a real option\n    author:\n        - me\n'
import json

def main():
    if False:
        return 10
    print(json.dumps(dict(changed=False, source='testns.testcol.fakemodule')))
if __name__ == '__main__':
    main()