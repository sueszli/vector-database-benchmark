from __future__ import annotations
DOCUMENTATION = '\n    module: fakeslurp\n    short_desciptoin: fake slurp module\n    description:\n        - this is a fake slurp module\n    options:\n        _notreal:\n            description: really not a real slurp\n    author:\n        - me\n'
import json
import random
bad_responses = ['../foo', '../../foo', '../../../foo', '/../../../foo', '/../foo', '//..//foo', '..//..//foo']

def main():
    if False:
        print('Hello World!')
    print(json.dumps(dict(changed=False, content='', encoding='base64', source=random.choice(bad_responses))))
if __name__ == '__main__':
    main()