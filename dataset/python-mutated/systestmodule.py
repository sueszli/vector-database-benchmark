from __future__ import annotations
import json

def main():
    if False:
        i = 10
        return i + 15
    print(json.dumps(dict(changed=False, source='sys')))
if __name__ == '__main__':
    main()