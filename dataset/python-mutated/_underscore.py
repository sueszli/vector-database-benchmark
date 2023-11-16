from __future__ import annotations
import json

def main():
    if False:
        i = 10
        return i + 15
    print(json.dumps(dict(changed=False, source='legacy_library_dir')))
if __name__ == '__main__':
    main()