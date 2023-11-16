from __future__ import annotations
ANSIBLE_METADATA = {'metadata_version': '1.1', 'status': ['stableinterface'], 'supported_by': 'core'}
DOCUMENTATION = '\n---\nmodule: testmodule2\nshort_description: Test module\ndescription:\n    - Test module\nauthor:\n    - Ansible Core Team\n'
EXAMPLES = '\n'
RETURN = '\n'
import json

def main():
    if False:
        for i in range(10):
            print('nop')
    print(json.dumps(dict(changed=False, source='sys')))
if __name__ == '__main__':
    main()