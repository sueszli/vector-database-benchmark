from __future__ import absolute_import, division, print_function
__metaclass__ = type
ANSIBLE_METADATA = {'metadata_version': '1.1', 'status': ['preview'], 'supported_by': 'community'}
DOCUMENTATION = '\n---\nmodule: controller_meta\nauthor: "Alan Rominger (@alancoding)"\nshort_description: Returns metadata about the collection this module lives in.\ndescription:\n    - Allows a user to find out what collection this module exists in.\n    - This takes common module parameters, but does nothing with them.\noptions: {}\nextends_documentation_fragment: awx.awx.auth\n'
RETURN = '\nprefix:\n    description: Collection namespace and name in the namespace.name format\n    returned: success\n    sample: awx.awx\n    type: str\nname:\n    description: Collection name\n    returned: success\n    sample: awx\n    type: str\nnamespace:\n    description: Collection namespace\n    returned: success\n    sample: awx\n    type: str\nversion:\n    description: Version of the collection\n    returned: success\n    sample: 0.0.1-devel\n    type: str\n'
EXAMPLES = '\n- controller_meta:\n  register: result\n\n- name: Show details about the collection\n  debug: var=result\n\n- name: Load the UI setting without hard-coding the collection name\n  debug:\n    msg: "{{ lookup(result.prefix + \'.controller_api\', \'settings/ui\') }}"\n'
from ..module_utils.controller_api import ControllerAPIModule

def main():
    if False:
        print('Hello World!')
    module = ControllerAPIModule(argument_spec={})
    namespace = {'awx': 'awx', 'controller': 'ansible'}.get(module._COLLECTION_TYPE, 'unknown')
    namespace_name = '{0}.{1}'.format(namespace, module._COLLECTION_TYPE)
    module.exit_json(prefix=namespace_name, name=module._COLLECTION_TYPE, namespace=namespace, version=module._COLLECTION_VERSION)
if __name__ == '__main__':
    main()