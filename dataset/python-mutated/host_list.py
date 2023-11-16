from __future__ import annotations
DOCUMENTATION = '\n    name: host_list\n    version_added: "2.4"\n    short_description: Parses a \'host list\' string\n    description:\n        - Parses a host list string as a comma separated values of hosts\n        - This plugin only applies to inventory strings that are not paths and contain a comma.\n'
EXAMPLES = "\n    # define 2 hosts in command line\n    # ansible -i '10.10.2.6, 10.10.2.4' -m ping all\n\n    # DNS resolvable names\n    # ansible -i 'host1.example.com, host2' -m user -a 'name=me state=absent' all\n\n    # just use localhost\n    # ansible-playbook -i 'localhost,' play.yml -c local\n"
import os
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.parsing.utils.addresses import parse_address
from ansible.plugins.inventory import BaseInventoryPlugin

class InventoryModule(BaseInventoryPlugin):
    NAME = 'host_list'

    def verify_file(self, host_list):
        if False:
            return 10
        valid = False
        b_path = to_bytes(host_list, errors='surrogate_or_strict')
        if not os.path.exists(b_path) and ',' in host_list:
            valid = True
        return valid

    def parse(self, inventory, loader, host_list, cache=True):
        if False:
            for i in range(10):
                print('nop')
        ' parses the inventory file '
        super(InventoryModule, self).parse(inventory, loader, host_list)
        try:
            for h in host_list.split(','):
                h = h.strip()
                if h:
                    try:
                        (host, port) = parse_address(h, allow_ranges=False)
                    except AnsibleError as e:
                        self.display.vvv('Unable to parse address from hostname, leaving unchanged: %s' % to_text(e))
                        host = h
                        port = None
                    if host not in self.inventory.hosts:
                        self.inventory.add_host(host, group='ungrouped', port=port)
        except Exception as e:
            raise AnsibleParserError('Invalid data from string, could not parse: %s' % to_native(e))