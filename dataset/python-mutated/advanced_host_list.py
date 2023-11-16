from __future__ import annotations
DOCUMENTATION = '\n    name: advanced_host_list\n    version_added: "2.4"\n    short_description: Parses a \'host list\' with ranges\n    description:\n        - Parses a host list string as a comma separated values of hosts and supports host ranges.\n        - This plugin only applies to inventory sources that are not paths and contain at least one comma.\n'
EXAMPLES = "\n    # simple range\n    # ansible -i 'host[1:10],' -m ping\n\n    # still supports w/o ranges also\n    # ansible-playbook -i 'localhost,' play.yml\n"
import os
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.plugins.inventory import BaseInventoryPlugin

class InventoryModule(BaseInventoryPlugin):
    NAME = 'advanced_host_list'

    def verify_file(self, host_list):
        if False:
            i = 10
            return i + 15
        valid = False
        b_path = to_bytes(host_list, errors='surrogate_or_strict')
        if not os.path.exists(b_path) and ',' in host_list:
            valid = True
        return valid

    def parse(self, inventory, loader, host_list, cache=True):
        if False:
            while True:
                i = 10
        ' parses the inventory file '
        super(InventoryModule, self).parse(inventory, loader, host_list)
        try:
            for h in host_list.split(','):
                h = h.strip()
                if h:
                    try:
                        (hostnames, port) = self._expand_hostpattern(h)
                    except AnsibleError as e:
                        self.display.vvv('Unable to parse address from hostname, leaving unchanged: %s' % to_text(e))
                        hostnames = [h]
                        port = None
                    for host in hostnames:
                        if host not in self.inventory.hosts:
                            self.inventory.add_host(host, group='ungrouped', port=port)
        except Exception as e:
            raise AnsibleParserError('Invalid data from string, could not parse: %s' % to_native(e))