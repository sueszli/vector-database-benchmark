from __future__ import annotations
DOCUMENTATION = '\n    name: inventory_hostnames\n    author:\n      - Michael DeHaan\n      - Steven Dossett (!UNKNOWN) <sdossett@panath.com>\n    version_added: "1.3"\n    short_description: list of inventory hosts matching a host pattern\n    description:\n      - "This lookup understands \'host patterns\' as used by the C(hosts:) keyword in plays\n        and can return a list of matching hosts from inventory"\n    notes:\n      - this is only worth for \'hostname patterns\' it is easier to loop over the group/group_names variables otherwise.\n'
EXAMPLES = '\n- name: show all the hosts matching the pattern, i.e. all but the group www\n  ansible.builtin.debug:\n    msg: "{{ item }}"\n  with_inventory_hostnames:\n    - all:!www\n'
RETURN = '\n _hostnames:\n    description: list of hostnames that matched the host pattern in inventory\n    type: list\n'
from ansible.errors import AnsibleError
from ansible.inventory.manager import InventoryManager
from ansible.plugins.lookup import LookupBase

class LookupModule(LookupBase):

    def run(self, terms, variables=None, **kwargs):
        if False:
            i = 10
            return i + 15
        manager = InventoryManager(self._loader, parse=False)
        for (group, hosts) in variables['groups'].items():
            manager.add_group(group)
            for host in hosts:
                manager.add_host(host, group=group)
        try:
            return [h.name for h in manager.get_hosts(pattern=terms)]
        except AnsibleError:
            return []