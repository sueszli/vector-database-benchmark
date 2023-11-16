"""
.. module: security_monkey.openstack.watchers.port
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Michael Stair <mstair@att.com>

"""
from security_monkey.watchers.openstack.openstack_watcher import OpenStackWatcher

class OpenStackPort(OpenStackWatcher):
    index = 'openstack_port'
    i_am_singular = 'Port'
    i_am_plural = 'Ports'
    account_type = 'OpenStack'

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(OpenStackPort, self).__init__(*args, **kwargs)
        self.honor_ephemerals = True
        self.item_type = 'port'
        self.service = 'network'
        self.generator = 'ports'
        self.ephemeral_paths = ['updated_at']

    def get_name_from_list_output(self, item):
        if False:
            for i in range(10):
                print('nop')
        return '{} ({})'.format(item.mac_address, item.id)