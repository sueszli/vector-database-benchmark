"""
.. module: security_monkey.openstack.watchers.router
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Michael Stair <mstair@att.com>

"""
from security_monkey.watchers.openstack.openstack_watcher import OpenStackWatcher

class OpenStackRouter(OpenStackWatcher):
    index = 'openstack_router'
    i_am_singular = 'Router'
    i_am_plural = 'Routers'
    account_type = 'OpenStack'

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(OpenStackRouter, self).__init__(*args, **kwargs)
        self.honor_ephemerals = True
        self.item_type = 'router'
        self.service = 'network'
        self.generator = 'routers'
        self.ephemeral_paths = ['updated_at']