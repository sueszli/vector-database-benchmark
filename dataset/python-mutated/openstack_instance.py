"""
.. module: security_monkey.openstack.watchers.instance
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Michael Stair <mstair@att.com>

"""
from security_monkey.watchers.openstack.openstack_watcher import OpenStackWatcher

class OpenStackInstance(OpenStackWatcher):
    index = 'openstack_instance'
    i_am_singular = 'Instance'
    i_am_plural = 'Instances'
    account_type = 'OpenStack'

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(OpenStackInstance, self).__init__(*args, **kwargs)
        self.item_type = 'instance'
        self.service = 'compute'
        self.generator = 'servers'