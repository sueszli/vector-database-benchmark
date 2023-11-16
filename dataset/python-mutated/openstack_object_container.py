"""
.. module: security_monkey.openstack.auditors.object_container
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Michael Stair <mstair@att.com>

"""
from security_monkey.auditor import Auditor
from security_monkey.watchers.openstack.object_store.openstack_object_container import OpenStackObjectContainer

class OpenStackObjectContainerAuditor(Auditor):
    index = OpenStackObjectContainer.index
    i_am_singular = OpenStackObjectContainer.i_am_singular
    i_am_plural = OpenStackObjectContainer.i_am_plural

    def __init__(self, accounts=None, debug=False):
        if False:
            for i in range(10):
                print('nop')
        super(OpenStackObjectContainerAuditor, self).__init__(accounts=accounts, debug=debug)

    def check_acls(self, container_item):
        if False:
            while True:
                i = 10
        read_acl = container_item.config.get('read_ACL')
        write_acl = container_item.config.get('write_ACL')
        if read_acl:
            for acl in read_acl.split(','):
                if acl == '.r:*':
                    message = 'ACL - World Readable'
                    self.add_issue(30, message, container_item)
                elif acl == '.rlistings':
                    message = 'ACL - World Listable'
                    self.add_issue(10, message, container_item)
        if write_acl:
            for acl in write_acl.split(','):
                if acl == '*:*':
                    message = 'ACL - World Writable'
                    self.add_issue(20, message, container_item)