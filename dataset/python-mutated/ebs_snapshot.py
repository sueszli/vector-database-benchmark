"""
.. module: security_monkey.auditors.ebs_snapshot
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor::  Patrick Kelley <patrick@netflix.com> @monkeysecurity

"""
from security_monkey.auditor import Auditor, Entity
from security_monkey.watchers.ec2.ebs_snapshot import EBSSnapshot

class EBSSnapshotAuditor(Auditor):
    index = EBSSnapshot.index
    i_am_singular = EBSSnapshot.i_am_singular
    i_am_plural = EBSSnapshot.i_am_plural

    def __init__(self, accounts=None, debug=False):
        if False:
            print('Hello World!')
        super(EBSSnapshotAuditor, self).__init__(accounts=accounts, debug=debug)

    def _get_permissions(self, item, key='UserId'):
        if False:
            return 10
        return {perm.get(key) for perm in item.config.get('create_volume_permissions', []) if key in perm}

    def check_friendly_access(self, item):
        if False:
            print('Hello World!')
        for uid in self._get_permissions(item):
            entity = Entity(category='account', value=uid)
            if 'FRIENDLY' in self.inspect_entity(entity, item):
                self.record_friendly_access(item, entity, actions=['createEBSVolume'])

    def check_thirdparty_access(self, item):
        if False:
            for i in range(10):
                print('nop')
        for uid in self._get_permissions(item):
            entity = Entity(category='account', value=uid)
            if 'THIRDPARTY' in self.inspect_entity(entity, item):
                self.record_thirdparty_access(item, entity, actions=['createEBSVolume'])

    def check_unknown_access(self, item):
        if False:
            while True:
                i = 10
        for uid in self._get_permissions(item):
            if 'aws-marketplace' == uid:
                continue
            entity = Entity(category='account', value=uid)
            if 'UNKNOWN' in self.inspect_entity(entity, item):
                self.record_unknown_access(item, entity, actions=['createEBSVolume'])

    def check_marketplace_access(self, item):
        if False:
            print('Hello World!')
        if 'aws-marketplace' in self._get_permissions(item):
            entity = Entity(category='shared_ebs', value='aws-marketplace')
            self.record_internet_access(item, entity, actions=['createEBSVolume'])

    def check_internet_accessible(self, item):
        if False:
            for i in range(10):
                print('nop')
        if 'all' in self._get_permissions(item, key='Group'):
            entity = Entity(category='shared_ebs', value='public')
            self.record_internet_access(item, entity, actions=['createEBSVolume'])