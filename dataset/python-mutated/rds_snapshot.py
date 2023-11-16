"""
.. module: security_monkey.auditors.rds.rds_snapshot
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor::  Patrick Kelley <patrick@netflix.com>

"""
from security_monkey.auditor import Auditor, Entity
from security_monkey.watchers.rds.rds_snapshot import RDSSnapshot

class RDSSnapshotAuditor(Auditor):
    index = RDSSnapshot.index
    i_am_singular = RDSSnapshot.i_am_singular
    i_am_plural = RDSSnapshot.i_am_plural

    def __init__(self, accounts=None, debug=False):
        if False:
            i = 10
            return i + 15
        super(RDSSnapshotAuditor, self).__init__(accounts=accounts, debug=debug)

    def prep_for_audit(self):
        if False:
            while True:
                i = 10
        super(RDSSnapshotAuditor, self).prep_for_audit()
        self.FRIENDLY = {account['identifier']: account['name'] for account in self.OBJECT_STORE['ACCOUNTS']['DESCRIPTIONS'] if account['label'] == 'friendly'}
        self.THIRDPARTY = {account['identifier']: account['name'] for account in self.OBJECT_STORE['ACCOUNTS']['DESCRIPTIONS'] if account['label'] == 'thirdparty'}

    def check_internet_accessible(self, item):
        if False:
            return 10
        if 'all' in item.config.get('Attributes', {}).get('restore', []):
            entity = Entity(category='account', value='all')
            self.record_internet_access(item, entity, actions=['restore'])

    def check_friendly_cross_account(self, item):
        if False:
            return 10
        accounts = item.config.get('Attributes', {}).get('restore', [])
        for account in accounts:
            if account == 'all':
                continue
            if account in self.FRIENDLY:
                entity = Entity(category='account', value=account, account_name=self.FRIENDLY[account], account_identifier=account)
                self.record_friendly_access(item, entity, actions=['restore'])

    def check_thirdparty_cross_account(self, item):
        if False:
            while True:
                i = 10
        accounts = item.config.get('Attributes', {}).get('restore', [])
        for account in accounts:
            if account == 'all':
                continue
            if account in self.THIRDPARTY:
                entity = Entity(category='account', value=account, account_name=self.THIRDPARTY[account], account_identifier=account)
                self.record_thirdparty_access(item, entity, actions=['restore'])

    def check_unknown_cross_account(self, item):
        if False:
            return 10
        accounts = item.config.get('Attributes', {}).get('restore', [])
        for account in accounts:
            if account == 'all':
                continue
            if account not in self.FRIENDLY and account not in self.THIRDPARTY:
                entity = Entity(category='account', value=account)
                self.record_unknown_access(item, entity, actions=['restore'])