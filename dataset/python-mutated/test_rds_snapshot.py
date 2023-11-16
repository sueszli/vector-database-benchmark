"""
.. module: security_monkey.tests.auditors.rds.test_rds_snapshot
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Patrick Kelley <pkelley@netflix.com> @monkeysecurity

"""
from security_monkey.tests import SecurityMonkeyTestCase
from security_monkey.auditors.rds.rds_snapshot import RDSSnapshotAuditor
from security_monkey.watchers.rds.rds_snapshot import RDSSnapshot, RDSSnapshotItem
from security_monkey.datastore import Account, AccountType
from security_monkey import db

class RDSSnapshotAuditorTestCase(SecurityMonkeyTestCase):

    def pre_test_setup(self):
        if False:
            for i in range(10):
                print('nop')
        RDSSnapshotAuditor(accounts=['TEST_ACCOUNT']).OBJECT_STORE.clear()
        account_type_result = AccountType(name='AWS')
        db.session.add(account_type_result)
        db.session.commit()
        account = Account(identifier='123456789123', name='TEST_ACCOUNT', account_type_id=account_type_result.id, notes='TEST_ACCOUNT', third_party=False, active=True)
        account2 = Account(identifier='222222222222', name='TEST_ACCOUNT_TWO', account_type_id=account_type_result.id, notes='TEST_ACCOUNT_TWO', third_party=False, active=True)
        account3 = Account(identifier='333333333333', name='TEST_ACCOUNT_THREE', account_type_id=account_type_result.id, notes='TEST_ACCOUNT_THREE', third_party=True, active=True)
        db.session.add(account)
        db.session.add(account2)
        db.session.add(account3)
        db.session.commit()

    def test_check_internet_accessible(self):
        if False:
            return 10
        config0 = {'Attributes': {'restore': ['all']}}
        config1 = {'Attributes': {'restore': []}}
        rsa = RDSSnapshotAuditor(accounts=['TEST_ACCOUNT'])
        rsa.prep_for_audit()
        item = RDSSnapshotItem(config=config0)
        rsa.check_internet_accessible(item)
        self.assertEqual(len(item.audit_issues), 1)
        self.assertEqual(item.audit_issues[0].score, 10)
        self.assertEqual(item.audit_issues[0].issue, 'Internet Accessible')
        self.assertEqual(item.audit_issues[0].notes, 'Entity: [account:all] Actions: ["restore"]')
        item = RDSSnapshotItem(config=config1)
        rsa.check_internet_accessible(item)
        self.assertEqual(len(item.audit_issues), 0)

    def test_check_friendly(self):
        if False:
            return 10
        config0 = {'Attributes': {'restore': ['222222222222']}}
        rsa = RDSSnapshotAuditor(accounts=['TEST_ACCOUNT'])
        rsa.prep_for_audit()
        item = RDSSnapshotItem(config=config0)
        rsa.check_friendly_cross_account(item)
        self.assertEqual(len(item.audit_issues), 1)
        self.assertEqual(item.audit_issues[0].score, 0)
        self.assertEqual(item.audit_issues[0].issue, 'Friendly Cross Account')
        self.assertEqual(item.audit_issues[0].notes, 'Account: [222222222222/TEST_ACCOUNT_TWO] Entity: [account:222222222222] Actions: ["restore"]')

    def test_check_thirdparty(self):
        if False:
            while True:
                i = 10
        config0 = {'Attributes': {'restore': ['333333333333']}}
        rsa = RDSSnapshotAuditor(accounts=['TEST_ACCOUNT'])
        rsa.prep_for_audit()
        item = RDSSnapshotItem(config=config0)
        rsa.check_thirdparty_cross_account(item)
        self.assertEqual(len(item.audit_issues), 1)
        self.assertEqual(item.audit_issues[0].score, 0)
        self.assertEqual(item.audit_issues[0].issue, 'Thirdparty Cross Account')
        self.assertEqual(item.audit_issues[0].notes, 'Account: [333333333333/TEST_ACCOUNT_THREE] Entity: [account:333333333333] Actions: ["restore"]')

    def test_check_unknown(self):
        if False:
            for i in range(10):
                print('nop')
        config0 = {'Attributes': {'restore': ['444444444444']}}
        rsa = RDSSnapshotAuditor(accounts=['TEST_ACCOUNT'])
        rsa.prep_for_audit()
        item = RDSSnapshotItem(config=config0)
        rsa.check_unknown_cross_account(item)
        self.assertEqual(len(item.audit_issues), 1)
        self.assertEqual(item.audit_issues[0].score, 10)
        self.assertEqual(item.audit_issues[0].issue, 'Unknown Access')
        self.assertEqual(item.audit_issues[0].notes, 'Entity: [account:444444444444] Actions: ["restore"]')