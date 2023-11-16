"""
.. module: security_monkey.tests.watchers.test_iam_role
    :platform: Unix
.. version:: $$VERSION$$
.. moduleauthor::  Mike Grima <mgrima@netflix.com>
"""
import json
import boto3
from moto import mock_iam
from moto import mock_sts
from security_monkey.datastore import Account, Technology, ExceptionLogs, AccountType, IgnoreListEntry
from security_monkey.tests import SecurityMonkeyTestCase, db
from security_monkey.watchers.iam.iam_role import IAMRole
from security_monkey import ARN_PREFIX

class IAMRoleTestCase(SecurityMonkeyTestCase):

    def pre_test_setup(self):
        if False:
            print('Hello World!')
        account_type_result = AccountType(name='AWS')
        db.session.add(account_type_result)
        db.session.commit()
        self.account = Account(identifier='012345678910', name='testing', third_party=False, active=True, account_type_id=account_type_result.id)
        self.technology = Technology(name='iamrole')
        self.total_roles = 75
        db.session.add(self.account)
        db.session.add(self.technology)
        db.session.commit()
        mock_iam().start()
        client = boto3.client('iam')
        aspd = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': 'sts:AssumeRole', 'Principal': {'Service': 'ec2.amazonaws.com'}}]}
        policy = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Deny', 'Action': '*', 'Resource': '*'}]}
        for x in range(0, self.total_roles):
            aspd['Statement'][0]['Resource'] = ARN_PREFIX + 'arn:aws:iam:012345678910:role/roleNumber{}'.format(x)
            client.create_role(Path='/', RoleName='roleNumber{}'.format(x), AssumeRolePolicyDocument=json.dumps(aspd, indent=4))
            client.put_role_policy(RoleName='roleNumber{}'.format(x), PolicyName='testpolicy', PolicyDocument=json.dumps(policy, indent=4))

    def test_slurp_list(self):
        if False:
            for i in range(10):
                print('nop')
        mock_sts().start()
        watcher = IAMRole(accounts=[self.account.name])
        (_, exceptions) = watcher.slurp_list()
        assert len(exceptions) == 0
        assert len(watcher.total_list) == self.total_roles
        assert not watcher.done_slurping

    def test_empty_slurp_list(self):
        if False:
            return 10
        mock_sts().start()
        watcher = IAMRole(accounts=[self.account.name])
        watcher.list_method = lambda **kwargs: []
        (_, exceptions) = watcher.slurp_list()
        assert len(exceptions) == 0
        assert len(watcher.total_list) == 0
        assert watcher.done_slurping

    def test_slurp_list_exceptions(self):
        if False:
            print('Hello World!')
        mock_sts().start()
        watcher = IAMRole(accounts=[self.account.name])

        def raise_exception():
            if False:
                while True:
                    i = 10
            raise Exception('LOL, HAY!')
        watcher.list_method = lambda **kwargs: raise_exception()
        (_, exceptions) = watcher.slurp_list()
        assert len(exceptions) == 1
        assert len(ExceptionLogs.query.all()) == 1

    def test_slurp_items(self):
        if False:
            for i in range(10):
                print('nop')
        mock_sts().start()
        watcher = IAMRole(accounts=[self.account.name])
        watcher.batched_size = 10
        watcher.slurp_list()
        (items, exceptions) = watcher.slurp()
        assert len(exceptions) == 0
        assert self.total_roles > len(items) == watcher.batched_size
        assert watcher.batch_counter == 1
        (items, exceptions) = watcher.slurp()
        assert len(exceptions) == 0
        assert self.total_roles > len(items) == watcher.batched_size
        assert watcher.batch_counter == 2

    def test_slurp_items_with_exceptions(self):
        if False:
            print('Hello World!')
        mock_sts().start()
        watcher = IAMRole(accounts=[self.account.name])
        watcher.batched_size = 10
        watcher.slurp_list()

        def raise_exception():
            if False:
                return 10
            raise Exception('LOL, HAY!')
        watcher.get_method = lambda *args, **kwargs: raise_exception()
        (items, exceptions) = watcher.slurp()
        assert len(exceptions) == watcher.batched_size
        assert len(items) == 0
        assert watcher.batch_counter == 1

class IAMRoleSkipTestCase(SecurityMonkeyTestCase):

    def pre_test_setup(self):
        if False:
            return 10
        account_type_result = AccountType(name='AWS')
        db.session.add(account_type_result)
        db.session.commit()
        self.account = Account(identifier='012345678910', name='testing', third_party=False, active=True, account_type_id=account_type_result.id)
        self.technology = Technology(name='iamrole')
        self.total_roles = 10
        db.session.add(self.account)
        db.session.add(self.technology)
        db.session.commit()
        mock_iam().start()
        client = boto3.client('iam')
        aspd = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': 'sts:AssumeRole', 'Principal': {'Service': 'ec2.amazonaws.com'}}]}
        policy = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Deny', 'Action': '*', 'Resource': '*'}]}
        for x in range(0, self.total_roles):
            aspd['Statement'][0]['Resource'] = ARN_PREFIX + 'arn:aws:iam:012345678910:role/roleNumber{}'.format(x)
            client.create_role(Path='/', RoleName='roleNumber{}'.format(x), AssumeRolePolicyDocument=json.dumps(aspd, indent=4))
            client.put_role_policy(RoleName='roleNumber{}'.format(x), PolicyName='testpolicy', PolicyDocument=json.dumps(policy, indent=4))

    def test_slurp_items_with_skipped(self):
        if False:
            for i in range(10):
                print('nop')
        mock_sts().start()
        watcher = IAMRole(accounts=[self.account.name])
        watcher.batched_size = 5
        watcher.slurp_list()
        watcher.ignore_list = [IgnoreListEntry(prefix='roleNumber0'), IgnoreListEntry(prefix='roleNumber1'), IgnoreListEntry(prefix='roleNumber6')]
        (first_batch, exceptions) = watcher.slurp()
        item_sum = len(first_batch)
        assert len(exceptions) == 0
        assert watcher.batch_counter == 1
        batch_lookup = {}
        for r in first_batch:
            assert r.name not in watcher.ignore_list
            batch_lookup[r.name] = True
        (second_batch, exceptions) = watcher.slurp()
        item_sum += len(second_batch)
        assert len(exceptions) == 0
        assert watcher.batch_counter == 2
        for r in second_batch:
            assert r.name not in watcher.ignore_list
            assert not batch_lookup.get(r.name)
            batch_lookup[r.name] = True
        assert self.total_roles - len(watcher.ignore_list) == item_sum == len(batch_lookup)