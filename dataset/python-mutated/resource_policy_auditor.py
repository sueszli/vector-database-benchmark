"""
.. module: security_monkey.auditors.resource_policy_auditor
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Patrick Kelley <patrick@netflix.com>

"""
from security_monkey import app
from security_monkey.auditor import Auditor, Entity
from security_monkey.datastore import Account
from policyuniverse.arn import ARN
from policyuniverse.policy import Policy
from policyuniverse.statement import Statement
import json
import ipaddr

class ResourcePolicyAuditor(Auditor):

    def __init__(self, accounts=None, debug=False):
        if False:
            while True:
                i = 10
        super(ResourcePolicyAuditor, self).__init__(accounts=accounts, debug=debug)
        self.policy_keys = ['Policy']

    def load_resource_policies(self, item):
        if False:
            while True:
                i = 10
        return self.load_policies(item, self.policy_keys)

    def check_internet_accessible(self, item):
        if False:
            i = 10
            return i + 15
        policies = self.load_resource_policies(item)
        for policy in policies:
            if policy.is_internet_accessible():
                entity = Entity(category='principal', value='*')
                actions = list(policy.internet_accessible_actions())
                self.record_internet_access(item, entity, actions)

    def check_friendly_cross_account(self, item):
        if False:
            i = 10
            return i + 15
        policies = self.load_resource_policies(item)
        for policy in policies:
            for statement in policy.statements:
                if statement.effect != 'Allow':
                    continue
                for who in statement.whos_allowed():
                    entity = Entity.from_tuple(who)
                    if 'FRIENDLY' in self.inspect_entity(entity, item):
                        self.record_friendly_access(item, entity, list(statement.actions))

    def check_thirdparty_cross_account(self, item):
        if False:
            i = 10
            return i + 15
        policies = self.load_resource_policies(item)
        for policy in policies:
            for statement in policy.statements:
                if statement.effect != 'Allow':
                    continue
                for who in statement.whos_allowed():
                    entity = Entity.from_tuple(who)
                    if 'THIRDPARTY' in self.inspect_entity(entity, item):
                        self.record_thirdparty_access(item, entity, list(statement.actions))

    def check_unknown_cross_account(self, item):
        if False:
            i = 10
            return i + 15
        policies = self.load_resource_policies(item)
        for policy in policies:
            if policy.is_internet_accessible():
                continue
            for statement in policy.statements:
                if statement.effect != 'Allow':
                    continue
                for who in statement.whos_allowed():
                    if who.value == '*' and who.category == 'principal':
                        continue
                    if who.category == 'principal':
                        arn = ARN(who.value)
                        if arn.service:
                            continue
                    entity = Entity.from_tuple(who)
                    if 'UNKNOWN' in self.inspect_entity(entity, item):
                        self.record_unknown_access(item, entity, list(statement.actions))

    def check_root_cross_account(self, item):
        if False:
            i = 10
            return i + 15
        policies = self.load_resource_policies(item)
        for policy in policies:
            for statement in policy.statements:
                if statement.effect != 'Allow':
                    continue
                for who in statement.whos_allowed():
                    if who.category not in ['arn', 'principal']:
                        continue
                    if who.value == '*':
                        continue
                    arn = ARN(who.value)
                    entity = Entity.from_tuple(who)
                    if arn.root and self.inspect_entity(entity, item).intersection(set(['FRIENDLY', 'THIRDPARTY', 'UNKNOWN'])):
                        self.record_cross_account_root(item, entity, list(statement.actions))