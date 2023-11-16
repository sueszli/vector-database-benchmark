"""
.. module: security_monkey.tests.auditors.test_elasticsearch_service
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor::  Mike Grima <mgrima@netflix.com>

"""
import json
from security_monkey.datastore import NetworkWhitelistEntry, Account, AccountType
from security_monkey.tests import SecurityMonkeyTestCase
from security_monkey import db
from security_monkey.watchers.elasticsearch_service import ElasticSearchServiceItem
from security_monkey.auditors.elasticsearch_service import ElasticSearchServiceAuditor
CONFIG_ONE = {'name': 'es_test', 'policy': json.loads(b'{\n        "Statement": [\n            {\n                "Action": "es:*",\n                "Effect": "Allow",\n                "Principal": {\n                  "AWS": "*"\n                },\n                "Resource": "arn:aws:es:us-east-1:012345678910:domain/es_test/*",\n                "Sid": ""\n            }\n        ],\n        "Version": "2012-10-17"\n      }\n    ')}
CONFIG_TWO = {'name': 'es_test_2', 'policy': json.loads(b'{\n      "Version": "2012-10-17",\n      "Statement": [\n        {\n          "Sid": "",\n          "Effect": "Allow",\n          "Principal": "*",\n          "Action": "es:*",\n          "Resource": "arn:aws:es:us-west-2:012345678910:domain/es_test_2/*"\n        }\n      ]\n    }\n    ')}
CONFIG_THREE = {'name': 'es_test_3', 'policy': json.loads(b'{\n      "Version": "2012-10-17",\n      "Statement": [\n        {\n          "Sid": "",\n          "Effect": "Allow",\n          "Principal": {\n            "AWS": "arn:aws:iam::012345678910:root"\n          },\n          "Action": "es:*",\n          "Resource": "arn:aws:es:eu-west-1:012345678910:domain/es_test_3/*"\n        },\n        {\n          "Sid": "",\n          "Effect": "Allow",\n          "Principal": "*",\n          "Action": "es:ESHttp*",\n          "Resource": "arn:aws:es:eu-west-1:012345678910:domain/es_test_3/*",\n          "Condition": {\n            "IpAddress": {\n              "aws:SourceIp": [\n                "192.168.1.1/32",\n                "10.0.0.1/8"\n              ]\n            }\n          }\n        }\n      ]\n    }\n    ')}
CONFIG_FOUR = {'name': 'es_test_4', 'policy': json.loads(b'{\n      "Version": "2012-10-17",\n      "Statement": [\n        {\n          "Sid": "",\n          "Effect": "Allow",\n          "Principal": {\n            "AWS": "arn:aws:iam::012345678910:root"\n          },\n          "Action": "es:*",\n          "Resource": "arn:aws:es:us-east-1:012345678910:domain/es_test_4/*"\n        },\n        {\n          "Sid": "",\n          "Effect": "Allow",\n          "Principal": "*",\n          "Action": "es:ESHttp*",\n          "Resource": "arn:aws:es:us-east-1:012345678910:domain/es_test_4/*",\n          "Condition": {\n            "IpAddress": {\n              "aws:SourceIp": [\n                "0.0.0.0/0"\n              ]\n            }\n          }\n        }\n      ]\n    }\n    ')}
CONFIG_FIVE = {'name': 'es_test_5', 'policy': json.loads(b'{\n      "Version": "2012-10-17",\n      "Statement": [\n        {\n          "Sid": "",\n          "Effect": "Allow",\n          "Principal": {\n            "AWS": "arn:aws:iam::012345678910:root"\n          },\n          "Action": "es:*",\n          "Resource": "arn:aws:es:us-east-1:012345678910:domain/es_test_5/*"\n        },\n        {\n          "Sid": "",\n          "Effect": "Deny",\n          "Principal": {\n            "AWS": "arn:aws:iam::012345678910:role/not_this_role"\n          },\n          "Action": "es:*",\n          "Resource": "arn:aws:es:us-east-1:012345678910:domain/es_test_5/*"\n        }\n      ]\n    }\n    ')}
CONFIG_SIX = {'name': 'es_test_6', 'policy': json.loads(b'{\n      "Version": "2012-10-17",\n      "Statement": [\n        {\n          "Sid": "",\n          "Effect": "Allow",\n          "Principal": {\n            "AWS": "arn:aws:iam::012345678910:role/a_good_role"\n          },\n          "Action": "es:*",\n          "Resource": "arn:aws:es:eu-west-1:012345678910:domain/es_test_6/*"\n        },\n        {\n          "Sid": "",\n          "Effect": "Allow",\n          "Principal": "*",\n          "Action": "es:ESHttp*",\n          "Resource": "arn:aws:es:eu-west-1:012345678910:domain/es_test_6/*",\n          "Condition": {\n            "IpAddress": {\n              "aws:SourceIp": [\n                "192.168.1.1/32",\n                "100.0.0.1/16"\n              ]\n            }\n          }\n        }\n      ]\n    }\n    ')}
CONFIG_SEVEN = {'name': 'es_test_7', 'policy': json.loads(b'{\n      "Version": "2012-10-17",\n      "Statement": [\n        {\n          "Sid": "",\n          "Effect": "Allow",\n          "Principal": {\n            "AWS": "arn:aws:iam::012345678910:role/a_good_role"\n          },\n          "Action": "es:*",\n          "Resource": "arn:aws:es:eu-west-1:012345678910:domain/es_test_7/*"\n        },\n        {\n          "Sid": "",\n          "Effect": "Allow",\n          "Principal": "*",\n          "Action": "es:ESHttp*",\n          "Resource": "arn:aws:es:eu-west-1:012345678910:domain/es_test_7/*",\n          "Condition": {\n            "IpAddress": {\n              "aws:SourceIp": [\n                "192.168.1.200/32",\n                "10.0.0.1/8"\n              ]\n            }\n          }\n        }\n      ]\n    }\n    ')}
CONFIG_EIGHT = {'name': 'es_test_8', 'policy': json.loads(b'{\n      "Version": "2012-10-17",\n      "Statement": [\n        {\n          "Sid": "",\n          "Effect": "Allow",\n          "Principal": {\n            "AWS": "*"\n          },\n          "Action": "es:*",\n          "Resource": "arn:aws:es:eu-west-1:012345678910:domain/es_test_8/*"\n        },\n        {\n          "Sid": "",\n          "Effect": "Allow",\n          "Principal": "*",\n          "Action": "es:ESHttp*",\n          "Resource": "arn:aws:es:eu-west-1:012345678910:domain/es_test_8/*",\n          "Condition": {\n            "IpAddress": {\n              "aws:SourceIp": [\n                "192.168.1.1/32",\n                "100.0.0.1/16"\n              ]\n            }\n          }\n        }\n      ]\n    }\n    ')}
CONFIG_NINE = {'name': 'es_test_9', 'policy': json.loads(b'{\n      "Version": "2012-10-17",\n      "Statement": [\n        {\n          "Sid": "",\n          "Effect": "Allow",\n          "Principal": {\n            "AWS": "arn:aws:iam::111111111111:root"\n          },\n          "Action": "es:*",\n          "Resource": "arn:aws:es:us-east-1:012345678910:domain/es_test_9/*"\n        }\n      ]\n    }\n    ')}

class ElasticSearchServiceTestCase(SecurityMonkeyTestCase):

    def pre_test_setup(self):
        if False:
            for i in range(10):
                print('nop')
        ElasticSearchServiceAuditor(accounts=['TEST_ACCOUNT']).OBJECT_STORE.clear()
        self.es_items = [ElasticSearchServiceItem(region='us-east-1', account='TEST_ACCOUNT', name='es_test', config=CONFIG_ONE), ElasticSearchServiceItem(region='us-west-2', account='TEST_ACCOUNT', name='es_test_2', config=CONFIG_TWO), ElasticSearchServiceItem(region='eu-west-1', account='TEST_ACCOUNT', name='es_test_3', config=CONFIG_THREE), ElasticSearchServiceItem(region='us-east-1', account='TEST_ACCOUNT', name='es_test_4', config=CONFIG_FOUR), ElasticSearchServiceItem(region='us-east-1', account='TEST_ACCOUNT', name='es_test_5', config=CONFIG_FIVE), ElasticSearchServiceItem(region='eu-west-1', account='TEST_ACCOUNT', name='es_test_6', config=CONFIG_SIX), ElasticSearchServiceItem(region='eu-west-1', account='TEST_ACCOUNT', name='es_test_7', config=CONFIG_SEVEN), ElasticSearchServiceItem(region='eu-west-1', account='TEST_ACCOUNT', name='es_test_8', config=CONFIG_EIGHT), ElasticSearchServiceItem(region='us-east-1', account='TEST_ACCOUNT', name='es_test_9', config=CONFIG_NINE)]
        account_type_result = AccountType(name='AWS')
        db.session.add(account_type_result)
        db.session.commit()
        account = Account(identifier='012345678910', name='TEST_ACCOUNT', account_type_id=account_type_result.id, notes='TEST_ACCOUNT', third_party=False, active=True)
        db.session.add(account)
        db.session.commit()
        WHITELIST_CIDRS = [('Test one', '192.168.1.1/32'), ('Test two', '100.0.0.0/16')]
        for cidr in WHITELIST_CIDRS:
            whitelist_cidr = NetworkWhitelistEntry()
            whitelist_cidr.name = cidr[0]
            whitelist_cidr.notes = cidr[0]
            whitelist_cidr.cidr = cidr[1]
            db.session.add(whitelist_cidr)
            db.session.commit()

    def test_es_auditor(self):
        if False:
            return 10
        es_auditor = ElasticSearchServiceAuditor(accounts=['012345678910'])
        es_auditor.prep_for_audit()
        for es_domain in self.es_items:
            es_auditor.check_internet_accessible(es_domain)
            es_auditor.check_friendly_cross_account(es_domain)
            es_auditor.check_unknown_cross_account(es_domain)
            es_auditor.check_root_cross_account(es_domain)
        self.assertEqual(len(self.es_items[0].audit_issues), 1)
        self.assertEqual(self.es_items[0].audit_issues[0].score, 10)
        self.assertEqual(len(self.es_items[1].audit_issues), 1)
        self.assertEqual(self.es_items[1].audit_issues[0].score, 10)
        self.assertEqual(len(self.es_items[2].audit_issues), 1)
        self.assertEqual(self.es_items[2].audit_issues[0].score, 10)
        self.assertEqual(len(self.es_items[3].audit_issues), 1)
        self.assertEqual(self.es_items[3].audit_issues[0].score, 10)
        self.assertEqual(len(self.es_items[4].audit_issues), 0)
        self.assertEqual(len(self.es_items[5].audit_issues), 0)
        self.assertEqual(len(self.es_items[6].audit_issues), 2)
        self.assertEqual(self.es_items[6].audit_issues[0].score, 10)
        self.assertEqual(self.es_items[6].audit_issues[1].score, 10)
        self.assertEqual(len(self.es_items[7].audit_issues), 1)
        self.assertEqual(self.es_items[7].audit_issues[0].score, 10)
        self.assertEqual(len(self.es_items[8].audit_issues), 2)
        self.assertEqual(self.es_items[8].audit_issues[0].score, 10)
        self.assertEqual(self.es_items[8].audit_issues[1].score, 6)