import json
from django.test import TestCase
from rest_framework.test import APIClient
from api.date_time_extractor import DEFAULT_RULES_PARAMS, OTHER_RULES_PARAMS
from api.models import User

class PredefinedRulesTest(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.admin = User.objects.create_superuser('test_admin', 'test_admin@test.com', 'test_password')
        self.client = APIClient()
        self.client.force_authenticate(user=self.admin)

    def test_predefined_rules(self):
        if False:
            print('Hello World!')
        response = self.client.get('/api/predefinedrules/')
        self.assertEqual(200, response.status_code)
        data = response.json()
        self.assertIsInstance(data, str)
        rules = json.loads(data)
        self.assertIsInstance(rules, list)
        self.assertEqual(15, len(rules))

    def test_default_rules_on_predefined_rules_endpoint(self):
        if False:
            while True:
                i = 10
        response = self.client.get('/api/predefinedrules/')
        rules = json.loads(response.json())
        default_rules = list(filter(lambda x: x['is_default'], rules))
        self.assertListEqual(DEFAULT_RULES_PARAMS, default_rules)

    def test_default_rules_endpoint(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get('/api/defaultrules/')
        rules = json.loads(response.json())
        self.assertListEqual(DEFAULT_RULES_PARAMS, rules)

    def test_other_rules(self):
        if False:
            print('Hello World!')
        response = self.client.get('/api/predefinedrules/')
        rules = json.loads(response.json())
        other_rules = list(filter(lambda x: not x['is_default'], rules))
        self.assertListEqual(OTHER_RULES_PARAMS, other_rules)