from __future__ import absolute_import
import os
import jsonschema
import st2common
import st2tests
from st2common.bootstrap.policiesregistrar import PolicyRegistrar
from st2common.bootstrap.policiesregistrar import register_policy_types
from st2common.bootstrap.policiesregistrar import register_policies
import st2common.bootstrap.policiesregistrar as policies_registrar
from st2common.persistence.policy import Policy
from st2common.persistence.policy import PolicyType
from st2tests.base import CleanDbTestCase
from st2tests.fixtures.packs.all_packs_glob import PACKS_PATH
from st2tests.fixtures.packs.dummy_pack_1.fixture import PACK_NAME as DUMMY_PACK_1, PACK_PATH as DUMMY_PACK_1_PATH
from st2tests.fixtures.packs.dummy_pack_2.fixture import PACK_NAME as DUMMY_PACK_2, PACK_PATH as DUMMY_PACK_2_PATH
from st2tests.fixtures.packs.orquesta_tests.fixture import PACK_NAME as ORQUESTA_TESTS
__all__ = ['PoliciesRegistrarTestCase']

class PoliciesRegistrarTestCase(CleanDbTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(PoliciesRegistrarTestCase, self).setUp()
        register_policy_types(st2common)

    def test_register_policy_types(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(register_policy_types(st2tests), 2)
        type1 = PolicyType.get_by_ref('action.concurrency')
        self.assertEqual(type1.name, 'concurrency')
        self.assertEqual(type1.resource_type, 'action')
        type2 = PolicyType.get_by_ref('action.mock_policy_error')
        self.assertEqual(type2.name, 'mock_policy_error')
        self.assertEqual(type2.resource_type, 'action')

    def test_register_all_policies(self):
        if False:
            for i in range(10):
                print('nop')
        policies_dbs = Policy.get_all()
        self.assertEqual(len(policies_dbs), 0)
        count = policies_registrar.register_policies(packs_base_paths=[PACKS_PATH])
        policies_dbs = Policy.get_all()
        policies = {policies_db.name: {'pack': policies_db.pack, 'type': policies_db.policy_type, 'parameters': policies_db.parameters} for policies_db in policies_dbs}
        expected_policies = {'test_policy_1': {'pack': DUMMY_PACK_1, 'type': 'action.concurrency', 'parameters': {'action': 'delay', 'threshold': 3}}, 'test_policy_3': {'pack': DUMMY_PACK_1, 'type': 'action.retry', 'parameters': {'retry_on': 'timeout', 'max_retry_count': 5}}, 'sequential.retry_on_failure': {'pack': ORQUESTA_TESTS, 'type': 'action.retry', 'parameters': {'retry_on': 'failure', 'max_retry_count': 1}}}
        self.assertEqual(len(expected_policies), count)
        self.assertEqual(len(expected_policies), len(policies_dbs))
        self.assertDictEqual(expected_policies, policies)

    def test_register_policies_from_pack(self):
        if False:
            i = 10
            return i + 15
        pack_dir = DUMMY_PACK_1_PATH
        self.assertEqual(register_policies(pack_dir=pack_dir), 2)
        p1 = Policy.get_by_ref('dummy_pack_1.test_policy_1')
        self.assertEqual(p1.name, 'test_policy_1')
        self.assertEqual(p1.pack, DUMMY_PACK_1)
        self.assertEqual(p1.resource_ref, 'dummy_pack_1.local')
        self.assertEqual(p1.policy_type, 'action.concurrency')
        self.assertEqual(p1.parameters['action'], 'delay')
        self.assertEqual(p1.metadata_file, 'policies/policy_1.yaml')
        p2 = Policy.get_by_ref('dummy_pack_1.test_policy_2')
        self.assertEqual(p2, None)

    def test_register_policy_invalid_policy_type_references(self):
        if False:
            while True:
                i = 10
        registrar = PolicyRegistrar()
        policy_path = os.path.join(DUMMY_PACK_1_PATH, 'policies/policy_2.yaml')
        expected_msg = 'Referenced policy_type "action.mock_policy_error" doesnt exist'
        self.assertRaisesRegexp(ValueError, expected_msg, registrar._register_policy, pack=DUMMY_PACK_1, policy=policy_path)

    def test_make_sure_policy_parameters_are_validated_during_register(self):
        if False:
            return 10
        registrar = PolicyRegistrar()
        policy_path = os.path.join(DUMMY_PACK_2_PATH, 'policies/policy_3.yaml')
        expected_msg = '100 is greater than the maximum of 5'
        self.assertRaisesRegexp(jsonschema.ValidationError, expected_msg, registrar._register_policy, pack=DUMMY_PACK_2, policy=policy_path)