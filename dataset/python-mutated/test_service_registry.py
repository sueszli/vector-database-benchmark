from st2common.service_setup import register_service_in_service_registry
from st2common.util import system_info
from st2common.services.coordination import get_member_id
from st2common.services import coordination
from st2tests import config as tests_config
from st2tests.api import FunctionalTest
__all__ = ['ServiceyRegistryControllerTestCase']

class ServiceyRegistryControllerTestCase(FunctionalTest):
    coordinator = None

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        super(ServiceyRegistryControllerTestCase, cls).setUpClass()
        tests_config.parse_args(coordinator_noop=True)
        cls.coordinator = coordination.get_coordinator(use_cache=False)
        register_service_in_service_registry(service='mock_service', capabilities={'key1': 'value1', 'name': 'mock_service'}, start_heart=True)

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        super(ServiceyRegistryControllerTestCase, cls).tearDownClass()
        coordination.coordinator_teardown(cls.coordinator)

    def test_get_groups(self):
        if False:
            return 10
        list_resp = self.app.get('/v1/service_registry/groups')
        self.assertEqual(list_resp.status_int, 200)
        self.assertEqual(list_resp.json, {'groups': ['mock_service']})

    def test_get_group_members(self):
        if False:
            i = 10
            return i + 15
        proc_info = system_info.get_process_info()
        member_id = get_member_id()
        resp = self.app.get('/v1/service_registry/groups/doesnt-exist/members', expect_errors=True)
        self.assertEqual(resp.status_int, 404)
        self.assertEqual(resp.json['faultstring'], 'Group with ID "doesnt-exist" not found.')
        resp = self.app.get('/v1/service_registry/groups/mock_service/members')
        self.assertEqual(resp.status_int, 200)
        self.assertEqual(resp.json, {'members': [{'group_id': 'mock_service', 'member_id': member_id.decode('utf-8'), 'capabilities': {'key1': 'value1', 'name': 'mock_service', 'hostname': proc_info['hostname'], 'pid': proc_info['pid']}}]})