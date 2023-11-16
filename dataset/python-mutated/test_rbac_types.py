from __future__ import absolute_import
from unittest2 import TestCase
from st2common.constants.types import ResourceType as SystemType
from st2common.rbac.types import PermissionType
from st2common.rbac.types import ResourceType

class RBACPermissionTypeTestCase(TestCase):

    def test_get_valid_permission_for_resource_type(self):
        if False:
            for i in range(10):
                print('nop')
        valid_action_permissions = PermissionType.get_valid_permissions_for_resource_type(resource_type=ResourceType.ACTION)
        for name in valid_action_permissions:
            self.assertTrue(name.startswith(ResourceType.ACTION + '_'))
        valid_rule_permissions = PermissionType.get_valid_permissions_for_resource_type(resource_type=ResourceType.RULE)
        for name in valid_rule_permissions:
            self.assertTrue(name.startswith(ResourceType.RULE + '_'))

    def test_get_resource_type(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(PermissionType.get_resource_type(PermissionType.PACK_LIST), SystemType.PACK)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.PACK_VIEW), SystemType.PACK)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.PACK_CREATE), SystemType.PACK)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.PACK_MODIFY), SystemType.PACK)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.PACK_DELETE), SystemType.PACK)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.PACK_ALL), SystemType.PACK)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.SENSOR_LIST), SystemType.SENSOR_TYPE)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.SENSOR_VIEW), SystemType.SENSOR_TYPE)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.SENSOR_MODIFY), SystemType.SENSOR_TYPE)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.SENSOR_ALL), SystemType.SENSOR_TYPE)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.ACTION_LIST), SystemType.ACTION)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.ACTION_VIEW), SystemType.ACTION)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.ACTION_CREATE), SystemType.ACTION)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.ACTION_MODIFY), SystemType.ACTION)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.ACTION_DELETE), SystemType.ACTION)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.ACTION_EXECUTE), SystemType.ACTION)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.ACTION_ALL), SystemType.ACTION)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.EXECUTION_LIST), SystemType.EXECUTION)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.EXECUTION_VIEW), SystemType.EXECUTION)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.EXECUTION_RE_RUN), SystemType.EXECUTION)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.EXECUTION_STOP), SystemType.EXECUTION)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.EXECUTION_ALL), SystemType.EXECUTION)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.RULE_LIST), SystemType.RULE)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.RULE_VIEW), SystemType.RULE)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.RULE_CREATE), SystemType.RULE)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.RULE_MODIFY), SystemType.RULE)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.RULE_DELETE), SystemType.RULE)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.RULE_ALL), SystemType.RULE)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.RULE_ENFORCEMENT_LIST), SystemType.RULE_ENFORCEMENT)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.RULE_ENFORCEMENT_VIEW), SystemType.RULE_ENFORCEMENT)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.KEY_VALUE_PAIR_LIST), SystemType.KEY_VALUE_PAIR)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.KEY_VALUE_PAIR_VIEW), SystemType.KEY_VALUE_PAIR)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.KEY_VALUE_PAIR_SET), SystemType.KEY_VALUE_PAIR)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.KEY_VALUE_PAIR_DELETE), SystemType.KEY_VALUE_PAIR)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.KEY_VALUE_PAIR_ALL), SystemType.KEY_VALUE_PAIR)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.WEBHOOK_CREATE), SystemType.WEBHOOK)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.WEBHOOK_SEND), SystemType.WEBHOOK)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.WEBHOOK_DELETE), SystemType.WEBHOOK)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.WEBHOOK_ALL), SystemType.WEBHOOK)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.API_KEY_LIST), SystemType.API_KEY)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.API_KEY_VIEW), SystemType.API_KEY)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.API_KEY_CREATE), SystemType.API_KEY)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.API_KEY_DELETE), SystemType.API_KEY)
        self.assertEqual(PermissionType.get_resource_type(PermissionType.API_KEY_ALL), SystemType.API_KEY)

    def test_get_permission_type(self):
        if False:
            print('Hello World!')
        self.assertEqual(PermissionType.get_permission_type(resource_type=ResourceType.ACTION, permission_name='view'), PermissionType.ACTION_VIEW)
        self.assertEqual(PermissionType.get_permission_type(resource_type=ResourceType.ACTION, permission_name='all'), PermissionType.ACTION_ALL)
        self.assertEqual(PermissionType.get_permission_type(resource_type=ResourceType.ACTION, permission_name='execute'), PermissionType.ACTION_EXECUTE)
        self.assertEqual(PermissionType.get_permission_type(resource_type=ResourceType.RULE, permission_name='view'), PermissionType.RULE_VIEW)
        self.assertEqual(PermissionType.get_permission_type(resource_type=ResourceType.RULE, permission_name='delete'), PermissionType.RULE_DELETE)
        self.assertEqual(PermissionType.get_permission_type(resource_type=ResourceType.SENSOR, permission_name='view'), PermissionType.SENSOR_VIEW)
        self.assertEqual(PermissionType.get_permission_type(resource_type=ResourceType.SENSOR, permission_name='all'), PermissionType.SENSOR_ALL)
        self.assertEqual(PermissionType.get_permission_type(resource_type=ResourceType.SENSOR, permission_name='modify'), PermissionType.SENSOR_MODIFY)
        self.assertEqual(PermissionType.get_permission_type(resource_type=ResourceType.RULE_ENFORCEMENT, permission_name='view'), PermissionType.RULE_ENFORCEMENT_VIEW)
        t = ResourceType.KEY_VALUE_PAIR
        self.assertEqual(PermissionType.get_permission_type(resource_type=t, permission_name='list'), PermissionType.KEY_VALUE_PAIR_LIST)
        self.assertEqual(PermissionType.get_permission_type(resource_type=t, permission_name='view'), PermissionType.KEY_VALUE_PAIR_VIEW)
        self.assertEqual(PermissionType.get_permission_type(resource_type=t, permission_name='set'), PermissionType.KEY_VALUE_PAIR_SET)
        self.assertEqual(PermissionType.get_permission_type(resource_type=t, permission_name='delete'), PermissionType.KEY_VALUE_PAIR_DELETE)
        self.assertEqual(PermissionType.get_permission_type(resource_type=t, permission_name='all'), PermissionType.KEY_VALUE_PAIR_ALL)

    def test_get_permission_name(self):
        if False:
            while True:
                i = 10
        self.assertEqual(PermissionType.get_permission_name(PermissionType.ACTION_LIST), 'list')
        self.assertEqual(PermissionType.get_permission_name(PermissionType.ACTION_CREATE), 'create')
        self.assertEqual(PermissionType.get_permission_name(PermissionType.ACTION_DELETE), 'delete')
        self.assertEqual(PermissionType.get_permission_name(PermissionType.ACTION_ALL), 'all')
        self.assertEqual(PermissionType.get_permission_name(PermissionType.PACK_ALL), 'all')
        self.assertEqual(PermissionType.get_permission_name(PermissionType.SENSOR_MODIFY), 'modify')
        self.assertEqual(PermissionType.get_permission_name(PermissionType.ACTION_EXECUTE), 'execute')
        self.assertEqual(PermissionType.get_permission_name(PermissionType.RULE_ENFORCEMENT_LIST), 'list')
        self.assertEqual(PermissionType.get_permission_name(PermissionType.KEY_VALUE_PAIR_LIST), 'list')
        self.assertEqual(PermissionType.get_permission_name(PermissionType.KEY_VALUE_PAIR_VIEW), 'view')
        self.assertEqual(PermissionType.get_permission_name(PermissionType.KEY_VALUE_PAIR_SET), 'set')
        self.assertEqual(PermissionType.get_permission_name(PermissionType.KEY_VALUE_PAIR_DELETE), 'delete')
        self.assertEqual(PermissionType.get_permission_name(PermissionType.KEY_VALUE_PAIR_ALL), 'all')