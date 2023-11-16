from __future__ import annotations
from unittest.mock import Mock
import pytest
from airflow.exceptions import AirflowException
from airflow.providers.amazon.aws.hooks.elasticache_replication_group import ElastiCacheReplicationGroupHook

class TestElastiCacheReplicationGroupHook:
    REPLICATION_GROUP_ID = 'test-elasticache-replication-group-hook'
    REPLICATION_GROUP_CONFIG = {'ReplicationGroupId': REPLICATION_GROUP_ID, 'ReplicationGroupDescription': REPLICATION_GROUP_ID, 'AutomaticFailoverEnabled': False, 'NumCacheClusters': 1, 'CacheNodeType': 'cache.m5.large', 'Engine': 'redis', 'EngineVersion': '5.0.4', 'CacheParameterGroupName': 'default.redis5.0'}
    VALID_STATES = frozenset({'creating', 'available', 'modifying', 'deleting', 'create - failed', 'snapshotting'})

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        self.hook = ElastiCacheReplicationGroupHook()
        self.hook.conn = Mock()
        self.hook.conn.create_replication_group.return_value = {'ReplicationGroup': {'ReplicationGroupId': self.REPLICATION_GROUP_ID, 'Status': 'creating'}}

    def _create_replication_group(self):
        if False:
            while True:
                i = 10
        return self.hook.create_replication_group(config=self.REPLICATION_GROUP_CONFIG)

    def test_conn_not_none(self):
        if False:
            print('Hello World!')
        assert self.hook.conn is not None

    def test_create_replication_group(self):
        if False:
            i = 10
            return i + 15
        response = self._create_replication_group()
        assert response['ReplicationGroup']['ReplicationGroupId'] == self.REPLICATION_GROUP_ID
        assert response['ReplicationGroup']['Status'] == 'creating'

    def test_describe_replication_group(self):
        if False:
            for i in range(10):
                print('nop')
        self._create_replication_group()
        self.hook.conn.describe_replication_groups.return_value = {'ReplicationGroups': [{'ReplicationGroupId': self.REPLICATION_GROUP_ID}]}
        response = self.hook.describe_replication_group(replication_group_id=self.REPLICATION_GROUP_ID)
        assert response['ReplicationGroups'][0]['ReplicationGroupId'] == self.REPLICATION_GROUP_ID

    def test_get_replication_group_status(self):
        if False:
            for i in range(10):
                print('nop')
        self._create_replication_group()
        self.hook.conn.describe_replication_groups.return_value = {'ReplicationGroups': [{'ReplicationGroupId': self.REPLICATION_GROUP_ID, 'Status': 'available'}]}
        response = self.hook.get_replication_group_status(replication_group_id=self.REPLICATION_GROUP_ID)
        assert response in self.VALID_STATES

    def test_is_replication_group_available(self):
        if False:
            return 10
        self._create_replication_group()
        self.hook.conn.describe_replication_groups.return_value = {'ReplicationGroups': [{'ReplicationGroupId': self.REPLICATION_GROUP_ID, 'Status': 'available'}]}
        response = self.hook.is_replication_group_available(replication_group_id=self.REPLICATION_GROUP_ID)
        assert response in (True, False)

    def test_wait_for_availability(self):
        if False:
            for i in range(10):
                print('nop')
        self._create_replication_group()
        self.hook.conn.describe_replication_groups.return_value = {'ReplicationGroups': [{'ReplicationGroupId': self.REPLICATION_GROUP_ID, 'Status': 'creating'}]}
        response = self.hook.wait_for_availability(replication_group_id=self.REPLICATION_GROUP_ID, max_retries=1, initial_sleep_time=1)
        assert response is False
        self.hook.conn.describe_replication_groups.return_value = {'ReplicationGroups': [{'ReplicationGroupId': self.REPLICATION_GROUP_ID, 'Status': 'available'}]}
        response = self.hook.wait_for_availability(replication_group_id=self.REPLICATION_GROUP_ID, max_retries=1, initial_sleep_time=1)
        assert response is True

    def test_delete_replication_group(self):
        if False:
            while True:
                i = 10
        self._create_replication_group()
        self.hook.conn.delete_replication_group.return_value = {'ReplicationGroup': {'ReplicationGroupId': self.REPLICATION_GROUP_ID, 'Status': 'deleting'}}
        self.hook.conn.describe_replication_groups.return_value = {'ReplicationGroups': [{'ReplicationGroupId': self.REPLICATION_GROUP_ID, 'Status': 'available'}]}
        response = self.hook.wait_for_availability(replication_group_id=self.REPLICATION_GROUP_ID, max_retries=1, initial_sleep_time=1)
        assert response is True
        response = self.hook.delete_replication_group(replication_group_id=self.REPLICATION_GROUP_ID)
        assert response['ReplicationGroup']['ReplicationGroupId'] == self.REPLICATION_GROUP_ID
        assert response['ReplicationGroup']['Status'] == 'deleting'

    def _raise_replication_group_not_found_exp(self):
        if False:
            for i in range(10):
                print('nop')
        self.hook.conn.exceptions.ReplicationGroupNotFoundFault = BaseException
        return self.hook.conn.exceptions.ReplicationGroupNotFoundFault

    def _mock_describe_side_effect(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return [{'ReplicationGroups': [{'ReplicationGroupId': self.REPLICATION_GROUP_ID, 'Status': 'available'}]}, {'ReplicationGroups': [{'ReplicationGroupId': self.REPLICATION_GROUP_ID, 'Status': 'deleting'}]}, self._raise_replication_group_not_found_exp()]

    def test_wait_for_deletion(self):
        if False:
            i = 10
            return i + 15
        self._create_replication_group()
        self.hook.conn.describe_replication_groups.side_effect = self._mock_describe_side_effect()
        self.hook.conn.delete_replication_group.return_value = {'ReplicationGroup': {'ReplicationGroupId': self.REPLICATION_GROUP_ID}}
        (response, deleted) = self.hook.wait_for_deletion(replication_group_id=self.REPLICATION_GROUP_ID, max_retries=2, initial_sleep_time=1)
        assert response['ReplicationGroup']['ReplicationGroupId'] == self.REPLICATION_GROUP_ID
        assert deleted is True

    def test_ensure_delete_replication_group_success(self):
        if False:
            i = 10
            return i + 15
        self._create_replication_group()
        self.hook.conn.describe_replication_groups.side_effect = self._mock_describe_side_effect()
        self.hook.conn.delete_replication_group.return_value = {'ReplicationGroup': {'ReplicationGroupId': self.REPLICATION_GROUP_ID}}
        response = self.hook.ensure_delete_replication_group(replication_group_id=self.REPLICATION_GROUP_ID, initial_sleep_time=1, max_retries=2)
        assert response['ReplicationGroup']['ReplicationGroupId'] == self.REPLICATION_GROUP_ID

    def test_ensure_delete_replication_group_failure(self):
        if False:
            while True:
                i = 10
        self._create_replication_group()
        self.hook.conn.describe_replication_groups.side_effect = self._mock_describe_side_effect()
        self.hook.conn.delete_replication_group.return_value = {'ReplicationGroup': {'ReplicationGroupId': self.REPLICATION_GROUP_ID}}
        with pytest.raises(AirflowException):
            self.hook.ensure_delete_replication_group(replication_group_id=self.REPLICATION_GROUP_ID, initial_sleep_time=1, max_retries=1)