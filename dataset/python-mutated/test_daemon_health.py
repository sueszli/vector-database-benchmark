import time
import pytest
from dagster._core.scheduler import DagsterDaemonScheduler
from dagster._daemon.daemon import SensorDaemon
from dagster._daemon.types import DaemonHeartbeat
from dagster._utils.error import SerializableErrorInfo
from dagster_graphql.test.utils import execute_dagster_graphql
from dagster_graphql_tests.graphql.graphql_context_test_suite import ExecutingGraphQLContextTestMatrix
INDIVIDUAL_DAEMON_QUERY = '\nquery InstanceDetailSummaryQuery {\n    instance {\n        daemonHealth {\n            id\n            sensor: daemonStatus(daemonType: "SENSOR") {\n                daemonType\n                required\n                healthy\n                lastHeartbeatTime\n            }\n            run_coordinator: daemonStatus(daemonType: "QUEUED_RUN_COORDINATOR") {\n                daemonType\n                required\n                healthy\n                lastHeartbeatTime\n            }\n            scheduler: daemonStatus(daemonType: "SCHEDULER") {\n                daemonType\n                required\n                healthy\n                lastHeartbeatTime\n            }\n        }\n    }\n}\n'
ALL_DAEMON_QUERY = '\nquery InstanceDetailSummaryQuery {\n    instance {\n        daemonHealth {\n            id\n            allDaemonStatuses {\n                daemonType\n                required\n                healthy\n                lastHeartbeatTime\n            }\n        }\n    }\n}\n'
DAEMON_HEALTH_QUERY = '\nquery InstanceDetailSummaryQuery {\n    instance {\n        daemonHealth {\n            id\n            sensor: daemonStatus(daemonType: "SENSOR"){\n                lastHeartbeatErrors {\n                    message\n                }\n                healthy\n            }\n        }\n    }\n}\n'

class TestDaemonHealth(ExecutingGraphQLContextTestMatrix):

    def test_get_individual_daemons(self, graphql_context):
        if False:
            while True:
                i = 10
        if graphql_context.instance.is_ephemeral:
            pytest.skip("The daemon isn't compatible with an in-memory instance")
        graphql_context.instance.add_daemon_heartbeat(DaemonHeartbeat(timestamp=100.0, daemon_type=SensorDaemon.daemon_type(), daemon_id=None, errors=None))
        results = execute_dagster_graphql(graphql_context, INDIVIDUAL_DAEMON_QUERY)
        scheduler_required = isinstance(graphql_context.instance.scheduler, DagsterDaemonScheduler)
        assert results.data == {'instance': {'daemonHealth': {'id': 'daemonHealth', 'sensor': {'daemonType': 'SENSOR', 'required': True, 'healthy': False, 'lastHeartbeatTime': 100.0}, 'run_coordinator': {'daemonType': 'QUEUED_RUN_COORDINATOR', 'required': False, 'healthy': None, 'lastHeartbeatTime': None}, 'scheduler': {'daemonType': 'SCHEDULER', 'required': scheduler_required, 'healthy': False if scheduler_required else None, 'lastHeartbeatTime': None}}}}

    def test_get_all_daemons(self, graphql_context):
        if False:
            i = 10
            return i + 15
        if graphql_context.instance.is_ephemeral:
            pytest.skip("The daemon isn't compatible with an in-memory instance")
        results = execute_dagster_graphql(graphql_context, ALL_DAEMON_QUERY)
        scheduler_required = isinstance(graphql_context.instance.scheduler, DagsterDaemonScheduler)
        assert results.data == {'instance': {'daemonHealth': {'id': 'daemonHealth', 'allDaemonStatuses': [{'daemonType': 'SENSOR', 'required': True, 'healthy': False, 'lastHeartbeatTime': None}, {'daemonType': 'BACKFILL', 'required': True, 'healthy': False, 'lastHeartbeatTime': None}, {'daemonType': 'ASSET', 'required': True, 'healthy': False, 'lastHeartbeatTime': None}] + ([{'daemonType': 'SCHEDULER', 'required': True, 'healthy': False if scheduler_required else None, 'lastHeartbeatTime': None}] if scheduler_required else [])}}}

    def test_get_daemon_error(self, graphql_context):
        if False:
            while True:
                i = 10
        if graphql_context.instance.is_ephemeral:
            pytest.skip("The daemon isn't compatible with an in-memory instance")
        graphql_context.instance.add_daemon_heartbeat(DaemonHeartbeat(timestamp=time.time(), daemon_type=SensorDaemon.daemon_type(), daemon_id=None, errors=[SerializableErrorInfo(message='foobar', stack=[], cls_name=None, cause=None)]))
        results = execute_dagster_graphql(graphql_context, DAEMON_HEALTH_QUERY)
        assert results.data['instance']['daemonHealth']['sensor'] == {'lastHeartbeatErrors': [{'message': 'foobar'}], 'healthy': True}