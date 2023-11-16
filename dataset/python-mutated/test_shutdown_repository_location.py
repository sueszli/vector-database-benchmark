import time
from typing import Any
from dagster._core.errors import DagsterUserCodeUnreachableError
from dagster_graphql import ShutdownRepositoryLocationStatus
from dagster_graphql.client.client_queries import SHUTDOWN_REPOSITORY_LOCATION_MUTATION
from dagster_graphql.test.utils import execute_dagster_graphql
from ..graphql.graphql_context_test_suite import GraphQLContextVariant, ReadonlyGraphQLContextTestMatrix, make_graphql_context_test_suite
BaseTestSuite: Any = make_graphql_context_test_suite(context_variants=[GraphQLContextVariant.non_launchable_sqlite_instance_deployed_grpc_env()])

class TestShutdownRepositoryLocationReadOnly(ReadonlyGraphQLContextTestMatrix):

    def test_shutdown_repository_location_permission_failure(self, graphql_context):
        if False:
            i = 10
            return i + 15
        result = execute_dagster_graphql(graphql_context, SHUTDOWN_REPOSITORY_LOCATION_MUTATION, {'repositoryLocationName': 'test'})
        assert result
        assert result.data
        assert result.data['shutdownRepositoryLocation']
        assert result.data['shutdownRepositoryLocation']['__typename'] == 'UnauthorizedError'

class TestShutdownRepositoryLocation(BaseTestSuite):

    def test_shutdown_repository_location(self, graphql_client, graphql_context):
        if False:
            i = 10
            return i + 15
        origin = next(iter(graphql_context.get_workspace_snapshot().values())).origin
        origin.create_client().heartbeat()
        result = graphql_client.shutdown_repository_location('test')
        assert result.status == ShutdownRepositoryLocationStatus.SUCCESS, result.message
        start_time = time.time()
        while time.time() - start_time < 15:
            try:
                origin.create_client().heartbeat()
            except DagsterUserCodeUnreachableError:
                return
            time.sleep(1)
        raise Exception('Timed out waiting for shutdown to take effect')

    def test_shutdown_repository_location_not_found(self, graphql_client):
        if False:
            for i in range(10):
                print('nop')
        result = graphql_client.shutdown_repository_location('not_real')
        assert result.status == ShutdownRepositoryLocationStatus.FAILURE
        assert 'Location not_real does not exist' in result.message