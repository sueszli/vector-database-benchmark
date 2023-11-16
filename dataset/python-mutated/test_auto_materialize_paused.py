from dagster_graphql.test.utils import execute_dagster_graphql
from dagster_graphql_tests.graphql.graphql_context_test_suite import ExecutingGraphQLContextTestMatrix
MUTATION = '\nmutation SetAutoMaterializePausedMutation($paused: Boolean!) {\n    setAutoMaterializePaused(paused: $paused)\n}\n'
QUERY = '\nquery GetAutoMaterializePausedQuery {\n    instance {\n        autoMaterializePaused\n    }\n}\n'

class TestDaemonHealth(ExecutingGraphQLContextTestMatrix):

    def test_paused(self, graphql_context):
        if False:
            for i in range(10):
                print('nop')
        results = execute_dagster_graphql(graphql_context, QUERY)
        assert results.data == {'instance': {'autoMaterializePaused': True}}
        results = execute_dagster_graphql(graphql_context, MUTATION, variables={'paused': False})
        assert results.data == {'setAutoMaterializePaused': False}
        results = execute_dagster_graphql(graphql_context, QUERY)
        assert results.data == {'instance': {'autoMaterializePaused': False}}
        results = execute_dagster_graphql(graphql_context, MUTATION, variables={'paused': True})
        assert results.data == {'setAutoMaterializePaused': True}
        results = execute_dagster_graphql(graphql_context, QUERY)
        assert results.data == {'instance': {'autoMaterializePaused': True}}
        results = execute_dagster_graphql(graphql_context, MUTATION, variables={'paused': False})
        assert results.data == {'setAutoMaterializePaused': False}
        results = execute_dagster_graphql(graphql_context, QUERY)
        assert results.data == {'instance': {'autoMaterializePaused': False}}