from typing import AbstractSet, Any, Mapping
from dagster._core.workspace.context import WorkspaceRequestContext
from dagster_graphql.test.utils import GqlResult, execute_dagster_graphql, infer_job_selector
from .graphql_context_test_suite import NonLaunchableGraphQLContextTestMatrix
SCHEMA_OR_ERROR_SUBSET_QUERY = '\nquery EnvironmentQuery($selector: PipelineSelector!){\n    runConfigSchemaOrError(selector: $selector) {\n        __typename\n        ... on RunConfigSchema {\n            allConfigTypes {\n                __typename\n                key\n                ... on CompositeConfigType {\n                    __typename\n                    fields {\n                        __typename\n                        name\n                        configType {\n                            key\n                            __typename\n                        }\n                    }\n                }\n            }\n        }\n        ... on InvalidSubsetError {\n            message\n        }\n        ... on PythonError {\n            message\n            stack\n        }\n    }\n}\n'

def field_names_of(type_dict: Any, typename: str) -> AbstractSet[str]:
    if False:
        print('Hello World!')
    return {field_data['name'] for field_data in type_dict[typename]['fields']}

def types_dict_of_result(subset_result: GqlResult, top_key: str) -> Mapping[str, Any]:
    if False:
        return 10
    return {type_data['name']: type_data for type_data in subset_result.data[top_key]['configTypes']}

class TestSolidSelections(NonLaunchableGraphQLContextTestMatrix):

    def test_csv_hello_world_pipeline_or_error_subset_wrong_solid_name(self, graphql_context: WorkspaceRequestContext):
        if False:
            i = 10
            return i + 15
        selector = infer_job_selector(graphql_context, 'csv_hello_world', ['nope'])
        result = execute_dagster_graphql(graphql_context, SCHEMA_OR_ERROR_SUBSET_QUERY, {'selector': selector})
        assert not result.errors
        assert result.data
        assert result.data['runConfigSchemaOrError']['__typename'] == 'InvalidSubsetError'
        assert 'No qualified ops to execute' in result.data['runConfigSchemaOrError']['message']

    def test_pipeline_with_invalid_definition_error(self, graphql_context: WorkspaceRequestContext):
        if False:
            return 10
        selector = infer_job_selector(graphql_context, 'job_with_invalid_definition_error', ['fail_subset'])
        result = execute_dagster_graphql(graphql_context, SCHEMA_OR_ERROR_SUBSET_QUERY, {'selector': selector})
        assert not result.errors
        assert result.data
        assert result.data['runConfigSchemaOrError']['__typename'] == 'InvalidSubsetError'
        error_msg = result.data['runConfigSchemaOrError']['message']
        assert 'DagsterInvalidSubsetError' in error_msg
        assert "Input 'some_input' of op 'fail_subset' has no way of being resolved" in error_msg