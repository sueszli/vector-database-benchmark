from dagster._core.workspace.context import WorkspaceRequestContext
from dagster_graphql.test.utils import execute_dagster_graphql, infer_repository_selector
INPUT_OUTPUT_DEFINITIONS_QUERY = '\n    query InputOutputDefinitionsQuery($repositorySelector: RepositorySelector!) {\n        repositoryOrError(repositorySelector: $repositorySelector) {\n           ... on Repository {\n                usedSolid(name: "op_with_input_output_metadata") {\n                    __typename\n                    definition {\n                        inputDefinitions {\n                            metadataEntries {\n                                label\n                            }\n                        }\n                        outputDefinitions {\n                            metadataEntries {\n                                label\n                            }\n                        }\n                    }\n                }\n            }\n        }\n    }\n'

def test_query_inputs_outputs(graphql_context: WorkspaceRequestContext, snapshot):
    if False:
        for i in range(10):
            print('nop')
    selector = infer_repository_selector(graphql_context)
    result = execute_dagster_graphql(graphql_context, INPUT_OUTPUT_DEFINITIONS_QUERY, variables={'repositorySelector': selector})
    assert result.data
    snapshot.assert_match(result.data)