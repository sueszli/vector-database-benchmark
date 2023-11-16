from dagster._core.workspace.context import WorkspaceRequestContext
from dagster_graphql.test.utils import execute_dagster_graphql, infer_repository_selector

def all_solids_query():
    if False:
        print('Hello World!')
    return '\n    query AllSolidsQuery($repositorySelector: RepositorySelector!) {\n        repositoryOrError(repositorySelector: $repositorySelector) {\n           ... on Repository {\n                usedSolids {\n                    __typename\n                    definition { name }\n                    invocations { pipeline { name } solidHandle { handleID } }\n                }\n            }\n        }\n    }\n    '

def get_solid_query_exists():
    if False:
        for i in range(10):
            print('nop')
    return '\n    query SolidsQuery($repositorySelector: RepositorySelector!) {\n        repositoryOrError(repositorySelector: $repositorySelector) {\n            ... on Repository {\n                usedSolid(name: "sum_op") {\n                    definition { name }\n                }\n            }\n            ... on PythonError {\n                message\n                stack\n            }\n        }\n    }\n    '

def test_query_all_solids(graphql_context: WorkspaceRequestContext, snapshot):
    if False:
        print('Hello World!')
    selector = infer_repository_selector(graphql_context)
    result = execute_dagster_graphql(graphql_context, all_solids_query(), variables={'repositorySelector': selector})
    snapshot.assert_match(result.data)

def test_query_get_solid_exists(graphql_context: WorkspaceRequestContext):
    if False:
        i = 10
        return i + 15
    selector = infer_repository_selector(graphql_context)
    result = execute_dagster_graphql(graphql_context, get_solid_query_exists(), variables={'repositorySelector': selector})
    assert not result.errors
    print(result.data['repositoryOrError'])
    assert result.data['repositoryOrError']['usedSolid']['definition']['name'] == 'sum_op'