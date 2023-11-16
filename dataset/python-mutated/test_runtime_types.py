from dagster_graphql.test.utils import execute_dagster_graphql, infer_job_selector
RUNTIME_TYPE_QUERY = '\nquery DagsterTypeQuery($selector: PipelineSelector! $dagsterTypeName: String!)\n{\n    pipelineOrError(params: $selector) {\n        __typename\n        ... on Pipeline {\n            dagsterTypeOrError(dagsterTypeName: $dagsterTypeName) {\n                __typename\n                ... on RegularDagsterType {\n                    name\n                    displayName\n                    isBuiltin\n                }\n                ... on DagsterTypeNotFoundError {\n                    dagsterTypeName\n                }\n            }\n        }\n        ... on PipelineNotFoundError {\n            pipelineName\n        }\n    }\n}\n'
ALL_RUNTIME_TYPES_QUERY = '\nfragment schemaTypeFragment on ConfigType {\n  key\n  ... on CompositeConfigType {\n    fields {\n      name\n      configType {\n        key\n      }\n    }\n    recursiveConfigTypes {\n      key\n    }\n  }\n}\n\nfragment dagsterTypeFragment on DagsterType {\n    key\n    name\n    displayName\n    isNullable\n    isList\n    description\n    inputSchemaType {\n        ...schemaTypeFragment\n    }\n    outputSchemaType {\n        ...schemaTypeFragment\n    }\n    innerTypes {\n        key\n    }\n    metadataEntries {\n        label\n    }\n    ... on WrappingDagsterType {\n        ofType {\n            key\n        }\n    }\n}\n\n{\n  repositoriesOrError {\n    ... on RepositoryConnection {\n      nodes {\n        pipelines {\n          name\n          dagsterTypes {\n            ...dagsterTypeFragment\n          }\n        }\n      }\n    }\n  }\n}\n'

def test_dagster_type_query_works(graphql_context):
    if False:
        for i in range(10):
            print('nop')
    selector = infer_job_selector(graphql_context, 'csv_hello_world')
    result = execute_dagster_graphql(graphql_context, RUNTIME_TYPE_QUERY, {'selector': selector, 'dagsterTypeName': 'PoorMansDataFrame'})
    assert not result.errors
    assert result.data
    assert result.data['pipelineOrError']['dagsterTypeOrError']['__typename'] == 'RegularDagsterType'
    assert result.data['pipelineOrError']['dagsterTypeOrError']['name'] == 'PoorMansDataFrame'

def test_dagster_type_builtin_query(graphql_context):
    if False:
        print('Hello World!')
    selector = infer_job_selector(graphql_context, 'csv_hello_world')
    result = execute_dagster_graphql(graphql_context, RUNTIME_TYPE_QUERY, {'selector': selector, 'dagsterTypeName': 'Int'})
    assert not result.errors
    assert result.data
    assert result.data['pipelineOrError']['dagsterTypeOrError']['__typename'] == 'RegularDagsterType'
    assert result.data['pipelineOrError']['dagsterTypeOrError']['name'] == 'Int'
    assert result.data['pipelineOrError']['dagsterTypeOrError']['isBuiltin']

def test_dagster_type_or_error_pipeline_not_found(graphql_context):
    if False:
        for i in range(10):
            print('nop')
    selector = infer_job_selector(graphql_context, 'nope')
    result = execute_dagster_graphql(graphql_context, RUNTIME_TYPE_QUERY, {'selector': selector, 'dagsterTypeName': 'nope'})
    assert not result.errors
    assert result.data
    assert result.data['pipelineOrError']['__typename'] == 'PipelineNotFoundError'
    assert result.data['pipelineOrError']['pipelineName'] == 'nope'

def test_dagster_type_or_error_type_not_found(graphql_context):
    if False:
        print('Hello World!')
    selector = infer_job_selector(graphql_context, 'csv_hello_world')
    result = execute_dagster_graphql(graphql_context, RUNTIME_TYPE_QUERY, {'selector': selector, 'dagsterTypeName': 'nope'})
    assert not result.errors
    assert result.data
    assert result.data['pipelineOrError']['dagsterTypeOrError']['__typename'] == 'DagsterTypeNotFoundError'
    assert result.data['pipelineOrError']['dagsterTypeOrError']['dagsterTypeName'] == 'nope'

def test_smoke_test_dagster_type_system(graphql_context):
    if False:
        i = 10
        return i + 15
    result = execute_dagster_graphql(graphql_context, ALL_RUNTIME_TYPES_QUERY)
    assert not result.errors
    assert result.data