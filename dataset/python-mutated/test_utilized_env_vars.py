from dagster_graphql.test.utils import execute_dagster_graphql, infer_repository_selector
UTILIZED_ENV_VARS_QUERY = '\nquery UtilizedEnvVarsQuery($selector: RepositorySelector!) {\n  utilizedEnvVarsOrError(repositorySelector: $selector) {\n    __typename\n    ... on EnvVarWithConsumersList{\n      results {\n        envVarName\n        envVarConsumers {\n            type\n            name\n        }\n      }\n    }\n  }\n}\n'

def test_get_used_env_vars(definitions_graphql_context, snapshot) -> None:
    if False:
        i = 10
        return i + 15
    selector = infer_repository_selector(definitions_graphql_context)
    result = execute_dagster_graphql(definitions_graphql_context, UTILIZED_ENV_VARS_QUERY, {'selector': selector})
    assert not result.errors
    assert result.data
    assert result.data['utilizedEnvVarsOrError']

    def sort_env_var_entry(entry):
        if False:
            i = 10
            return i + 15
        return {'envVarName': entry['envVarName'], 'envVarConsumers': sorted(entry['envVarConsumers'], key=lambda x: x['name'])}
    sorted_env_vars = sorted([sort_env_var_entry(x) for x in result.data['utilizedEnvVarsOrError']['results']], key=lambda x: x['envVarName'])
    assert sorted_env_vars == [{'envVarName': 'MY_OTHER_STRING', 'envVarConsumers': [{'type': 'RESOURCE', 'name': 'my_resource_two_env_vars'}]}, {'envVarName': 'MY_STRING', 'envVarConsumers': [{'type': 'RESOURCE', 'name': 'my_resource_env_vars'}, {'type': 'RESOURCE', 'name': 'my_resource_two_env_vars'}]}]
    result.data['utilizedEnvVarsOrError']['results'] = sorted_env_vars
    snapshot.assert_match(result.data)