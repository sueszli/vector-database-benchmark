from dagster._core.workspace.context import WorkspaceRequestContext
from dagster_graphql.test.utils import execute_dagster_graphql, infer_job_selector
from .graphql_context_test_suite import NonLaunchableGraphQLContextTestMatrix
from .repo import csv_hello_world_ops_config
RUN_CONFIG_SCHEMA_QUERY = '\nquery($selector: PipelineSelector! $mode: String!)\n{\n  runConfigSchemaOrError(selector: $selector, mode: $mode){\n    __typename\n    ... on RunConfigSchema {\n      rootConfigType {\n        key\n      }\n      allConfigTypes {\n        key\n      }\n      rootDefaultYaml\n    }\n  }\n}\n'
RUN_CONFIG_SCHEMA_ROOT_DEFAULT_YAML_QUERY = '\nquery($selector: PipelineSelector! $mode: String!)\n{\n  runConfigSchemaOrError(selector: $selector, mode: $mode){\n    __typename\n    ... on RunConfigSchema {\n      rootDefaultYaml\n    }\n  }\n}\n'
RUN_CONFIG_SCHEMA_CONFIG_TYPE_QUERY = '\nquery($selector: PipelineSelector! $mode: String! $configTypeName: String!)\n{\n  runConfigSchemaOrError(selector: $selector, mode: $mode){\n    __typename\n    ... on RunConfigSchema {\n      configTypeOrError(configTypeName: $configTypeName) {\n        __typename\n        ... on EnumConfigType {\n          name\n        }\n        ... on RegularConfigType {\n          name\n        }\n        ... on CompositeConfigType {\n          name\n        }\n      }\n    }\n  }\n}\n'
RUN_CONFIG_SCHEMA_CONFIG_VALIDATION_QUERY = '\nquery PipelineQuery(\n    $runConfigData: RunConfigData,\n    $selector: PipelineSelector!,\n    $mode: String!\n) {\n  runConfigSchemaOrError(selector: $selector mode: $mode) {\n    ... on RunConfigSchema {\n      isRunConfigValid(runConfigData: $runConfigData) {\n        __typename\n        ... on PipelineConfigValidationValid {\n            pipelineName\n        }\n        ... on RunConfigValidationInvalid {\n            pipelineName\n            errors {\n                __typename\n                ... on RuntimeMismatchConfigError {\n                    valueRep\n                }\n                ... on MissingFieldConfigError {\n                    field { name }\n                }\n                ... on MissingFieldsConfigError {\n                    fields { name }\n                }\n                ... on FieldNotDefinedConfigError {\n                    fieldName\n                }\n                ... on FieldsNotDefinedConfigError {\n                    fieldNames\n                }\n                ... on SelectorTypeConfigError {\n                    incomingFields\n                }\n                message\n                reason\n                stack {\n                    entries {\n                        __typename\n                        ... on EvaluationStackPathEntry {\n                            fieldName\n                        }\n                        ... on EvaluationStackListItemEntry {\n                            listIndex\n                        }\n                        ... on EvaluationStackMapKeyEntry {\n                            mapKey\n                        }\n                        ... on EvaluationStackMapValueEntry {\n                            mapKey\n                        }\n                    }\n                }\n            }\n        }\n        ... on PipelineNotFoundError {\n            pipelineName\n        }\n      }\n    }\n  }\n}\n'

class TestEnvironmentSchema(NonLaunchableGraphQLContextTestMatrix):

    def test_successful_run_config_schema(self, graphql_context: WorkspaceRequestContext):
        if False:
            for i in range(10):
                print('nop')
        selector = infer_job_selector(graphql_context, 'required_resource_job')
        result = execute_dagster_graphql(graphql_context, RUN_CONFIG_SCHEMA_QUERY, variables={'selector': selector, 'mode': 'default'})
        assert result.data['runConfigSchemaOrError']['__typename'] == 'RunConfigSchema'

    def test_run_config_schema_pipeline_not_found(self, graphql_context: WorkspaceRequestContext):
        if False:
            i = 10
            return i + 15
        selector = infer_job_selector(graphql_context, 'jkdjfkdjfd')
        result = execute_dagster_graphql(graphql_context, RUN_CONFIG_SCHEMA_QUERY, variables={'selector': selector, 'mode': 'default'})
        assert result.data['runConfigSchemaOrError']['__typename'] == 'PipelineNotFoundError'

    def test_run_config_schema_op_not_found(self, graphql_context: WorkspaceRequestContext):
        if False:
            print('Hello World!')
        selector = infer_job_selector(graphql_context, 'required_resource_job', ['kdjfkdj'])
        result = execute_dagster_graphql(graphql_context, RUN_CONFIG_SCHEMA_QUERY, variables={'selector': selector, 'mode': 'default'})
        assert result.data['runConfigSchemaOrError']['__typename'] == 'InvalidSubsetError'

    def test_run_config_schema_mode_not_found(self, graphql_context: WorkspaceRequestContext):
        if False:
            return 10
        selector = infer_job_selector(graphql_context, 'required_resource_job')
        result = execute_dagster_graphql(graphql_context, RUN_CONFIG_SCHEMA_QUERY, variables={'selector': selector, 'mode': 'kdjfdk'})
        assert result.data['runConfigSchemaOrError']['__typename'] == 'ModeNotFoundError'

    def test_basic_valid_config_on_run_config_schema(self, graphql_context: WorkspaceRequestContext, snapshot):
        if False:
            while True:
                i = 10
        selector = infer_job_selector(graphql_context, 'csv_hello_world')
        result = execute_dagster_graphql(graphql_context, RUN_CONFIG_SCHEMA_CONFIG_VALIDATION_QUERY, variables={'selector': selector, 'mode': 'default', 'runConfigData': csv_hello_world_ops_config()})
        assert not result.errors
        assert result.data
        assert result.data['runConfigSchemaOrError']['isRunConfigValid']['__typename'] == 'PipelineConfigValidationValid'
        snapshot.assert_match(result.data)

    def test_full_yaml(self, graphql_context, snapshot):
        if False:
            return 10
        selector = infer_job_selector(graphql_context, 'csv_hello_world')
        result = execute_dagster_graphql(graphql_context, RUN_CONFIG_SCHEMA_ROOT_DEFAULT_YAML_QUERY, variables={'selector': selector, 'mode': 'default', 'runConfigData': csv_hello_world_ops_config()})
        assert result
        assert not result.errors
        assert result.data
        snapshot.assert_match(result.data)

    def test_basic_invalid_config_on_run_config_schema(self, graphql_context: WorkspaceRequestContext, snapshot):
        if False:
            i = 10
            return i + 15
        selector = infer_job_selector(graphql_context, 'csv_hello_world')
        result = execute_dagster_graphql(graphql_context, RUN_CONFIG_SCHEMA_CONFIG_VALIDATION_QUERY, variables={'selector': selector, 'mode': 'default', 'runConfigData': {'nope': 'kdjfd'}})
        assert not result.errors
        assert result.data
        assert result.data['runConfigSchemaOrError']['isRunConfigValid']['__typename'] == 'RunConfigValidationInvalid'
        snapshot.assert_match(result.data)