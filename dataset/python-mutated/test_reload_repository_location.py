import sys
from typing import Any
from unittest import mock
from dagster import file_relative_path, repository
from dagster._core.code_pointer import CodePointer
from dagster._core.host_representation import ManagedGrpcPythonEnvCodeLocationOrigin, external_repository_data_from_def
from dagster._core.types.loadable_target_origin import LoadableTargetOrigin
from dagster._core.workspace.load import location_origins_from_yaml_paths
from dagster._grpc.types import ListRepositoriesResponse
from dagster_graphql.test.utils import execute_dagster_graphql
from .graphql_context_test_suite import GraphQLContextVariant, ReadonlyGraphQLContextTestMatrix, make_graphql_context_test_suite
RELOAD_REPOSITORY_LOCATION_QUERY = '\nmutation ($repositoryLocationName: String!) {\n   reloadRepositoryLocation(repositoryLocationName: $repositoryLocationName) {\n      __typename\n      ... on WorkspaceLocationEntry {\n        id\n        name\n        loadStatus\n        locationOrLoadError {\n            __typename\n            ... on RepositoryLocation {\n                name\n                repositories {\n                    name\n                    displayMetadata {\n                        key\n                        value\n                    }\n                }\n                isReloadSupported\n            }\n            ... on PythonError {\n                message\n            }\n        }\n      }\n   }\n}\n'
RELOAD_WORKSPACE_QUERY = '\nmutation {\n   reloadWorkspace {\n      __typename\n      ... on Workspace {\n        locationEntries {\n          __typename\n          id\n          name\n          loadStatus\n          locationOrLoadError {\n            __typename\n            ... on RepositoryLocation {\n                id\n                name\n                repositories {\n                    name\n                }\n                isReloadSupported\n            }\n            ... on PythonError {\n                message\n            }\n          }\n        }\n      }\n  }\n}\n'
MultiLocationTestSuite: Any = make_graphql_context_test_suite(context_variants=[GraphQLContextVariant.non_launchable_sqlite_instance_multi_location()])
OutOfProcessTestSuite: Any = make_graphql_context_test_suite(context_variants=[GraphQLContextVariant.non_launchable_sqlite_instance_managed_grpc_env()])
ManagedTestSuite: Any = make_graphql_context_test_suite(context_variants=[GraphQLContextVariant.non_launchable_sqlite_instance_managed_grpc_env()])
CodeServerCliTestSuite: Any = make_graphql_context_test_suite(context_variants=[GraphQLContextVariant.sqlite_with_default_run_launcher_code_server_cli_env()])

class TestReloadWorkspaceReadOnly(ReadonlyGraphQLContextTestMatrix):

    def test_reload_workspace_permission_failure(self, graphql_context):
        if False:
            for i in range(10):
                print('nop')
        result = execute_dagster_graphql(graphql_context, RELOAD_WORKSPACE_QUERY)
        assert result
        assert result.data
        assert result.data['reloadWorkspace']
        assert result.data['reloadWorkspace']['__typename'] == 'UnauthorizedError'

class TestReloadWorkspace(MultiLocationTestSuite):

    def test_reload_workspace(self, graphql_context):
        if False:
            i = 10
            return i + 15
        result = execute_dagster_graphql(graphql_context, RELOAD_WORKSPACE_QUERY)
        assert result
        assert result.data
        assert result.data['reloadWorkspace']
        assert result.data['reloadWorkspace']['__typename'] == 'Workspace'
        nodes = result.data['reloadWorkspace']['locationEntries']
        assert len(nodes) == 2
        assert all([node['locationOrLoadError']['__typename'] == 'RepositoryLocation' for node in nodes])
        original_origins = location_origins_from_yaml_paths([file_relative_path(__file__, 'multi_location.yaml')])
        with mock.patch('dagster._core.workspace.load_target.location_origins_from_yaml_paths') as origins_mock:
            origins_mock.return_value = original_origins[0:1]
            result = execute_dagster_graphql(graphql_context, RELOAD_WORKSPACE_QUERY)
            assert result
            assert result.data
            assert result.data['reloadWorkspace']
            assert result.data['reloadWorkspace']['__typename'] == 'Workspace'
            nodes = result.data['reloadWorkspace']['locationEntries']
            assert len(nodes) == 1
            assert all([node['locationOrLoadError']['__typename'] == 'RepositoryLocation' and node['loadStatus'] == 'LOADED' for node in nodes])
            original_origins.append(ManagedGrpcPythonEnvCodeLocationOrigin(location_name='error_location', loadable_target_origin=LoadableTargetOrigin(python_file='made_up_file.py', executable_path=sys.executable)))
            origins_mock.return_value = original_origins
            result = execute_dagster_graphql(graphql_context, RELOAD_WORKSPACE_QUERY)
            assert result
            assert result.data
            assert result.data['reloadWorkspace']
            assert result.data['reloadWorkspace']['__typename'] == 'Workspace'
            nodes = result.data['reloadWorkspace']['locationEntries']
            assert len(nodes) == 3
            assert len([node for node in nodes if node['locationOrLoadError']['__typename'] == 'RepositoryLocation' and node['loadStatus'] == 'LOADED']) == 2
            failures = [node for node in nodes if node['locationOrLoadError']['__typename'] == 'PythonError']
            assert len(failures) == 1
            assert failures[0]['name'] == 'error_location'
            assert failures[0]['loadStatus'] == 'LOADED'
            original_origins.append(original_origins[0]._replace(location_name='location_copy'))
            origins_mock.return_value = original_origins
            result = execute_dagster_graphql(graphql_context, RELOAD_WORKSPACE_QUERY)
            nodes = result.data['reloadWorkspace']['locationEntries']
            assert len(nodes) == 4
            assert len([node for node in nodes if node['locationOrLoadError']['__typename'] == 'RepositoryLocation']) == 3
            failures = [node for node in nodes if node['locationOrLoadError']['__typename'] == 'PythonError']
            assert len(failures) == 1
            assert 'location_copy' in [node['name'] for node in nodes]
            assert original_origins[0].location_name in [node['name'] for node in nodes]
            original_origins[0] = original_origins[0]._replace(location_name='new_location_name')
            result = execute_dagster_graphql(graphql_context, RELOAD_WORKSPACE_QUERY)
            nodes = result.data['reloadWorkspace']['locationEntries']
            assert len(nodes) == 4
            assert len([node for node in nodes if node['locationOrLoadError']['__typename'] == 'RepositoryLocation']) == 3
            failures = [node for node in nodes if node['locationOrLoadError']['__typename'] == 'PythonError']
            assert len(failures) == 1
            assert 'new_location_name' in [node['name'] for node in nodes]

class TestReloadRepositoriesReadOnly(ReadonlyGraphQLContextTestMatrix):

    def test_reload_repository_permission_failure(self, graphql_context):
        if False:
            for i in range(10):
                print('nop')
        result = execute_dagster_graphql(graphql_context, RELOAD_REPOSITORY_LOCATION_QUERY, {'repositoryLocationName': 'test'})
        assert result
        assert result.data
        assert result.data['reloadRepositoryLocation']
        assert result.data['reloadRepositoryLocation']['__typename'] == 'UnauthorizedError'

class TestReloadRepositoriesOutOfProcess(OutOfProcessTestSuite):

    def test_out_of_process_reload_location(self, graphql_context):
        if False:
            print('Hello World!')
        result = execute_dagster_graphql(graphql_context, RELOAD_REPOSITORY_LOCATION_QUERY, {'repositoryLocationName': 'test'})
        assert result
        assert result.data
        assert result.data['reloadRepositoryLocation']
        assert result.data['reloadRepositoryLocation']['__typename'] == 'WorkspaceLocationEntry'
        assert result.data['reloadRepositoryLocation']['name'] == 'test'
        repositories = result.data['reloadRepositoryLocation']['locationOrLoadError']['repositories']
        assert len(repositories) == 1
        assert repositories[0]['name'] == 'test_repo'
        assert result.data['reloadRepositoryLocation']['locationOrLoadError']['isReloadSupported'] is True
        with mock.patch('dagster._core.host_representation.code_location.sync_list_repositories_grpc') as cli_command_mock:
            with mock.patch('dagster._core.host_representation.code_location.sync_get_streaming_external_repositories_data_grpc') as external_repository_mock:

                @repository
                def new_repo():
                    if False:
                        return 10
                    return []
                new_repo_data = external_repository_data_from_def(new_repo)
                external_repository_mock.return_value = {'new_repo': new_repo_data}
                cli_command_mock.return_value = ListRepositoriesResponse(repository_symbols=[], executable_path=sys.executable, repository_code_pointer_dict={'new_repo': CodePointer.from_python_file(__file__, 'new_repo', None)})
                result = execute_dagster_graphql(graphql_context, RELOAD_REPOSITORY_LOCATION_QUERY, {'repositoryLocationName': 'test'})
                assert cli_command_mock.call_count == 1
                assert external_repository_mock.call_count == 1
                repositories = result.data['reloadRepositoryLocation']['locationOrLoadError']['repositories']
                assert len(repositories) == 1
                assert repositories[0]['name'] == 'new_repo'

    def test_reload_failure(self, graphql_context):
        if False:
            i = 10
            return i + 15
        result = execute_dagster_graphql(graphql_context, RELOAD_REPOSITORY_LOCATION_QUERY, {'repositoryLocationName': 'test'})
        assert result
        assert result.data
        assert result.data['reloadRepositoryLocation']
        assert result.data['reloadRepositoryLocation']['locationOrLoadError']['__typename'] == 'RepositoryLocation'
        assert result.data['reloadRepositoryLocation']['name'] == 'test'
        repositories = result.data['reloadRepositoryLocation']['locationOrLoadError']['repositories']
        assert len(repositories) == 1
        assert repositories[0]['name'] == 'test_repo'
        assert result.data['reloadRepositoryLocation']['locationOrLoadError']['isReloadSupported'] is True
        with mock.patch('dagster._core.host_representation.code_location.sync_list_repositories_grpc') as cli_command_mock:
            cli_command_mock.side_effect = Exception('Mocked repository load failure')
            result = execute_dagster_graphql(graphql_context, RELOAD_REPOSITORY_LOCATION_QUERY, {'repositoryLocationName': 'test'})
            assert result
            assert result.data
            assert result.data['reloadRepositoryLocation']
            assert result.data['reloadRepositoryLocation']['locationOrLoadError']['__typename'] == 'PythonError'
            assert result.data['reloadRepositoryLocation']['name'] == 'test'
            assert 'Mocked repository load failure' in result.data['reloadRepositoryLocation']['locationOrLoadError']['message']
            result = execute_dagster_graphql(graphql_context, RELOAD_REPOSITORY_LOCATION_QUERY, {'repositoryLocationName': 'test'})
            assert result
            assert result.data
            assert result.data['reloadRepositoryLocation']
            assert result.data['reloadRepositoryLocation']['locationOrLoadError']['__typename'] == 'PythonError'
            assert result.data['reloadRepositoryLocation']['name'] == 'test'
            assert 'Mocked repository load failure' in result.data['reloadRepositoryLocation']['locationOrLoadError']['message']
        result = execute_dagster_graphql(graphql_context, RELOAD_REPOSITORY_LOCATION_QUERY, {'repositoryLocationName': 'test'})
        assert result
        assert result.data
        assert result.data['reloadRepositoryLocation']
        assert result.data['reloadRepositoryLocation']['locationOrLoadError']['__typename'] == 'RepositoryLocation'
        assert result.data['reloadRepositoryLocation']['name'] == 'test'
        assert result.data['reloadRepositoryLocation']['loadStatus'] == 'LOADED'
        repositories = result.data['reloadRepositoryLocation']['locationOrLoadError']['repositories']
        assert len(repositories) == 1
        assert repositories[0]['name'] == 'test_repo'
        assert result.data['reloadRepositoryLocation']['locationOrLoadError']['isReloadSupported'] is True

class TestReloadRepositoriesManagedGrpc(ManagedTestSuite):

    def test_managed_grpc_reload_location(self, graphql_context):
        if False:
            return 10
        result = execute_dagster_graphql(graphql_context, RELOAD_REPOSITORY_LOCATION_QUERY, {'repositoryLocationName': 'test'})
        assert result
        assert result.data
        assert result.data['reloadRepositoryLocation']
        assert result.data['reloadRepositoryLocation']['locationOrLoadError']['__typename'] == 'RepositoryLocation'
        assert result.data['reloadRepositoryLocation']['name'] == 'test'
        assert result.data['reloadRepositoryLocation']['loadStatus'] == 'LOADED'
        repositories = result.data['reloadRepositoryLocation']['locationOrLoadError']['repositories']
        assert len(repositories) == 1
        assert repositories[0]['name'] == 'test_repo'
        metadatas = repositories[0]['displayMetadata']
        metadata_dict = {metadata['key']: metadata['value'] for metadata in metadatas}
        assert 'python_file' in metadata_dict or 'module_name' in metadata_dict or 'package_name' in metadata_dict
        assert result.data['reloadRepositoryLocation']['locationOrLoadError']['isReloadSupported'] is True

class TestReloadLocationCodeServerCliGrpc(CodeServerCliTestSuite):

    def test_code_server_cli_reload_location(self, graphql_context):
        if False:
            i = 10
            return i + 15
        old_server_id = graphql_context.get_code_location('test').server_id
        result = execute_dagster_graphql(graphql_context, RELOAD_REPOSITORY_LOCATION_QUERY, {'repositoryLocationName': 'test'})
        assert result
        assert result.data
        assert result.data['reloadRepositoryLocation']
        assert result.data['reloadRepositoryLocation']['locationOrLoadError']['__typename'] == 'RepositoryLocation'
        assert result.data['reloadRepositoryLocation']['name'] == 'test'
        assert result.data['reloadRepositoryLocation']['loadStatus'] == 'LOADED'
        new_location = graphql_context.process_context.create_snapshot()['test'].code_location
        assert new_location.server_id != old_server_id