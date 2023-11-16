from __future__ import annotations
import jmespath
import pytest
from tests.charts.helm_template_generator import render_chart

class TestGitSyncWebserver:
    """Test git sync webserver."""

    def test_should_add_dags_volume_to_the_webserver_if_git_sync_and_persistence_is_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        docs = render_chart(values={'airflowVersion': '1.10.14', 'dags': {'gitSync': {'enabled': True}, 'persistence': {'enabled': True}}}, show_only=['templates/webserver/webserver-deployment.yaml'])
        assert 'dags' == jmespath.search('spec.template.spec.volumes[1].name', docs[0])

    def test_should_add_dags_volume_to_the_webserver_if_git_sync_is_enabled_and_persistence_is_disabled(self):
        if False:
            for i in range(10):
                print('nop')
        docs = render_chart(values={'airflowVersion': '1.10.14', 'dags': {'gitSync': {'enabled': True}, 'persistence': {'enabled': False}}}, show_only=['templates/webserver/webserver-deployment.yaml'])
        assert 'dags' == jmespath.search('spec.template.spec.volumes[1].name', docs[0])

    def test_should_add_git_sync_container_to_webserver_if_persistence_is_not_enabled_but_git_sync_is(self):
        if False:
            print('Hello World!')
        docs = render_chart(values={'airflowVersion': '1.10.14', 'dags': {'gitSync': {'enabled': True, 'containerName': 'git-sync'}, 'persistence': {'enabled': False}}}, show_only=['templates/webserver/webserver-deployment.yaml'])
        assert 'git-sync' == jmespath.search('spec.template.spec.containers[1].name', docs[0])

    def test_should_have_service_account_defined(self):
        if False:
            i = 10
            return i + 15
        docs = render_chart(values={'dags': {'gitSync': {'enabled': True}, 'persistence': {'enabled': True}}}, show_only=['templates/webserver/webserver-deployment.yaml'])
        assert 'release-name-airflow-webserver' == jmespath.search('spec.template.spec.serviceAccountName', docs[0])

    @pytest.mark.parametrize('airflow_version, exclude_webserver', [('2.0.0', True), ('2.0.2', True), ('1.10.14', False), ('1.9.0', False), ('2.1.0', True)])
    def test_git_sync_with_different_airflow_versions(self, airflow_version, exclude_webserver):
        if False:
            return 10
        'If Airflow >= 2.0.0 - git sync related containers, volume mounts & volumes are not created.'
        docs = render_chart(values={'airflowVersion': airflow_version, 'dags': {'gitSync': {'enabled': True}, 'persistence': {'enabled': False}}}, show_only=['templates/webserver/webserver-deployment.yaml'])
        containers_names = [container['name'] for container in jmespath.search('spec.template.spec.containers', docs[0])]
        volume_mount_names = [vm['name'] for vm in jmespath.search('spec.template.spec.containers[0].volumeMounts', docs[0])]
        volume_names = [volume['name'] for volume in jmespath.search('spec.template.spec.volumes', docs[0])]
        if exclude_webserver:
            assert 'git-sync' not in containers_names
            assert 'dags' not in volume_mount_names
            assert 'dags' not in volume_names
        else:
            assert 'git-sync' in containers_names
            assert 'dags' in volume_mount_names
            assert 'dags' in volume_names

    def test_should_add_env(self):
        if False:
            while True:
                i = 10
        docs = render_chart(values={'airflowVersion': '1.10.14', 'dags': {'gitSync': {'enabled': True, 'env': [{'name': 'FOO', 'value': 'bar'}]}}}, show_only=['templates/webserver/webserver-deployment.yaml'])
        assert {'name': 'FOO', 'value': 'bar'} in jmespath.search('spec.template.spec.containers[1].env', docs[0])

    def test_resources_are_configurable(self):
        if False:
            print('Hello World!')
        docs = render_chart(values={'airflowVersion': '1.10.14', 'dags': {'gitSync': {'enabled': True, 'resources': {'limits': {'cpu': '200m', 'memory': '128Mi'}, 'requests': {'cpu': '300m', 'memory': '169Mi'}}}}}, show_only=['templates/webserver/webserver-deployment.yaml'])
        assert '128Mi' == jmespath.search('spec.template.spec.containers[1].resources.limits.memory', docs[0])
        assert '169Mi' == jmespath.search('spec.template.spec.containers[1].resources.requests.memory', docs[0])
        assert '300m' == jmespath.search('spec.template.spec.containers[1].resources.requests.cpu', docs[0])

    def test_validate_sshkeysecret_not_added_when_persistence_is_enabled(self):
        if False:
            print('Hello World!')
        docs = render_chart(values={'dags': {'gitSync': {'enabled': True, 'containerName': 'git-sync-test', 'sshKeySecret': 'ssh-secret', 'knownHosts': None, 'branch': 'test-branch'}, 'persistence': {'enabled': True}}}, show_only=['templates/webserver/webserver-deployment.yaml'])
        assert 'git-sync-ssh-key' not in jmespath.search('spec.template.spec.volumes[].name', docs[0])