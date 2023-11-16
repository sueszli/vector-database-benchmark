from __future__ import annotations
import jmespath
from tests.charts.helm_template_generator import render_chart

class TestGitSyncWorker:
    """Test git sync worker."""

    def test_should_add_dags_volume_to_the_worker_if_git_sync_and_persistence_is_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        docs = render_chart(values={'executor': 'CeleryExecutor', 'dags': {'persistence': {'enabled': True}, 'gitSync': {'enabled': True}}}, show_only=['templates/workers/worker-deployment.yaml'])
        assert 'config' == jmespath.search('spec.template.spec.volumes[0].name', docs[0])
        assert 'dags' == jmespath.search('spec.template.spec.volumes[1].name', docs[0])

    def test_should_add_dags_volume_to_the_worker_if_git_sync_is_enabled_and_persistence_is_disabled(self):
        if False:
            print('Hello World!')
        docs = render_chart(values={'executor': 'CeleryExecutor', 'dags': {'gitSync': {'enabled': True}, 'persistence': {'enabled': False}}}, show_only=['templates/workers/worker-deployment.yaml'])
        assert 'config' == jmespath.search('spec.template.spec.volumes[0].name', docs[0])
        assert 'dags' == jmespath.search('spec.template.spec.volumes[1].name', docs[0])

    def test_should_add_git_sync_container_to_worker_if_persistence_is_not_enabled_but_git_sync_is(self):
        if False:
            for i in range(10):
                print('nop')
        docs = render_chart(values={'executor': 'CeleryExecutor', 'dags': {'gitSync': {'enabled': True, 'containerName': 'git-sync'}, 'persistence': {'enabled': False}}}, show_only=['templates/workers/worker-deployment.yaml'])
        assert 'git-sync' == jmespath.search('spec.template.spec.containers[1].name', docs[0])

    def test_should_not_add_sync_container_to_worker_if_git_sync_and_persistence_are_enabled(self):
        if False:
            print('Hello World!')
        docs = render_chart(values={'executor': 'CeleryExecutor', 'dags': {'gitSync': {'enabled': True, 'containerName': 'git-sync'}, 'persistence': {'enabled': True}}}, show_only=['templates/workers/worker-deployment.yaml'])
        assert 'git-sync' != jmespath.search('spec.template.spec.containers[1].name', docs[0])

    def test_should_add_env(self):
        if False:
            while True:
                i = 10
        docs = render_chart(values={'dags': {'gitSync': {'enabled': True, 'env': [{'name': 'FOO', 'value': 'bar'}]}}}, show_only=['templates/workers/worker-deployment.yaml'])
        assert {'name': 'FOO', 'value': 'bar'} in jmespath.search('spec.template.spec.containers[1].env', docs[0])

    def test_resources_are_configurable(self):
        if False:
            for i in range(10):
                print('nop')
        docs = render_chart(values={'dags': {'gitSync': {'enabled': True, 'resources': {'limits': {'cpu': '200m', 'memory': '128Mi'}, 'requests': {'cpu': '300m', 'memory': '169Mi'}}}}}, show_only=['templates/workers/worker-deployment.yaml'])
        assert '128Mi' == jmespath.search('spec.template.spec.containers[1].resources.limits.memory', docs[0])
        assert '169Mi' == jmespath.search('spec.template.spec.containers[1].resources.requests.memory', docs[0])
        assert '300m' == jmespath.search('spec.template.spec.containers[1].resources.requests.cpu', docs[0])

    def test_validate_sshkeysecret_not_added_when_persistence_is_enabled(self):
        if False:
            while True:
                i = 10
        docs = render_chart(values={'dags': {'gitSync': {'enabled': True, 'containerName': 'git-sync-test', 'sshKeySecret': 'ssh-secret', 'knownHosts': None, 'branch': 'test-branch'}, 'persistence': {'enabled': True}}}, show_only=['templates/workers/worker-deployment.yaml'])
        assert 'git-sync-ssh-key' not in jmespath.search('spec.template.spec.volumes[].name', docs[0])