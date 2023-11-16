from __future__ import annotations
import jmespath
from tests.charts.helm_template_generator import render_chart

class TestGitSyncTriggerer:
    """Test git sync triggerer."""

    def test_validate_sshkeysecret_not_added_when_persistence_is_enabled(self):
        if False:
            print('Hello World!')
        docs = render_chart(values={'dags': {'gitSync': {'enabled': True, 'containerName': 'git-sync-test', 'sshKeySecret': 'ssh-secret', 'knownHosts': None, 'branch': 'test-branch'}, 'persistence': {'enabled': True}}}, show_only=['templates/triggerer/triggerer-deployment.yaml'])
        assert 'git-sync-ssh-key' not in jmespath.search('spec.template.spec.volumes[].name', docs[0])