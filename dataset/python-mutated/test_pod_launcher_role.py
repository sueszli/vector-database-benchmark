from __future__ import annotations
import jmespath
import pytest
from tests.charts.helm_template_generator import render_chart

class TestPodLauncher:
    """Tests pod launcher."""

    @pytest.mark.parametrize('executor, rbac, allow, expected_accounts', [('CeleryKubernetesExecutor', True, True, ['scheduler', 'worker']), ('KubernetesExecutor', True, True, ['scheduler', 'worker']), ('CeleryExecutor', True, True, ['worker']), ('LocalExecutor', True, True, ['scheduler']), ('LocalExecutor', False, False, [])])
    def test_pod_launcher_role(self, executor, rbac, allow, expected_accounts):
        if False:
            for i in range(10):
                print('nop')
        docs = render_chart(values={'rbac': {'create': rbac}, 'allowPodLaunching': allow, 'executor': executor}, show_only=['templates/rbac/pod-launcher-rolebinding.yaml'])
        if expected_accounts:
            for (idx, suffix) in enumerate(expected_accounts):
                assert f'release-name-airflow-{suffix}' == jmespath.search(f'subjects[{idx}].name', docs[0])
        else:
            assert [] == docs