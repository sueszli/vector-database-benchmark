from __future__ import annotations
import jmespath
import pytest
from tests.charts.helm_template_generator import render_chart

class TestPodReader:
    """Tests RBAC Pod Reader."""

    @pytest.mark.parametrize('triggerer, webserver, expected', [(True, True, ['release-name-airflow-webserver', 'release-name-airflow-triggerer']), (True, False, ['release-name-airflow-triggerer']), (False, True, ['release-name-airflow-webserver']), (False, False, [])])
    def test_pod_log_reader_rolebinding(self, triggerer, webserver, expected):
        if False:
            for i in range(10):
                print('nop')
        docs = render_chart(values={'triggerer': {'enabled': triggerer}, 'webserver': {'allowPodLogReading': webserver}}, show_only=['templates/rbac/pod-log-reader-rolebinding.yaml'])
        actual = jmespath.search('subjects[*].name', docs[0]) if docs else []
        assert actual == expected

    @pytest.mark.parametrize('triggerer, webserver, expected', [(True, True, 'release-name-pod-log-reader-role'), (True, False, 'release-name-pod-log-reader-role'), (False, True, 'release-name-pod-log-reader-role'), (False, False, None)])
    def test_pod_log_reader_role(self, triggerer, webserver, expected):
        if False:
            return 10
        docs = render_chart(values={'triggerer': {'enabled': triggerer}, 'webserver': {'allowPodLogReading': webserver}}, show_only=['templates/rbac/pod-log-reader-role.yaml'])
        actual = jmespath.search('metadata.name', docs[0]) if docs else None
        assert actual == expected