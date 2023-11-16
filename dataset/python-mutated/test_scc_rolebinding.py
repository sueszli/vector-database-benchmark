from __future__ import annotations
import jmespath
import pytest
from tests.charts.helm_template_generator import render_chart

class TestSCCActivation:
    """Tests SCCs."""

    @pytest.mark.parametrize('rbac_enabled,scc_enabled,created', [(False, False, False), (False, True, False), (True, True, True), (True, False, False)])
    def test_create_scc(self, rbac_enabled, scc_enabled, created):
        if False:
            print('Hello World!')
        docs = render_chart(values={'multiNamespaceMode': False, 'webserver': {'defaultUser': {'enabled': True}}, 'cleanup': {'enabled': True}, 'flower': {'enabled': True}, 'rbac': {'create': rbac_enabled, 'createSCCRoleBinding': scc_enabled}}, show_only=['templates/rbac/security-context-constraint-rolebinding.yaml'])
        assert bool(docs) is created
        if created:
            assert 'RoleBinding' == jmespath.search('kind', docs[0])
            assert 'ClusterRole' == jmespath.search('roleRef.kind', docs[0])
            assert 'release-name-scc-rolebinding' == jmespath.search('metadata.name', docs[0])
            assert 'system:openshift:scc:anyuid' == jmespath.search('roleRef.name', docs[0])
            assert 'release-name-airflow-webserver' == jmespath.search('subjects[0].name', docs[0])
            assert 'release-name-airflow-worker' == jmespath.search('subjects[1].name', docs[0])
            assert 'release-name-airflow-scheduler' == jmespath.search('subjects[2].name', docs[0])
            assert 'release-name-airflow-statsd' == jmespath.search('subjects[3].name', docs[0])
            assert 'release-name-airflow-flower' == jmespath.search('subjects[4].name', docs[0])
            assert 'release-name-airflow-triggerer' == jmespath.search('subjects[5].name', docs[0])
            assert 'release-name-airflow-migrate-database-job' == jmespath.search('subjects[6].name', docs[0])
            assert 'release-name-airflow-create-user-job' == jmespath.search('subjects[7].name', docs[0])
            assert 'release-name-airflow-cleanup' == jmespath.search('subjects[8].name', docs[0])

    @pytest.mark.parametrize('rbac_enabled,scc_enabled,created', [(True, True, True)])
    def test_create_scc_multinamespace(self, rbac_enabled, scc_enabled, created):
        if False:
            while True:
                i = 10
        docs = render_chart(values={'multiNamespaceMode': True, 'webserver': {'defaultUser': {'enabled': False}}, 'cleanup': {'enabled': False}, 'flower': {'enabled': False}, 'rbac': {'create': rbac_enabled, 'createSCCRoleBinding': scc_enabled}}, show_only=['templates/rbac/security-context-constraint-rolebinding.yaml'])
        assert bool(docs) is created
        if created:
            assert 'ClusterRoleBinding' == jmespath.search('kind', docs[0])
            assert 'ClusterRole' == jmespath.search('roleRef.kind', docs[0])
            assert 'release-name-scc-rolebinding' == jmespath.search('metadata.name', docs[0])
            assert 'system:openshift:scc:anyuid' == jmespath.search('roleRef.name', docs[0])

    @pytest.mark.parametrize('rbac_enabled,scc_enabled,created', [(True, True, True)])
    def test_create_scc_worker_only(self, rbac_enabled, scc_enabled, created):
        if False:
            for i in range(10):
                print('nop')
        docs = render_chart(values={'multiNamespaceMode': False, 'webserver': {'defaultUser': {'enabled': False}}, 'cleanup': {'enabled': False}, 'flower': {'enabled': False}, 'statsd': {'enabled': False}, 'rbac': {'create': rbac_enabled, 'createSCCRoleBinding': scc_enabled}}, show_only=['templates/rbac/security-context-constraint-rolebinding.yaml'])
        assert bool(docs) is created
        if created:
            assert 'RoleBinding' == jmespath.search('kind', docs[0])
            assert 'ClusterRole' == jmespath.search('roleRef.kind', docs[0])
            assert 'release-name-scc-rolebinding' == jmespath.search('metadata.name', docs[0])
            assert 'system:openshift:scc:anyuid' == jmespath.search('roleRef.name', docs[0])
            assert 'release-name-airflow-webserver' == jmespath.search('subjects[0].name', docs[0])
            assert 'release-name-airflow-worker' == jmespath.search('subjects[1].name', docs[0])
            assert 'release-name-airflow-scheduler' == jmespath.search('subjects[2].name', docs[0])
            assert 'release-name-airflow-triggerer' == jmespath.search('subjects[3].name', docs[0])
            assert 'release-name-airflow-migrate-database-job' == jmespath.search('subjects[4].name', docs[0])