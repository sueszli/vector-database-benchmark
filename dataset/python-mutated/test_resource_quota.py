from __future__ import annotations
import jmespath
from tests.charts.helm_template_generator import render_chart

class TestResourceQuota:
    """Tests resource quota."""

    def test_resource_quota_template(self):
        if False:
            while True:
                i = 10
        docs = render_chart(values={'quotas': {'configmaps': '10', 'persistentvolumeclaims': '4', 'pods': '4', 'replicationcontrollers': '20', 'secrets': '10', 'services': '10'}}, show_only=['templates/resourcequota.yaml'])
        assert 'ResourceQuota' == jmespath.search('kind', docs[0])
        assert '20' == jmespath.search('spec.hard.replicationcontrollers', docs[0])

    def test_resource_quota_are_not_added_by_default(self):
        if False:
            while True:
                i = 10
        docs = render_chart(show_only=['templates/resourcequota.yaml'])
        assert docs == []