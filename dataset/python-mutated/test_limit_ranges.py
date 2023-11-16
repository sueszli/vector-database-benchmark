from __future__ import annotations
import jmespath
from tests.charts.helm_template_generator import render_chart

class TestLimitRanges:
    """Tests limit ranges."""

    def test_limit_ranges_template(self):
        if False:
            print('Hello World!')
        docs = render_chart(values={'limits': [{'max': {'cpu': '500m'}, 'min': {'min': '200m'}, 'type': 'Container'}]}, show_only=['templates/limitrange.yaml'])
        assert 'LimitRange' == jmespath.search('kind', docs[0])
        assert '500m' == jmespath.search('spec.limits[0].max.cpu', docs[0])

    def test_limit_ranges_are_not_added_by_default(self):
        if False:
            i = 10
            return i + 15
        docs = render_chart(show_only=['templates/limitrange.yaml'])
        assert docs == []