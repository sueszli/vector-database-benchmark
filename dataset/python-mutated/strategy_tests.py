"""Unit tests for Superset cache warmup"""
from unittest.mock import MagicMock
from tests.integration_tests.fixtures.birth_names_dashboard import load_birth_names_dashboard_with_slices, load_birth_names_data
from sqlalchemy import String, Date, Float
import pytest
import pandas as pd
from superset.models.slice import Slice
from superset.utils.database import get_example_database
from superset import db
from superset.models.core import Log
from superset.tags.models import get_tag, ObjectType, TaggedObject, TagType
from superset.tasks.cache import DashboardTagsStrategy, TopNDashboardsStrategy
from superset.utils.urls import get_url_host
from .base_tests import SupersetTestCase
from .dashboard_utils import create_dashboard, create_slice, create_table_metadata
from .fixtures.unicode_dashboard import load_unicode_dashboard_with_slice, load_unicode_data
mock_positions = {'DASHBOARD_VERSION_KEY': 'v2', 'DASHBOARD_CHART_TYPE-1': {'type': 'CHART', 'id': 'DASHBOARD_CHART_TYPE-1', 'children': [], 'meta': {'width': 4, 'height': 50, 'chartId': 1}}, 'DASHBOARD_CHART_TYPE-2': {'type': 'CHART', 'id': 'DASHBOARD_CHART_TYPE-2', 'children': [], 'meta': {'width': 4, 'height': 50, 'chartId': 2}}}

class TestCacheWarmUp(SupersetTestCase):

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_top_n_dashboards_strategy(self):
        if False:
            i = 10
            return i + 15
        db.session.query(Log).delete()
        self.login(username='admin')
        dash = self.get_dash_by_slug('births')
        for _ in range(10):
            self.client.get(f'/superset/dashboard/{dash.id}/')
        strategy = TopNDashboardsStrategy(1)
        result = strategy.get_payloads()
        expected = [{'chart_id': chart.id, 'dashboard_id': dash.id} for chart in dash.slices]
        self.assertCountEqual(result, expected)

    def reset_tag(self, tag):
        if False:
            for i in range(10):
                print('nop')
        'Remove associated object from tag, used to reset tests'
        if tag.objects:
            for o in tag.objects:
                db.session.delete(o)
            db.session.commit()

    @pytest.mark.usefixtures('load_unicode_dashboard_with_slice', 'load_birth_names_dashboard_with_slices')
    def test_dashboard_tags_strategy(self):
        if False:
            return 10
        tag1 = get_tag('tag1', db.session, TagType.custom)
        self.reset_tag(tag1)
        strategy = DashboardTagsStrategy(['tag1'])
        result = strategy.get_payloads()
        expected = []
        self.assertEqual(result, expected)
        tag1 = get_tag('tag1', db.session, TagType.custom)
        dash = self.get_dash_by_slug('births')
        tag1_urls = [{'chart_id': chart.id} for chart in dash.slices]
        tagged_object = TaggedObject(tag_id=tag1.id, object_id=dash.id, object_type=ObjectType.dashboard)
        db.session.add(tagged_object)
        db.session.commit()
        self.assertCountEqual(strategy.get_payloads(), tag1_urls)
        strategy = DashboardTagsStrategy(['tag2'])
        tag2 = get_tag('tag2', db.session, TagType.custom)
        self.reset_tag(tag2)
        result = strategy.get_payloads()
        expected = []
        self.assertEqual(result, expected)
        dash = self.get_dash_by_slug('unicode-test')
        chart = dash.slices[0]
        tag2_urls = [{'chart_id': chart.id}]
        object_id = chart.id
        tagged_object = TaggedObject(tag_id=tag2.id, object_id=object_id, object_type=ObjectType.chart)
        db.session.add(tagged_object)
        db.session.commit()
        result = strategy.get_payloads()
        self.assertCountEqual(result, tag2_urls)
        strategy = DashboardTagsStrategy(['tag1', 'tag2'])
        result = strategy.get_payloads()
        expected = tag1_urls + tag2_urls
        self.assertCountEqual(result, expected)