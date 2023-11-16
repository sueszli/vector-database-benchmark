import pytest
from superset.connectors.sqla.models import SqlaTable
from superset.extensions import db
from superset.models.core import FavStar
from superset.models.dashboard import Dashboard
from superset.models.slice import Slice
from superset.models.sql_lab import SavedQuery
from superset.tags.models import TaggedObject
from superset.utils.core import DatasourceType
from superset.utils.database import get_main_database
from tests.integration_tests.base_tests import SupersetTestCase
from tests.integration_tests.conftest import with_feature_flags
from tests.integration_tests.fixtures.tags import with_tagging_system_feature

class TestTagging(SupersetTestCase):

    def query_tagged_object_table(self):
        if False:
            print('Hello World!')
        query = db.session.query(TaggedObject).all()
        return query

    def clear_tagged_object_table(self):
        if False:
            i = 10
            return i + 15
        db.session.query(TaggedObject).delete()
        db.session.commit()

    @pytest.mark.usefixtures('with_tagging_system_feature')
    def test_dataset_tagging(self):
        if False:
            while True:
                i = 10
        '\n        Test to make sure that when a new dataset is created,\n        a corresponding tag in the tagged_objects table\n        is created\n        '
        self.clear_tagged_object_table()
        self.assertEqual([], self.query_tagged_object_table())
        test_dataset = SqlaTable(table_name='foo', schema=None, owners=[], database=get_main_database(), sql=None, extra='{"certification": 1}')
        db.session.add(test_dataset)
        db.session.commit()
        tags = self.query_tagged_object_table()
        self.assertEqual(1, len(tags))
        self.assertEqual('ObjectType.dataset', str(tags[0].object_type))
        self.assertEqual(test_dataset.id, tags[0].object_id)
        db.session.delete(test_dataset)
        db.session.commit()
        self.assertEqual([], self.query_tagged_object_table())

    @pytest.mark.usefixtures('with_tagging_system_feature')
    def test_chart_tagging(self):
        if False:
            while True:
                i = 10
        '\n        Test to make sure that when a new chart is created,\n        a corresponding tag in the tagged_objects table\n        is created\n        '
        self.clear_tagged_object_table()
        self.assertEqual([], self.query_tagged_object_table())
        test_chart = Slice(slice_name='test_chart', datasource_type=DatasourceType.TABLE, viz_type='bubble', datasource_id=1, id=1)
        db.session.add(test_chart)
        db.session.commit()
        tags = self.query_tagged_object_table()
        self.assertEqual(1, len(tags))
        self.assertEqual('ObjectType.chart', str(tags[0].object_type))
        self.assertEqual(test_chart.id, tags[0].object_id)
        db.session.delete(test_chart)
        db.session.commit()
        self.assertEqual([], self.query_tagged_object_table())

    @pytest.mark.usefixtures('with_tagging_system_feature')
    def test_dashboard_tagging(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test to make sure that when a new dashboard is created,\n        a corresponding tag in the tagged_objects table\n        is created\n        '
        self.clear_tagged_object_table()
        self.assertEqual([], self.query_tagged_object_table())
        test_dashboard = Dashboard()
        test_dashboard.dashboard_title = 'test_dashboard'
        test_dashboard.slug = 'test_slug'
        test_dashboard.published = True
        db.session.add(test_dashboard)
        db.session.commit()
        tags = self.query_tagged_object_table()
        self.assertEqual(1, len(tags))
        self.assertEqual('ObjectType.dashboard', str(tags[0].object_type))
        self.assertEqual(test_dashboard.id, tags[0].object_id)
        db.session.delete(test_dashboard)
        db.session.commit()
        self.assertEqual([], self.query_tagged_object_table())

    @pytest.mark.usefixtures('with_tagging_system_feature')
    def test_saved_query_tagging(self):
        if False:
            print('Hello World!')
        '\n        Test to make sure that when a new saved query is\n        created, a corresponding tag in the tagged_objects\n        table is created\n        '
        self.clear_tagged_object_table()
        self.assertEqual([], self.query_tagged_object_table())
        test_saved_query = SavedQuery(id=1, label='test saved query')
        db.session.add(test_saved_query)
        db.session.commit()
        tags = self.query_tagged_object_table()
        self.assertEqual(2, len(tags))
        self.assertEqual('ObjectType.query', str(tags[0].object_type))
        self.assertEqual('owner:None', str(tags[0].tag.name))
        self.assertEqual('TagType.owner', str(tags[0].tag.type))
        self.assertEqual(test_saved_query.id, tags[0].object_id)
        self.assertEqual('ObjectType.query', str(tags[1].object_type))
        self.assertEqual('type:query', str(tags[1].tag.name))
        self.assertEqual('TagType.type', str(tags[1].tag.type))
        self.assertEqual(test_saved_query.id, tags[1].object_id)
        db.session.delete(test_saved_query)
        db.session.commit()
        self.assertEqual([], self.query_tagged_object_table())

    @pytest.mark.usefixtures('with_tagging_system_feature')
    def test_favorite_tagging(self):
        if False:
            return 10
        '\n        Test to make sure that when a new favorite object is\n        created, a corresponding tag in the tagged_objects\n        table is created\n        '
        self.clear_tagged_object_table()
        self.assertEqual([], self.query_tagged_object_table())
        test_saved_query = FavStar(user_id=1, class_name='slice', obj_id=1)
        db.session.add(test_saved_query)
        db.session.commit()
        tags = self.query_tagged_object_table()
        self.assertEqual(1, len(tags))
        self.assertEqual('ObjectType.chart', str(tags[0].object_type))
        self.assertEqual(test_saved_query.obj_id, tags[0].object_id)
        db.session.delete(test_saved_query)
        db.session.commit()
        self.assertEqual([], self.query_tagged_object_table())

    @with_feature_flags(TAGGING_SYSTEM=False)
    def test_tagging_system(self):
        if False:
            i = 10
            return i + 15
        '\n        Test to make sure that when the TAGGING_SYSTEM\n        feature flag is false, that no tags are created\n        '
        self.clear_tagged_object_table()
        self.assertEqual([], self.query_tagged_object_table())
        test_dataset = SqlaTable(table_name='foo', schema=None, owners=[], database=get_main_database(), sql=None, extra='{"certification": 1}')
        test_chart = Slice(slice_name='test_chart', datasource_type=DatasourceType.TABLE, viz_type='bubble', datasource_id=1, id=1)
        test_dashboard = Dashboard()
        test_dashboard.dashboard_title = 'test_dashboard'
        test_dashboard.slug = 'test_slug'
        test_dashboard.published = True
        test_saved_query = SavedQuery(id=1, label='test saved query')
        test_favorited_object = FavStar(user_id=1, class_name='slice', obj_id=1)
        db.session.add(test_dataset)
        db.session.add(test_chart)
        db.session.add(test_dashboard)
        db.session.add(test_saved_query)
        db.session.add(test_favorited_object)
        db.session.commit()
        tags = self.query_tagged_object_table()
        self.assertEqual(0, len(tags))
        db.session.delete(test_dataset)
        db.session.delete(test_chart)
        db.session.delete(test_dashboard)
        db.session.delete(test_saved_query)
        db.session.delete(test_favorited_object)
        db.session.commit()
        self.assertEqual([], self.query_tagged_object_table())