from mock import patch
from redash.tasks import refresh_schemas
from tests import BaseTestCase

class TestRefreshSchemas(BaseTestCase):

    def test_calls_refresh_of_all_data_sources(self):
        if False:
            print('Hello World!')
        self.factory.data_source
        with patch('redash.tasks.queries.maintenance.refresh_schema.delay') as refresh_job:
            refresh_schemas()
            refresh_job.assert_called()

    def test_skips_paused_data_sources(self):
        if False:
            return 10
        self.factory.data_source.pause()
        with patch('redash.tasks.queries.maintenance.refresh_schema.delay') as refresh_job:
            refresh_schemas()
            refresh_job.assert_not_called()
        self.factory.data_source.resume()
        with patch('redash.tasks.queries.maintenance.refresh_schema.delay') as refresh_job:
            refresh_schemas()
            refresh_job.assert_called()