from concurrent.futures import TimeoutError
import os
import backoff
from google.api_core.exceptions import RetryError
import pytest
import admin
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']
BUCKET = os.environ['CLOUD_STORAGE_BUCKET']

class TestDatastoreAdminSnippets:

    def test_client_create(self):
        if False:
            while True:
                i = 10
        assert admin.client_create()

    def test_get_index(self):
        if False:
            return 10
        indexes = admin.list_indexes(PROJECT)
        if not indexes:
            pytest.skip('Skipping datastore test. At least one index should present in database.')
        assert admin.get_index(PROJECT, indexes[0].index_id)

    def test_list_index(self):
        if False:
            while True:
                i = 10
        assert admin.list_indexes(PROJECT)

    @pytest.mark.flaky
    @backoff.on_exception(backoff.expo, (RetryError, TimeoutError), max_tries=3)
    def test_export_import_entities(self):
        if False:
            for i in range(10):
                print('nop')
        response = admin.export_entities(PROJECT, 'gs://' + BUCKET)
        assert response
        assert admin.import_entities(PROJECT, response.output_url)