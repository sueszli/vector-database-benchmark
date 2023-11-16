import os
import uuid
import backoff
from google.api_core.exceptions import NotFound
from google.cloud import bigquery
import pytest
import quickstart_searchallresources
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']
DATASET = f'dataset_{uuid.uuid4().hex}'

@pytest.fixture(scope='module')
def bigquery_client():
    if False:
        return 10
    yield bigquery.Client()

@pytest.fixture(scope='module')
def asset_dataset(bigquery_client):
    if False:
        print('Hello World!')
    dataset = bigquery_client.create_dataset(DATASET)
    yield DATASET
    try:
        bigquery_client.delete_dataset(dataset)
    except NotFound as e:
        print(f'Failed to delete dataset {DATASET}')
        raise e

@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_search_all_resources(asset_dataset, capsys):
    if False:
        print('Hello World!')
    scope = f'projects/{PROJECT}'
    query = f'name:{DATASET}'

    @backoff.on_exception(backoff.expo, AssertionError, max_time=240)
    def eventually_consistent_test():
        if False:
            while True:
                i = 10
        quickstart_searchallresources.search_all_resources(scope, query=query)
        (out, _) = capsys.readouterr()
        assert DATASET in out
    eventually_consistent_test()