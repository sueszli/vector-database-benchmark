import os
import backoff
from google.api_core.exceptions import ServerError
import pytest
import create_annotation_spec_set
import testing_lib
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')

@pytest.fixture(scope='module')
def cleaner():
    if False:
        while True:
            i = 10
    resource_names = []
    yield resource_names
    for resource_name in resource_names:
        testing_lib.delete_annotation_spec_set(resource_name)

@pytest.mark.skip(reason='service is limited due to covid')
def test_create_annotation_spec_set(cleaner, capsys):
    if False:
        print('Hello World!')

    @backoff.on_exception(backoff.expo, ServerError, max_time=testing_lib.RETRY_DEADLINE)
    def run_sample():
        if False:
            for i in range(10):
                print('nop')
        return create_annotation_spec_set.create_annotation_spec_set(PROJECT_ID)
    response = run_sample()
    cleaner.append(response.name)
    (out, _) = capsys.readouterr()
    assert 'The annotation_spec_set resource name:' in out