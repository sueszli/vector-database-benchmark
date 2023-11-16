import os
import backoff
from google.api_core.exceptions import ServerError
from google.cloud import datalabeling
import pytest
import create_instruction
import testing_lib
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
INSTRUCTION_GCS_URI = 'gs://cloud-samples-data/datalabeling/instruction/test.pdf'

@pytest.fixture(scope='module')
def cleaner():
    if False:
        while True:
            i = 10
    resource_names = []
    yield resource_names
    for resource_name in resource_names:
        testing_lib.delete_instruction(resource_name)

@pytest.mark.skip(reason='service is limited due to covid')
def test_create_instruction(cleaner, capsys):
    if False:
        print('Hello World!')

    @backoff.on_exception(backoff.expo, ServerError, max_time=testing_lib.RETRY_DEADLINE)
    def run_sample():
        if False:
            return 10
        return create_instruction.create_instruction(PROJECT_ID, datalabeling.DataType.IMAGE, INSTRUCTION_GCS_URI)
    instruction = run_sample()
    cleaner.append(instruction.name)
    (out, _) = capsys.readouterr()
    assert 'The instruction resource name: ' in out