import os
import backoff
from google.api_core.exceptions import ServerError
from google.cloud import datalabeling
import pytest
import label_video
import testing_lib
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
INPUT_GCS_URI = 'gs://cloud-samples-data/datalabeling/videos/video_dataset.csv'
INSTRUCTION_GCS_URI = 'gs://cloud-samples-data/datalabeling/instruction/test.pdf'

@pytest.fixture(scope='module')
def dataset():
    if False:
        while True:
            i = 10
    dataset = testing_lib.create_dataset(PROJECT_ID)
    testing_lib.import_data(dataset.name, datalabeling.DataType.VIDEO, INPUT_GCS_URI)
    yield dataset
    testing_lib.delete_dataset(dataset.name)

@pytest.fixture(scope='module')
def annotation_spec_set():
    if False:
        print('Hello World!')
    response = testing_lib.create_annotation_spec_set(PROJECT_ID)
    yield response
    testing_lib.delete_annotation_spec_set(response.name)

@pytest.fixture(scope='module')
def instruction():
    if False:
        while True:
            i = 10
    instruction = testing_lib.create_instruction(PROJECT_ID, datalabeling.DataType.VIDEO, INSTRUCTION_GCS_URI)
    yield instruction
    testing_lib.delete_instruction(instruction.name)

@pytest.fixture(scope='module')
def cleaner():
    if False:
        for i in range(10):
            print('nop')
    resource_names = []
    yield resource_names
    for resource_name in resource_names:
        testing_lib.cancel_operation(resource_name)

@pytest.mark.skip(reason='currently unavailable')
def test_label_video(capsys, annotation_spec_set, instruction, dataset, cleaner):
    if False:
        print('Hello World!')

    @backoff.on_exception(backoff.expo, ServerError, max_time=testing_lib.RETRY_DEADLINE)
    def run_sample():
        if False:
            return 10
        return label_video.label_video(dataset.name, instruction.name, annotation_spec_set.name)
    response = run_sample()
    cleaner.append(response.operation.name)
    (out, _) = capsys.readouterr()
    assert 'Label_video operation name: ' in out
    response.cancel()
    assert response.cancelled() is True