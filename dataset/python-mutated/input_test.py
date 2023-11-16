import os
import uuid
from google.api_core.exceptions import FailedPrecondition, NotFound
from google.protobuf import empty_pb2 as empty
import pytest
import create_input
import delete_input
import get_input
import list_inputs
import update_input
import utils
project_name = os.environ['GOOGLE_CLOUD_PROJECT']
location = 'us-central1'
input_id = f'my-python-test-input-{uuid.uuid4()}'

def test_input_operations(capsys: pytest.fixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    responses = list_inputs.list_inputs(project_name, location)
    for response in responses:
        next_input_id = response.name.rsplit('/', 1)[-1]
        if utils.is_resource_stale(response.create_time):
            try:
                delete_input.delete_input(project_name, location, next_input_id)
            except FailedPrecondition as e:
                print(f'Ignoring FailedPrecondition, details: {e}')
            except NotFound as e:
                print(f'Ignoring NotFound, details: {e}')
    input_name_project_id = f'projects/{project_name}/locations/{location}/inputs/{input_id}'
    response = create_input.create_input(project_name, location, input_id)
    assert input_name_project_id in response.name
    list_inputs.list_inputs(project_name, location)
    (out, _) = capsys.readouterr()
    assert input_name_project_id in out
    response = update_input.update_input(project_name, location, input_id)
    assert input_name_project_id in response.name
    assert response.preprocessing_config.crop.top_pixels == 5
    response = get_input.get_input(project_name, location, input_id)
    assert input_name_project_id in response.name
    response = delete_input.delete_input(project_name, location, input_id)
    assert response == empty.Empty()