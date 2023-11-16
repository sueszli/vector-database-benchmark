import os
import uuid
import pytest
import create_channel
import create_channel_event
import create_input
import delete_channel
import delete_channel_event
import delete_input
import get_channel_event
import list_channel_events
import start_channel
import stop_channel
project_name = os.environ['GOOGLE_CLOUD_PROJECT']
location = 'us-central1'
input_id = f'python-test-input-{uuid.uuid4()}'
channel_id = f'python-test-channel-{uuid.uuid4()}'
event_id = f'python-test-event-{uuid.uuid4()}'
output_bucket_name = f'python-test-bucket-{uuid.uuid4()}'
output_uri = f'gs://{output_bucket_name}/channel-test/'

def test_channel_event_operations(capsys: pytest.fixture) -> None:
    if False:
        print('Hello World!')
    event_name_project_id = f'projects/{project_name}/locations/{location}/channels/{channel_id}/events/{event_id}'
    create_input.create_input(project_name, location, input_id)
    create_channel.create_channel(project_name, location, channel_id, input_id, output_uri)
    start_channel.start_channel(project_name, location, channel_id)
    response = create_channel_event.create_channel_event(project_name, location, channel_id, event_id)
    assert event_name_project_id in response.name
    response = get_channel_event.get_channel_event(project_name, location, channel_id, event_id)
    assert event_name_project_id in response.name
    list_channel_events.list_channel_events(project_name, location, channel_id)
    (out, _) = capsys.readouterr()
    assert event_name_project_id in out
    response = delete_channel_event.delete_channel_event(project_name, location, channel_id, event_id)
    assert response is None
    stop_channel.stop_channel(project_name, location, channel_id)
    delete_channel.delete_channel(project_name, location, channel_id)
    delete_input.delete_input(project_name, location, input_id)