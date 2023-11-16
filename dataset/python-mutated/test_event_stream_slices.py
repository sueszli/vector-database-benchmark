from itertools import islice
import pendulum
from airbyte_cdk.models import SyncMode
from source_aws_cloudtrail.source import ManagementEvents
config = {'aws_key_id': '1', 'aws_secret_key': '1', 'aws_region_name': 'us-west-1', 'start_date': '2020-05-01'}

def test_full_refresh_slice():
    if False:
        i = 10
        return i + 15
    current_time = pendulum.now().int_timestamp
    stream = ManagementEvents(**config)
    slices = stream.stream_slices(sync_mode=SyncMode.full_refresh, cursor_field=stream.cursor_field)
    assert slices[0]['StartTime'] >= current_time - ManagementEvents.data_lifetime
    assert slices[-1]['EndTime'] >= current_time

def test_incremental_slice():
    if False:
        print('Hello World!')
    current_time = pendulum.now().int_timestamp
    stream = ManagementEvents(**config)
    stream_state = {'EventTime': pendulum.today().subtract(days=15).int_timestamp}
    slices = stream.stream_slices(sync_mode=SyncMode.incremental, cursor_field=stream.cursor_field, stream_state=stream_state)
    assert slices[0]['StartTime'] == stream_state['EventTime']
    assert slices[-1]['EndTime'] >= current_time

def test_incremental_slice_state_less_than_start_date():
    if False:
        return 10
    current_time = pendulum.now().int_timestamp
    stream = ManagementEvents(**config)
    stream_state = {'EventTime': 1}
    slices = stream.stream_slices(sync_mode=SyncMode.incremental, cursor_field=stream.cursor_field, stream_state=stream_state)
    assert slices[0]['StartTime'] != stream_state['EventTime']
    assert slices[0]['StartTime'] >= current_time - ManagementEvents.data_lifetime

def test_full_refresh_slice_start_date_greater_than_now():
    if False:
        i = 10
        return i + 15
    config_with_big_start_date = config.copy()
    config_with_big_start_date['start_date'] = pendulum.now().add(days=1).format(ManagementEvents.start_date_format)
    stream = ManagementEvents(**config_with_big_start_date)
    slices = stream.stream_slices(sync_mode=SyncMode.full_refresh, cursor_field=stream.cursor_field)
    assert not slices

def test_slices_not_intersect():
    if False:
        while True:
            i = 10
    stream = ManagementEvents(**config)
    slices = stream.stream_slices(sync_mode=SyncMode.full_refresh, cursor_field=stream.cursor_field)
    for (slice, next_slice) in zip(slices, islice(slices, 1, None)):
        if next_slice is None:
            break
        assert slice['EndTime'] + 1 == next_slice['StartTime']
        assert slice['EndTime'] > slice['StartTime']