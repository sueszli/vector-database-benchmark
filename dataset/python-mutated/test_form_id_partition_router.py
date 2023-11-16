from unittest.mock import Mock
import pytest
from airbyte_cdk.sources.declarative.partition_routers.substream_partition_router import ParentStreamConfig
from source_typeform.components import FormIdPartitionRouter
test_cases = [(['form_id_1', 'form_id_2'], [{'stream': Mock(read_records=Mock(return_value=[{'id': 'form_id_3'}, {'id': 'form_id_4'}]))}], [{'form_id': 'form_id_1'}, {'form_id': 'form_id_2'}]), ([], [{'stream': Mock(read_records=Mock(return_value=[{'id': 'form_id_3'}, {'id': 'form_id_4'}]))}, {'stream': Mock(read_records=Mock(return_value=[{'id': 'form_id_5'}, {'id': 'form_id_6'}]))}], [{'form_id': 'form_id_3'}, {'form_id': 'form_id_4'}, {'form_id': 'form_id_5'}, {'form_id': 'form_id_6'}])]

@pytest.mark.parametrize('form_ids, parent_stream_configs, expected_slices', test_cases)
def test_stream_slices(form_ids, parent_stream_configs, expected_slices):
    if False:
        i = 10
        return i + 15
    stream_configs = []
    for parent_stream_config in parent_stream_configs:
        stream_config = ParentStreamConfig(stream=parent_stream_config['stream'], parent_key=None, partition_field=None, config=None, parameters=None)
        stream_configs.append(stream_config)
    if not stream_configs:
        stream_configs = [None]
    router = FormIdPartitionRouter(config={'form_ids': form_ids}, parent_stream_configs=stream_configs, parameters=None)
    slices = list(router.stream_slices())
    assert slices == expected_slices