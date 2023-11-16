from typing import Dict
from pyflink.fn_execution.embedded.converters import from_type_info_proto, DataConverter

class SideOutputContext(object):

    def __init__(self, j_side_output_context):
        if False:
            while True:
                i = 10
        self._j_side_output_context = j_side_output_context
        self._side_output_converters = {tag_id: from_type_info_proto(_parse_type_info_proto(payload)) for (tag_id, payload) in j_side_output_context.getAllSideOutputTypeInfoPayloads().items()}

    def collect(self, tag_id: str, record):
        if False:
            for i in range(10):
                print('nop')
        try:
            self._j_side_output_context.collectSideOutputById(tag_id, self._side_output_converters[tag_id].to_external(record))
        except KeyError:
            raise Exception('Unknown OutputTag id {0}, supported OutputTag ids are {1}'.format(tag_id, list(self._side_output_converters.keys())))

def _parse_type_info_proto(type_info_payload):
    if False:
        return 10
    from pyflink.fn_execution import flink_fn_execution_pb2
    type_info = flink_fn_execution_pb2.TypeInfo()
    type_info.ParseFromString(type_info_payload)
    return type_info