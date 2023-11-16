"""Shared protobuf message mocking utilities."""
from streamlit.cursor import make_delta_path
from streamlit.elements import arrow
from streamlit.elements.arrow import Data
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.proto.RootContainer_pb2 import RootContainer

def create_dataframe_msg(df: Data, id: int=1) -> ForwardMsg:
    if False:
        print('Hello World!')
    'Create a mock legacy_data_frame ForwardMsg.'
    msg = ForwardMsg()
    msg.metadata.delta_path[:] = make_delta_path(RootContainer.SIDEBAR, (), id)
    arrow.marshall(msg.delta.new_element.arrow_data_frame, df)
    return msg

def create_script_finished_message(status: 'ForwardMsg.ScriptFinishedStatus.ValueType') -> ForwardMsg:
    if False:
        return 10
    'Create a script_finished ForwardMsg.'
    msg = ForwardMsg()
    msg.script_finished = status
    return msg