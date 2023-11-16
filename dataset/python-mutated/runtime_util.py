"""Runtime-related utility functions"""
from typing import Any, Optional
from streamlit import config
from streamlit.errors import MarkdownFormattedException
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.forward_msg_cache import populate_hash_if_needed

class MessageSizeError(MarkdownFormattedException):
    """Exception raised when a websocket message is larger than the configured limit."""

    def __init__(self, failed_msg_str: Any):
        if False:
            while True:
                i = 10
        msg = self._get_message(failed_msg_str)
        super(MessageSizeError, self).__init__(msg)

    def _get_message(self, failed_msg_str: Any) -> str:
        if False:
            for i in range(10):
                print('nop')
        return "\n**Data of size {message_size_mb:.1f} MB exceeds the message size limit of {message_size_limit_mb} MB.**\n\nThis is often caused by a large chart or dataframe. Please decrease the amount of data sent\nto the browser, or increase the limit by setting the config option `server.maxMessageSize`.\n[Click here to learn more about config options](https://docs.streamlit.io/library/advanced-features/configuration#set-configuration-options).\n\n_Note that increasing the limit may lead to long loading times and large memory consumption\nof the client's browser and the Streamlit server._\n".format(message_size_mb=len(failed_msg_str) / 1000000.0, message_size_limit_mb=get_max_message_size_bytes() / 1000000.0).strip('\n')

def is_cacheable_msg(msg: ForwardMsg) -> bool:
    if False:
        while True:
            i = 10
    'True if the given message qualifies for caching.'
    if msg.WhichOneof('type') in {'ref_hash', 'initialize'}:
        return False
    return msg.ByteSize() >= int(config.get_option('global.minCachedMessageSize'))

def serialize_forward_msg(msg: ForwardMsg) -> bytes:
    if False:
        while True:
            i = 10
    'Serialize a ForwardMsg to send to a client.\n\n    If the message is too large, it will be converted to an exception message\n    instead.\n    '
    populate_hash_if_needed(msg)
    msg_str = msg.SerializeToString()
    if len(msg_str) > get_max_message_size_bytes():
        import streamlit.elements.exception as exception
        exception.marshall(msg.delta.new_element.exception, MessageSizeError(msg_str))
        msg_str = msg.SerializeToString()
    return msg_str
_max_message_size_bytes: Optional[int] = None

def get_max_message_size_bytes() -> int:
    if False:
        return 10
    'Returns the max websocket message size in bytes.\n\n    This will lazyload the value from the config and store it in the global symbol table.\n    '
    global _max_message_size_bytes
    if _max_message_size_bytes is None:
        _max_message_size_bytes = config.get_option('server.maxMessageSize') * int(1000000.0)
    return _max_message_size_bytes