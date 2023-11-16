from typing import Any, Dict, List, Optional, Tuple
from streamlit.logger import get_logger
from streamlit.proto.Delta_pb2 import Delta
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
LOGGER = get_logger(__name__)

class ForwardMsgQueue:
    """Accumulates a session's outgoing ForwardMsgs.

    Each AppSession adds messages to its queue, and the Server periodically
    flushes all session queues and delivers their messages to the appropriate
    clients.

    ForwardMsgQueue is not thread-safe - a queue should only be used from
    a single thread.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._queue: List[ForwardMsg] = []
        self._delta_index_map: Dict[Tuple[int, ...], int] = dict()

    def get_debug(self) -> Dict[str, Any]:
        if False:
            return 10
        from google.protobuf.json_format import MessageToDict
        return {'queue': [MessageToDict(m) for m in self._queue], 'ids': list(self._delta_index_map.keys())}

    def is_empty(self) -> bool:
        if False:
            while True:
                i = 10
        return len(self._queue) == 0

    def enqueue(self, msg: ForwardMsg) -> None:
        if False:
            while True:
                i = 10
        'Add message into queue, possibly composing it with another message.'
        if not _is_composable_message(msg):
            self._queue.append(msg)
            return
        delta_key = tuple(msg.metadata.delta_path)
        if delta_key in self._delta_index_map:
            index = self._delta_index_map[delta_key]
            old_msg = self._queue[index]
            composed_delta = _maybe_compose_deltas(old_msg.delta, msg.delta)
            if composed_delta is not None:
                new_msg = ForwardMsg()
                new_msg.delta.CopyFrom(composed_delta)
                new_msg.metadata.CopyFrom(msg.metadata)
                self._queue[index] = new_msg
                return
        self._delta_index_map[delta_key] = len(self._queue)
        self._queue.append(msg)

    def clear(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Clear the queue.'
        self._queue = []
        self._delta_index_map = dict()

    def flush(self) -> List[ForwardMsg]:
        if False:
            while True:
                i = 10
        'Clear the queue and return a list of the messages it contained\n        before being cleared.\n        '
        queue = self._queue
        self.clear()
        return queue

def _is_composable_message(msg: ForwardMsg) -> bool:
    if False:
        i = 10
        return i + 15
    'True if the ForwardMsg is potentially composable with other ForwardMsgs.'
    if not msg.HasField('delta'):
        return False
    delta_type = msg.delta.WhichOneof('type')
    return delta_type != 'add_rows' and delta_type != 'arrow_add_rows'

def _maybe_compose_deltas(old_delta: Delta, new_delta: Delta) -> Optional[Delta]:
    if False:
        for i in range(10):
            print('nop')
    'Combines new_delta onto old_delta if possible.\n\n    If the combination takes place, the function returns a new Delta that\n    should replace old_delta in the queue.\n\n    If the new_delta is incompatible with old_delta, the function returns None.\n    In this case, the new_delta should just be appended to the queue as normal.\n    '
    old_delta_type = old_delta.WhichOneof('type')
    if old_delta_type == 'add_block':
        return None
    new_delta_type = new_delta.WhichOneof('type')
    if new_delta_type == 'new_element':
        return new_delta
    if new_delta_type == 'add_block':
        return new_delta
    return None