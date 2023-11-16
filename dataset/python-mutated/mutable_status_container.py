from __future__ import annotations
import time
from types import TracebackType
from typing import List, Optional, Type, cast
from typing_extensions import Literal, TypeAlias
from streamlit.cursor import Cursor
from streamlit.delta_generator import DeltaGenerator, _enqueue_message
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Block_pb2 import Block as BlockProto
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
States: TypeAlias = Literal['running', 'complete', 'error']

class StatusContainer(DeltaGenerator):

    @staticmethod
    def _create(parent: DeltaGenerator, label: str, expanded: bool=False, state: States='running') -> StatusContainer:
        if False:
            return 10
        expandable_proto = BlockProto.Expandable()
        expandable_proto.expanded = expanded
        expandable_proto.label = label or ''
        if state == 'running':
            expandable_proto.icon = 'spinner'
        elif state == 'complete':
            expandable_proto.icon = 'check'
        elif state == 'error':
            expandable_proto.icon = 'error'
        else:
            raise StreamlitAPIException(f"Unknown state ({state}). Must be one of 'running', 'complete', or 'error'.")
        block_proto = BlockProto()
        block_proto.allow_empty = True
        block_proto.expandable.CopyFrom(expandable_proto)
        delta_path: List[int] = parent._active_dg._cursor.delta_path if parent._active_dg._cursor else []
        status_container = cast(StatusContainer, parent._block(block_proto=block_proto, dg_type=StatusContainer))
        status_container._delta_path = delta_path
        status_container._current_proto = block_proto
        status_container._current_state = state
        time.sleep(0.05)
        return status_container

    def __init__(self, root_container: int | None, cursor: Cursor | None, parent: DeltaGenerator | None, block_type: str | None):
        if False:
            print('Hello World!')
        super().__init__(root_container, cursor, parent, block_type)
        self._current_proto: BlockProto | None = None
        self._current_state: States | None = None
        self._delta_path: List[int] | None = None

    def update(self, *, label: str | None=None, expanded: bool | None=None, state: States | None=None) -> None:
        if False:
            return 10
        'Update the status container.\n\n        Only specified arguments are updated. Container contents and unspecified\n        arguments remain unchanged.\n\n        Parameters\n        ----------\n        label : str or None\n            A new label of the status container. If None, the label is not\n            changed.\n\n        expanded : bool or None\n            The new expanded state of the status container. If None,\n            the expanded state is not changed.\n\n        state : "running", "complete", "error", or None\n            The new state of the status container. This mainly changes the\n            icon. If None, the state is not changed.\n        '
        assert self._current_proto is not None, 'Status not correctly initialized!'
        assert self._delta_path is not None, 'Status not correctly initialized!'
        msg = ForwardMsg()
        msg.metadata.delta_path[:] = self._delta_path
        msg.delta.add_block.CopyFrom(self._current_proto)
        if expanded is not None:
            msg.delta.add_block.expandable.expanded = expanded
        else:
            msg.delta.add_block.expandable.ClearField('expanded')
        if label is not None:
            msg.delta.add_block.expandable.label = label
        if state is not None:
            if state == 'running':
                msg.delta.add_block.expandable.icon = 'spinner'
            elif state == 'complete':
                msg.delta.add_block.expandable.icon = 'check'
            elif state == 'error':
                msg.delta.add_block.expandable.icon = 'error'
            else:
                raise StreamlitAPIException(f"Unknown state ({state}). Must be one of 'running', 'complete', or 'error'.")
            self._current_state = state
        self._current_proto = msg.delta.add_block
        _enqueue_message(msg)

    def __enter__(self) -> StatusContainer:
        if False:
            return 10
        super().__enter__()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> Literal[False]:
        if False:
            print('Hello World!')
        if self._current_state == 'running':
            time.sleep(0.05)
            if exc_type is not None:
                self.update(state='error')
            else:
                self.update(state='complete')
        return super().__exit__(exc_type, exc_val, exc_tb)