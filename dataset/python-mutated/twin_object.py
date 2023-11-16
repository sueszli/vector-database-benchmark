from __future__ import annotations
from typing import Any
from typing import Dict
from typing import Optional
import pydantic
from ..serde.serializable import serializable
from ..service.action.action_object import ActionObject
from ..service.action.action_object import TwinMode
from ..service.action.action_types import action_types
from .syft_object import SyftObject
from .uid import UID

def to_action_object(obj: Any) -> ActionObject:
    if False:
        for i in range(10):
            print('nop')
    if isinstance(obj, ActionObject):
        return obj
    if type(obj) in action_types:
        return action_types[type(obj)](syft_action_data_cache=obj)
    raise Exception(f'{type(obj)} not in action_types')

@serializable()
class TwinObject(SyftObject):
    __canonical_name__ = 'TwinObject'
    __version__ = 1
    __attr_searchable__ = []
    id: UID
    private_obj: ActionObject
    private_obj_id: UID = None
    mock_obj: ActionObject
    mock_obj_id: UID = None

    @pydantic.validator('private_obj', pre=True, always=True)
    def make_private_obj(cls, v: ActionObject) -> ActionObject:
        if False:
            i = 10
            return i + 15
        return to_action_object(v)

    @pydantic.validator('private_obj_id', pre=True, always=True)
    def make_private_obj_id(cls, v: Optional[UID], values: Dict) -> UID:
        if False:
            print('Hello World!')
        return values['private_obj'].id if v is None else v

    @pydantic.validator('mock_obj', pre=True, always=True)
    def make_mock_obj(cls, v: ActionObject):
        if False:
            print('Hello World!')
        return to_action_object(v)

    @pydantic.validator('mock_obj_id', pre=True, always=True)
    def make_mock_obj_id(cls, v: Optional[UID], values: Dict) -> UID:
        if False:
            i = 10
            return i + 15
        return values['mock_obj'].id if v is None else v

    @property
    def private(self) -> ActionObject:
        if False:
            return 10
        twin_id = self.id
        private = self.private_obj
        private.syft_twin_type = TwinMode.PRIVATE
        private.id = twin_id
        return private

    @property
    def mock(self) -> ActionObject:
        if False:
            i = 10
            return i + 15
        twin_id = self.id
        mock = self.mock_obj
        mock.syft_twin_type = TwinMode.MOCK
        mock.id = twin_id
        return mock

    def _save_to_blob_storage(self):
        if False:
            i = 10
            return i + 15
        self.private_obj._set_obj_location_(self.syft_node_location, self.syft_client_verify_key)
        return self.private_obj._save_to_blob_storage()