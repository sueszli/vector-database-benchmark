from typing import List
from typing import Tuple
from typing import Union
from pydantic.error_wrappers import ValidationError
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import PartitionKey
from ...store.document_store import QueryKeys
from ...types.uid import UID
from ..code.user_code import UserVerifyKeyPartitionKey
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from .action_graph import ActionGraphStore
from .action_graph import ExecutionStatus
from .action_graph import NodeActionData
from .action_graph import NodeActionDataUpdate
from .action_graph import NodeType
from .action_object import Action
from .action_object import ActionObject
ExecutionStatusPartitionKey = PartitionKey(key='status', type_=ExecutionStatus)

@serializable()
class ActionGraphService(AbstractService):
    store: ActionGraphStore

    def __init__(self, store: ActionGraphStore):
        if False:
            return 10
        self.store = store

    @service_method(path='graph.add_action', name='add_action')
    def add_action(self, context: AuthedServiceContext, action: Action) -> Union[NodeActionData, SyftError]:
        if False:
            while True:
                i = 10
        (input_uids, output_uid) = self._extract_input_and_output_from_action(action=action)
        node = NodeActionData.from_action(action=action, credentials=context.credentials)
        result = self.store.set(credentials=context.credentials, node=node, parent_uids=input_uids)
        if result.is_err():
            return SyftError(message=result.err())
        action_node = result.ok()
        if action_node.is_mutagen:
            result = self.store.update_non_mutated_successor(node_id=action.remote_self.id, nm_successor_id=action_node.id, credentials=context.credentials)
        else:
            node = NodeActionData(id=output_uid, user_verify_key=context.credentials, type=NodeType.ACTION_OBJECT)
            result = self.store.set(credentials=context.credentials, node=node, parent_uids=[action.id])
        if result.is_err():
            return SyftError(message=result.err())
        result_node = result.ok()
        return (action_node, result_node)

    @service_method(path='graph.add_action_obj', name='add_action_obj')
    def add_action_obj(self, context: AuthedServiceContext, action_obj: ActionObject) -> Union[NodeActionData, SyftError]:
        if False:
            for i in range(10):
                print('nop')
        node = NodeActionData.from_action_obj(action_obj=action_obj, credentials=context.credentials)
        result = self.store.set(credentials=context.credentials, node=node)
        if result.is_err():
            return SyftError(message=result.err())
        return result.ok()

    def _extract_input_and_output_from_action(self, action: Action) -> Tuple[UID]:
        if False:
            return 10
        input_uids = set()
        if action.remote_self is not None:
            input_uids.add(action.remote_self.id)
        for arg in action.args:
            input_uids.add(arg.id)
        for (_, kwarg) in action.kwargs.items():
            input_uids.add(kwarg.id)
        output_uid = action.result_id.id
        return (input_uids, output_uid)

    def get(self, uid: UID, context: AuthedServiceContext) -> Union[NodeActionData, SyftError]:
        if False:
            print('Hello World!')
        result = self.store.get(uid=uid, credentials=context.credentials)
        if result.is_err():
            return SyftError(message=result.err())
        return result.ok()

    def remove_node(self, context: AuthedServiceContext, uid: UID) -> Union[SyftSuccess, SyftError]:
        if False:
            return 10
        result = self.store.delete(uid=uid, credentials=context.credentials)
        if result.is_ok():
            return SyftSuccess(message=f'Successfully deleted node with uid: {uid} from the graph.')
        return SyftError(message=result.err())

    def get_all_nodes(self, context: AuthedServiceContext) -> Union[List, SyftError]:
        if False:
            for i in range(10):
                print('nop')
        result = self.store.nodes(context.credentials)
        if result.is_ok():
            return result.ok()
        return SyftError(message='Failed to fetch nodes from the graph')

    def get_all_edges(self, context: AuthedServiceContext) -> Union[List, SyftError]:
        if False:
            return 10
        result = self.store.edges(context.credentials)
        if result.is_ok():
            return result.ok()
        return SyftError(message='Failed to fetch nodes from the graph')

    def update(self, context: AuthedServiceContext, uid: UID, node_data: NodeActionDataUpdate) -> Union[NodeActionData, SyftError]:
        if False:
            i = 10
            return i + 15
        result = self.store.update(uid=uid, data=node_data, credentials=context.credentials)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    def update_action_status(self, context: AuthedServiceContext, action_id: UID, status: ExecutionStatus) -> Union[SyftSuccess, SyftError]:
        if False:
            return 10
        try:
            node_data = NodeActionDataUpdate(status=status)
        except ValidationError as e:
            return SyftError(message=f'ValidationError: {e}')
        result = self.store.update(uid=action_id, data=node_data, credentials=context.credentials)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    def get_by_action_status(self, context: AuthedServiceContext, status: ExecutionStatus) -> Union[List[NodeActionData], SyftError]:
        if False:
            for i in range(10):
                print('nop')
        qks = QueryKeys(qks=[ExecutionStatusPartitionKey.with_obj(status)])
        result = self.store.query(qks=qks, credentials=context.credentials)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    def get_by_verify_key(self, context: AuthedServiceContext, verify_key: SyftVerifyKey) -> Union[List[NodeActionData], SyftError]:
        if False:
            i = 10
            return i + 15
        qks = QueryKeys(qks=[UserVerifyKeyPartitionKey.with_obj(verify_key)])
        result = self.store.query(qks=qks, credentials=context.credentials)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())