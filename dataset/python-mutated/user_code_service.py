from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from result import OkErr
from result import Result
from ...abstract_node import NodeType
from ...client.enclave_client import EnclaveClient
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...store.linked_obj import LinkedObject
from ...types.twin_object import TwinObject
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_object import ActionObject
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from ..context import AuthedServiceContext
from ..network.routes import route_to_connection
from ..request.request import SubmitRequest
from ..request.request import UserCodeStatusChange
from ..request.request_service import RequestService
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import GUEST_ROLE_LEVEL
from .user_code import SubmitUserCode
from .user_code import UserCode
from .user_code import UserCodeStatus
from .user_code import load_approved_policy_code
from .user_code_stash import UserCodeStash

@instrument
@serializable()
class UserCodeService(AbstractService):
    store: DocumentStore
    stash: UserCodeStash

    def __init__(self, store: DocumentStore) -> None:
        if False:
            while True:
                i = 10
        self.store = store
        self.stash = UserCodeStash(store=store)

    @service_method(path='code.submit', name='submit', roles=GUEST_ROLE_LEVEL)
    def submit(self, context: AuthedServiceContext, code: SubmitUserCode) -> Union[UserCode, SyftError]:
        if False:
            return 10
        'Add User Code'
        result = self._submit(context=context, code=code)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message='User Code Submitted')

    def _submit(self, context: AuthedServiceContext, code: SubmitUserCode) -> Result:
        if False:
            i = 10
            return i + 15
        result = self.stash.set(context.credentials, code.to(UserCode, context=context))
        return result

    def _request_code_execution(self, context: AuthedServiceContext, code: SubmitUserCode, reason: Optional[str]=''):
        if False:
            for i in range(10):
                print('nop')
        user_code: UserCode = code.to(UserCode, context=context)
        if not all((x in user_code.input_owner_verify_keys for x in user_code.output_readers)):
            raise ValueError('outputs can only be distributed to input owners')
        result = self.stash.set(context.credentials, user_code)
        if result.is_err():
            return SyftError(message=str(result.err()))
        code_history_service = context.node.get_service('codehistoryservice')
        result = code_history_service.submit_version(context=context, code=user_code)
        if isinstance(result, SyftError):
            return result
        self.stash.add_permissions([ActionObjectPermission(user_code.id, ActionPermission.READ, x) for x in user_code.output_readers])
        linked_obj = LinkedObject.from_obj(user_code, node_uid=context.node.id)
        CODE_EXECUTE = UserCodeStatusChange(value=UserCodeStatus.APPROVED, linked_obj=linked_obj)
        changes = [CODE_EXECUTE]
        request = SubmitRequest(changes=changes)
        method = context.node.get_service_method(RequestService.submit)
        result = method(context=context, request=request, reason=reason)
        return result

    @service_method(path='code.request_code_execution', name='request_code_execution', roles=GUEST_ROLE_LEVEL)
    def request_code_execution(self, context: AuthedServiceContext, code: SubmitUserCode, reason: Optional[str]='') -> Union[SyftSuccess, SyftError]:
        if False:
            while True:
                i = 10
        'Request Code execution on user code'
        return self._request_code_execution(context=context, code=code, reason=reason)

    @service_method(path='code.get_all', name='get_all', roles=GUEST_ROLE_LEVEL)
    def get_all(self, context: AuthedServiceContext) -> Union[List[UserCode], SyftError]:
        if False:
            return 10
        'Get a Dataset'
        result = self.stash.get_all(context.credentials)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(path='code.get_by_id', name='get_by_id')
    def get_by_uid(self, context: AuthedServiceContext, uid: UID) -> Union[SyftSuccess, SyftError]:
        if False:
            i = 10
            return i + 15
        'Get a User Code Item'
        result = self.stash.get_by_uid(context.credentials, uid=uid)
        if result.is_ok():
            user_code = result.ok()
            if user_code and user_code.input_policy_state:
                user_code.node_uid = context.node.id
            return user_code
        return SyftError(message=result.err())

    @service_method(path='code.get_all_for_user', name='get_all_for_user')
    def get_all_for_user(self, context: AuthedServiceContext) -> Union[SyftSuccess, SyftError]:
        if False:
            return 10
        "Get All User Code Items for User's VerifyKey"
        result = self.stash.get_all(context.credentials)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    def update_code_state(self, context: AuthedServiceContext, code_item: UserCode) -> Union[SyftSuccess, SyftError]:
        if False:
            for i in range(10):
                print('nop')
        result = self.stash.update(context.credentials, code_item)
        if result.is_ok():
            return SyftSuccess(message='Code State Updated')
        return SyftError(message='Unable to Update Code State')

    def load_user_code(self, context: AuthedServiceContext) -> None:
        if False:
            i = 10
            return i + 15
        result = self.stash.get_all(credentials=context.credentials)
        if result.is_ok():
            user_code_items = result.ok()
            load_approved_policy_code(user_code_items=user_code_items)

    @service_method(path='code.get_results', name='get_results', roles=GUEST_ROLE_LEVEL)
    def get_results(self, context: AuthedServiceContext, inp: Union[UID, UserCode]) -> Union[List[UserCode], SyftError]:
        if False:
            while True:
                i = 10
        uid = inp.id if isinstance(inp, UserCode) else inp
        code_result = self.stash.get_by_uid(context.credentials, uid=uid)
        if code_result.is_err():
            return SyftError(message=code_result.err())
        code = code_result.ok()
        if code.is_enclave_code:
            if not context.node.node_type == NodeType.ENCLAVE:
                connection = route_to_connection(code.enclave_metadata.route)
                enclave_client = EnclaveClient(connection=connection, credentials=context.node.signing_key)
                outputs = enclave_client.code.get_results(code.id)
                if isinstance(outputs, list):
                    for output in outputs:
                        output.syft_action_data
                else:
                    outputs.syft_action_data
                return outputs
            else:
                if not code.status.approved:
                    return code.status.get_status_message()
                if (output_policy := code.output_policy) is None:
                    return SyftError(message=f'Output policy not approved {code}')
                if len(output_policy.output_history) > 0:
                    return resolve_outputs(context=context, output_ids=output_policy.last_output_ids)
                else:
                    return SyftError(message='No results available')
        else:
            return SyftError(message='Endpoint only supported for enclave code')

    @service_method(path='code.call', name='call', roles=GUEST_ROLE_LEVEL)
    def call(self, context: AuthedServiceContext, uid: UID, **kwargs: Any) -> Union[SyftSuccess, SyftError]:
        if False:
            print('Hello World!')
        'Call a User Code Function'
        try:
            kwarg2id = map_kwargs_to_id(kwargs)
            code_result = self.stash.get_by_uid(context.credentials, uid=uid)
            if code_result.is_err():
                return SyftError(message=code_result.err())
            code: UserCode = code_result.ok()
            if not code.status.approved:
                return code.status.get_status_message()
            if not (has_code_permission := self.has_code_permission(code, context)):
                return has_code_permission
            if (output_policy := code.output_policy) is None:
                return SyftError('Output policy not approved', code)
            if not (is_valid := output_policy.valid):
                if len(output_policy.output_history) > 0:
                    result = resolve_outputs(context=context, output_ids=output_policy.last_output_ids)
                    return result.as_empty()
                return is_valid
            action_service = context.node.get_service('actionservice')
            output_result: Result[Union[ActionObject, TwinObject], str] = action_service._user_code_execute(context, code, kwarg2id)
            if output_result.is_err():
                return SyftError(message=output_result.err())
            result = output_result.ok()
            output_policy.apply_output(context=context, outputs=result)
            code.output_policy = output_policy
            if not (update_success := self.update_code_state(context=context, code_item=code)):
                return update_success
            if isinstance(result, TwinObject):
                return result.mock
            else:
                return result.as_empty()
        except Exception as e:
            return SyftError(message=f'Failed to run. {e}')

    def has_code_permission(self, code_item, context):
        if False:
            while True:
                i = 10
        if not (context.credentials == context.node.verify_key or context.credentials == code_item.user_verify_key):
            return SyftError(message=f'Code Execution Permission: {context.credentials} denied')
        return SyftSuccess(message='you have permission')

def resolve_outputs(context: AuthedServiceContext, output_ids: Optional[Union[List[UID], Dict[str, UID]]]) -> Any:
    if False:
        return 10
    from ...service.action.action_object import TwinMode
    if isinstance(output_ids, list):
        if len(output_ids) == 0:
            return None
        outputs = []
        for output_id in output_ids:
            action_service = context.node.get_service('actionservice')
            result = action_service.get(context, uid=output_id, twin_mode=TwinMode.PRIVATE)
            if isinstance(result, OkErr):
                result = result.value
            outputs.append(result)
        if len(outputs) == 1:
            return outputs[0]
        return outputs
    else:
        raise NotImplementedError

def map_kwargs_to_id(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if False:
        while True:
            i = 10
    from ...types.twin_object import TwinObject
    from ..action.action_object import ActionObject
    from ..dataset.dataset import Asset
    filtered_kwargs = {}
    for (k, v) in kwargs.items():
        value = v
        if isinstance(v, ActionObject):
            value = v.id
        if isinstance(v, TwinObject):
            value = v.id
        if isinstance(v, Asset):
            value = v.action_id
        if not isinstance(value, UID):
            raise Exception(f'Input {k} must have a UID not {type(v)}')
        filtered_kwargs[k] = value
    return filtered_kwargs
TYPE_TO_SERVICE[UserCode] = UserCodeService
SERVICE_TO_TYPES[UserCodeService].update({UserCode})