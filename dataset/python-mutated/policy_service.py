from typing import List
from typing import Union
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.uid import UID
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from .policy import SubmitUserPolicy
from .policy import UserPolicy
from .user_policy_stash import UserPolicyStash

@serializable()
class PolicyService(AbstractService):
    store: DocumentStore
    stash: UserPolicyStash

    def __init__(self, store: DocumentStore) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.store = store
        self.stash = UserPolicyStash(store=store)

    @service_method(path='policy.get_all', name='get_all')
    def get_all_user_policy(self, context: AuthedServiceContext) -> Union[List[UserPolicy], SyftError]:
        if False:
            while True:
                i = 10
        result = self.stash.get_all(context.credentials)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(path='policy.add', name='add')
    def add_user_policy(self, context: AuthedServiceContext, policy_code: Union[SubmitUserPolicy, UserPolicy]) -> Union[SyftSuccess, SyftError]:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(policy_code, SubmitUserPolicy):
            policy_code = policy_code.to(UserPolicy, context=context)
        result = self.stash.set(context.credentials, policy_code)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message='Policy Code Submitted')

    @service_method(path='policy.get_by_uid', name='get_by_uid')
    def get_policy_by_uid(self, context: AuthedServiceContext, uid: UID) -> Union[SyftSuccess, SyftError]:
        if False:
            for i in range(10):
                print('nop')
        result = self.stash.get_by_uid(context.credentials, uid=uid)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())
TYPE_TO_SERVICE[UserPolicy] = UserPolicy