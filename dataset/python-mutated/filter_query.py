from __future__ import annotations
import abc
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Generic, List, Optional, TypeVar, Union
from django.db.models import Model, QuerySet
from sentry.services.hybrid_cloud import RpcModel
from sentry.silo import SiloMode
if TYPE_CHECKING:
    from sentry.api.serializers import Serializer
    from sentry.services.hybrid_cloud.auth import AuthenticationContext
    from sentry.services.hybrid_cloud.user import RpcUser
FILTER_ARGS = TypeVar('FILTER_ARGS')
RPC_RESPONSE = TypeVar('RPC_RESPONSE', bound=RpcModel)
SERIALIZER_ENUM = TypeVar('SERIALIZER_ENUM', bound=Union[Enum, None])
BASE_MODEL = TypeVar('BASE_MODEL', bound=Model)
OpaqueSerializedResponse = Any

class FilterQueryDatabaseImpl(Generic[BASE_MODEL, FILTER_ARGS, RPC_RESPONSE, SERIALIZER_ENUM], abc.ABC):

    @abc.abstractmethod
    def base_query(self, ids_only: bool=False) -> QuerySet[BASE_MODEL]:
        if False:
            return 10
        pass

    @abc.abstractmethod
    def filter_arg_validator(self) -> Callable[[FILTER_ARGS], Optional[str]]:
        if False:
            print('Hello World!')
        pass

    @abc.abstractmethod
    def serialize_api(self, serializer: Optional[SERIALIZER_ENUM]) -> Serializer:
        if False:
            print('Hello World!')
        pass

    @abc.abstractmethod
    def apply_filters(self, query: QuerySet[BASE_MODEL], filters: FILTER_ARGS) -> QuerySet[BASE_MODEL]:
        if False:
            for i in range(10):
                print('nop')
        pass

    @abc.abstractmethod
    def serialize_rpc(self, object: BASE_MODEL) -> RPC_RESPONSE:
        if False:
            while True:
                i = 10
        pass

    def _filter_has_any_key_validator(self, *keys: str) -> Callable[[FILTER_ARGS], Optional[str]]:
        if False:
            for i in range(10):
                print('nop')

        def validator(d: FILTER_ARGS) -> Optional[str]:
            if False:
                print('Hello World!')
            for k in keys:
                if k in d:
                    return None
            return f'Filter must contain at least one of: {keys}'
        return validator

    def _query_many(self, filter: FILTER_ARGS, ids_only: bool=False) -> QuerySet:
        if False:
            return 10
        validation_error = self.filter_arg_validator()(filter)
        if validation_error is not None:
            raise TypeError(f'Failed to validate filter arguments passed to {self.__class__.__name__}: {validation_error}')
        query = self.base_query(ids_only=ids_only)
        return self.apply_filters(query, filter)

    def serialize_many(self, filter: FILTER_ARGS, as_user: Optional[RpcUser]=None, auth_context: Optional[AuthenticationContext]=None, serializer: Optional[SERIALIZER_ENUM]=None) -> List[OpaqueSerializedResponse]:
        if False:
            return 10
        from sentry.api.serializers import serialize
        from sentry.services.hybrid_cloud.user import RpcUser
        if as_user is not None and SiloMode.get_current_mode() != SiloMode.MONOLITH:
            if not isinstance(as_user, RpcUser):
                raise TypeError('`as_user` must be serialized first')
        if as_user is None and auth_context:
            as_user = auth_context.user
        result = self._query_many(filter=filter)
        return serialize(list(result), user=as_user, serializer=self.serialize_api(serializer))

    def get_many(self, filter: FILTER_ARGS) -> List[RPC_RESPONSE]:
        if False:
            for i in range(10):
                print('nop')
        return [self.serialize_rpc(o) for o in self._query_many(filter=filter)]

    def get_many_ids(self, filter: FILTER_ARGS) -> List[int]:
        if False:
            for i in range(10):
                print('nop')
        return [o.id for o in self._query_many(filter=filter, ids_only=True)]