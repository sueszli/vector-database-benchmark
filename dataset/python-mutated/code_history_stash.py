from typing import List
from typing import Optional
from result import Result
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from .code_history import CodeHistory
NamePartitionKey = PartitionKey(key='service_func_name', type_=str)
VerifyKeyPartitionKey = PartitionKey(key='user_verify_key', type_=SyftVerifyKey)

@serializable()
class CodeHistoryStash(BaseUIDStoreStash):
    object_type = CodeHistory
    settings: PartitionSettings = PartitionSettings(name=CodeHistory.__canonical_name__, object_type=CodeHistory)

    def __init__(self, store: DocumentStore) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(store=store)

    def get_by_service_func_name_and_verify_key(self, credentials: SyftVerifyKey, service_func_name: str, user_verify_key: SyftVerifyKey) -> Result[List[CodeHistory], str]:
        if False:
            print('Hello World!')
        qks = QueryKeys(qks=[NamePartitionKey.with_obj(service_func_name), VerifyKeyPartitionKey.with_obj(user_verify_key)])
        return self.query_one(credentials=credentials, qks=qks)

    def get_by_service_func_name(self, credentials: SyftVerifyKey, service_func_name: str) -> Result[List[CodeHistory], str]:
        if False:
            i = 10
            return i + 15
        qks = QueryKeys(qks=[NamePartitionKey.with_obj(service_func_name)])
        return self.query_all(credentials=credentials, qks=qks)

    def get_by_verify_key(self, credentials: SyftVerifyKey, user_verify_key: SyftVerifyKey) -> Result[Optional[CodeHistory], str]:
        if False:
            while True:
                i = 10
        if isinstance(user_verify_key, str):
            user_verify_key = SyftVerifyKey.from_string(user_verify_key)
        qks = QueryKeys(qks=[VerifyKeyPartitionKey.with_obj(user_verify_key)])
        return self.query_all(credentials=credentials, qks=qks)