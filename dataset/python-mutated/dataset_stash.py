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
from ...types.uid import UID
from ...util.telemetry import instrument
from .dataset import Dataset
from .dataset import DatasetUpdate
NamePartitionKey = PartitionKey(key='name', type_=str)
ActionIDsPartitionKey = PartitionKey(key='action_ids', type_=List[UID])

@instrument
@serializable()
class DatasetStash(BaseUIDStoreStash):
    object_type = Dataset
    settings: PartitionSettings = PartitionSettings(name=Dataset.__canonical_name__, object_type=Dataset)

    def __init__(self, store: DocumentStore) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(store=store)

    def get_by_name(self, credentials: SyftVerifyKey, name: str) -> Result[Optional[Dataset], str]:
        if False:
            print('Hello World!')
        qks = QueryKeys(qks=[NamePartitionKey.with_obj(name)])
        return self.query_one(credentials=credentials, qks=qks)

    def update(self, credentials: SyftVerifyKey, dataset_update: DatasetUpdate) -> Result[Dataset, str]:
        if False:
            while True:
                i = 10
        res = self.check_type(dataset_update, DatasetUpdate)
        if res.is_err():
            return res
        return super().update(credentials=credentials, obj=res.ok())

    def search_action_ids(self, credentials: SyftVerifyKey, uid: UID) -> Result[List[Dataset], str]:
        if False:
            return 10
        qks = QueryKeys(qks=[ActionIDsPartitionKey.with_obj(uid)])
        return self.query_all(credentials=credentials, qks=qks)