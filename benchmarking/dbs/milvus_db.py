from base_db import BaseDB
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    exceptions,
    utility,
)

class MilvusDB(BaseDB):

    def __init__(self):
        self.dimension = 1024  # >32 #ENV variable
        connections.connect(
        alias="default",
        user='username',
        password='password',
        host='localhost',
        port='19530'
        )
        self.count = 0

    def get_or_create_chunk_collection(self, collection_name: str):
        try:
            connections.connect(alias='default')
            return Collection(collection_name)
        except exceptions.SchemaNotReadyException:
            fields = [
                FieldSchema(
                    name="ID", dtype=DataType.INT64, is_primary=True, auto_id=True
                ),
                FieldSchema(
                    name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=self.dimension
                )
            ]
            return Collection(collection_name, CollectionSchema(fields))

    def insert_item(self, collection, emb):
        collection = self.get_or_create_chunk_collection(collection)
        collection.insert(emb)
        # collection.flush()


    def index(self, collection_name: str):
        collection = self.get_or_create_chunk_collection(collection_name)

        if collection.has_index():
            return

        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        }
        collection.create_index(
            field_name="embeddings", index_params=index_params, index_name="basicindex"
        )

    def query_item(self, collection, emb):
        collection = self.get_or_create_chunk_collection(collection)
        collection.load()
        search_results = collection.search(
            data=[emb],
            anns_field="embeddings",
            param={"metric_type": "L2"},
            limit=5,
            output_fields=["embeddings"],
        )
        return search_results[0]


if __name__ == "__main__":
    db = MilvusDB()
    db.insert_item("test_collection4", [[[1, 2, 3, 4, 5]]])
    db.index("test_collection4")
    print(db.query_item("test_collection4", [1, 2, 3, 4, 5]))
