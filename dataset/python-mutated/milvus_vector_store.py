from core.index.vector_index.milvus import Milvus

class MilvusVectorStore(Milvus):

    def del_texts(self, where_filter: dict):
        if False:
            while True:
                i = 10
        if not where_filter:
            raise ValueError('where_filter must not be empty')
        self.col.delete(where_filter.get('filter'))

    def del_text(self, uuid: str) -> None:
        if False:
            i = 10
            return i + 15
        expr = f'id == {uuid}'
        self.col.delete(expr)

    def text_exists(self, uuid: str) -> bool:
        if False:
            print('Hello World!')
        result = self.col.query(expr=f'metadata["doc_id"] == "{uuid}"', output_fields=['id'])
        return len(result) > 0

    def get_ids_by_document_id(self, document_id: str):
        if False:
            print('Hello World!')
        result = self.col.query(expr=f'metadata["document_id"] == "{document_id}"', output_fields=['id'])
        if result:
            return [item['id'] for item in result]
        else:
            return None

    def get_ids_by_doc_ids(self, doc_ids: list):
        if False:
            print('Hello World!')
        result = self.col.query(expr=f'metadata["doc_id"] in {doc_ids}', output_fields=['id'])
        if result:
            return [item['id'] for item in result]
        else:
            return None

    def delete(self):
        if False:
            return 10
        from pymilvus import utility
        utility.drop_collection(self.collection_name, None, self.alias)