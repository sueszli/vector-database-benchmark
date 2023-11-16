from typing import Optional, List
from langchain.schema import Document
from core.index.index import IndexBuilder
from models.dataset import Dataset, DocumentSegment

class VectorService:

    @classmethod
    def create_segment_vector(cls, keywords: Optional[List[str]], segment: DocumentSegment, dataset: Dataset):
        if False:
            for i in range(10):
                print('nop')
        document = Document(page_content=segment.content, metadata={'doc_id': segment.index_node_id, 'doc_hash': segment.index_node_hash, 'document_id': segment.document_id, 'dataset_id': segment.dataset_id})
        index = IndexBuilder.get_index(dataset, 'high_quality')
        if index:
            index.add_texts([document], duplicate_check=True)
        index = IndexBuilder.get_index(dataset, 'economy')
        if index:
            if keywords and len(keywords) > 0:
                index.create_segment_keywords(segment.index_node_id, keywords)
            else:
                index.add_texts([document])

    @classmethod
    def multi_create_segment_vector(cls, pre_segment_data_list: list, dataset: Dataset):
        if False:
            while True:
                i = 10
        documents = []
        for pre_segment_data in pre_segment_data_list:
            segment = pre_segment_data['segment']
            document = Document(page_content=segment.content, metadata={'doc_id': segment.index_node_id, 'doc_hash': segment.index_node_hash, 'document_id': segment.document_id, 'dataset_id': segment.dataset_id})
            documents.append(document)
        index = IndexBuilder.get_index(dataset, 'high_quality')
        if index:
            index.add_texts(documents, duplicate_check=True)
        keyword_index = IndexBuilder.get_index(dataset, 'economy')
        if keyword_index:
            keyword_index.multi_create_segment_keywords(pre_segment_data_list)

    @classmethod
    def update_segment_vector(cls, keywords: Optional[List[str]], segment: DocumentSegment, dataset: Dataset):
        if False:
            for i in range(10):
                print('nop')
        vector_index = IndexBuilder.get_index(dataset, 'high_quality')
        kw_index = IndexBuilder.get_index(dataset, 'economy')
        if vector_index:
            vector_index.delete_by_ids([segment.index_node_id])
        kw_index.delete_by_ids([segment.index_node_id])
        document = Document(page_content=segment.content, metadata={'doc_id': segment.index_node_id, 'doc_hash': segment.index_node_hash, 'document_id': segment.document_id, 'dataset_id': segment.dataset_id})
        if vector_index:
            vector_index.add_texts([document], duplicate_check=True)
        if keywords and len(keywords) > 0:
            kw_index.create_segment_keywords(segment.index_node_id, keywords)
        else:
            kw_index.add_texts([document])