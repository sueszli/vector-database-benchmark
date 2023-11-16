import logging
import time
import click
from celery import shared_task
from langchain.schema import Document
from core.index.index import IndexBuilder
from extensions.ext_database import db
from models.dataset import DocumentSegment, Dataset
from models.dataset import Document as DatasetDocument

@shared_task(queue='dataset')
def deal_dataset_vector_index_task(dataset_id: str, action: str):
    if False:
        print('Hello World!')
    '\n    Async deal dataset from index\n    :param dataset_id: dataset_id\n    :param action: action\n    Usage: deal_dataset_vector_index_task.delay(dataset_id, action)\n    '
    logging.info(click.style('Start deal dataset vector index: {}'.format(dataset_id), fg='green'))
    start_at = time.perf_counter()
    try:
        dataset = Dataset.query.filter_by(id=dataset_id).first()
        if not dataset:
            raise Exception('Dataset not found')
        if action == 'remove':
            index = IndexBuilder.get_index(dataset, 'high_quality', ignore_high_quality_check=True)
            index.delete_by_group_id(dataset.id)
        elif action == 'add':
            dataset_documents = db.session.query(DatasetDocument).filter(DatasetDocument.dataset_id == dataset_id, DatasetDocument.indexing_status == 'completed', DatasetDocument.enabled == True, DatasetDocument.archived == False).all()
            if dataset_documents:
                index = IndexBuilder.get_index(dataset, 'high_quality', ignore_high_quality_check=False)
                documents = []
                for dataset_document in dataset_documents:
                    segments = db.session.query(DocumentSegment).filter(DocumentSegment.document_id == dataset_document.id, DocumentSegment.enabled == True).order_by(DocumentSegment.position.asc()).all()
                    for segment in segments:
                        document = Document(page_content=segment.content, metadata={'doc_id': segment.index_node_id, 'doc_hash': segment.index_node_hash, 'document_id': segment.document_id, 'dataset_id': segment.dataset_id})
                        documents.append(document)
                index.create(documents)
        end_at = time.perf_counter()
        logging.info(click.style('Deal dataset vector index: {} latency: {}'.format(dataset_id, end_at - start_at), fg='green'))
    except Exception:
        logging.exception('Deal dataset vector index failed')