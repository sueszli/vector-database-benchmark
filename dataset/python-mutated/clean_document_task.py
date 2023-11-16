import logging
import time
import click
from celery import shared_task
from core.index.index import IndexBuilder
from extensions.ext_database import db
from models.dataset import DocumentSegment, Dataset

@shared_task(queue='dataset')
def clean_document_task(document_id: str, dataset_id: str):
    if False:
        while True:
            i = 10
    '\n    Clean document when document deleted.\n    :param document_id: document id\n    :param dataset_id: dataset id\n\n    Usage: clean_document_task.delay(document_id, dataset_id)\n    '
    logging.info(click.style('Start clean document when document deleted: {}'.format(document_id), fg='green'))
    start_at = time.perf_counter()
    try:
        dataset = db.session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise Exception('Document has no dataset')
        vector_index = IndexBuilder.get_index(dataset, 'high_quality')
        kw_index = IndexBuilder.get_index(dataset, 'economy')
        segments = db.session.query(DocumentSegment).filter(DocumentSegment.document_id == document_id).all()
        if segments:
            index_node_ids = [segment.index_node_id for segment in segments]
            if vector_index:
                vector_index.delete_by_document_id(document_id)
            if index_node_ids:
                kw_index.delete_by_ids(index_node_ids)
            for segment in segments:
                db.session.delete(segment)
            db.session.commit()
            end_at = time.perf_counter()
            logging.info(click.style('Cleaned document when document deleted: {} latency: {}'.format(document_id, end_at - start_at), fg='green'))
    except Exception:
        logging.exception('Cleaned document when document deleted failed')