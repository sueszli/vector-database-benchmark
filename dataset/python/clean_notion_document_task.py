import logging
import time
from typing import List

import click
from celery import shared_task

from core.index.index import IndexBuilder
from extensions.ext_database import db
from models.dataset import DocumentSegment, Dataset, Document


@shared_task(queue='dataset')
def clean_notion_document_task(document_ids: List[str], dataset_id: str):
    """
    Clean document when document deleted.
    :param document_ids: document ids
    :param dataset_id: dataset id

    Usage: clean_notion_document_task.delay(document_ids, dataset_id)
    """
    logging.info(click.style('Start clean document when import form notion document deleted: {}'.format(dataset_id), fg='green'))
    start_at = time.perf_counter()

    try:
        dataset = db.session.query(Dataset).filter(Dataset.id == dataset_id).first()

        if not dataset:
            raise Exception('Document has no dataset')

        vector_index = IndexBuilder.get_index(dataset, 'high_quality')
        kw_index = IndexBuilder.get_index(dataset, 'economy')
        for document_id in document_ids:
            document = db.session.query(Document).filter(
                Document.id == document_id
            ).first()
            db.session.delete(document)

            segments = db.session.query(DocumentSegment).filter(DocumentSegment.document_id == document_id).all()
            index_node_ids = [segment.index_node_id for segment in segments]

            # delete from vector index
            if vector_index:
                vector_index.delete_by_document_id(document_id)

            # delete from keyword index
            if index_node_ids:
                kw_index.delete_by_ids(index_node_ids)

            for segment in segments:
                db.session.delete(segment)
        db.session.commit()
        end_at = time.perf_counter()
        logging.info(
            click.style('Clean document when import form notion document deleted end :: {} latency: {}'.format(
                dataset_id, end_at - start_at),
                        fg='green'))
    except Exception:
        logging.exception("Cleaned document when import form notion document deleted  failed")
