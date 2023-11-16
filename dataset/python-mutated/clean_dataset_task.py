import logging
import time
import click
from celery import shared_task
from flask import current_app
from core.index.index import IndexBuilder
from core.index.vector_index.vector_index import VectorIndex
from extensions.ext_database import db
from models.dataset import DocumentSegment, Dataset, DatasetKeywordTable, DatasetQuery, DatasetProcessRule, AppDatasetJoin, Document

@shared_task(queue='dataset')
def clean_dataset_task(dataset_id: str, tenant_id: str, indexing_technique: str, index_struct: str, collection_binding_id: str):
    if False:
        i = 10
        return i + 15
    '\n    Clean dataset when dataset deleted.\n    :param dataset_id: dataset id\n    :param tenant_id: tenant id\n    :param indexing_technique: indexing technique\n    :param index_struct: index struct dict\n    :param collection_binding_id: collection binding id\n\n    Usage: clean_dataset_task.delay(dataset_id, tenant_id, indexing_technique, index_struct)\n    '
    logging.info(click.style('Start clean dataset when dataset deleted: {}'.format(dataset_id), fg='green'))
    start_at = time.perf_counter()
    try:
        dataset = Dataset(id=dataset_id, tenant_id=tenant_id, indexing_technique=indexing_technique, index_struct=index_struct, collection_binding_id=collection_binding_id)
        documents = db.session.query(Document).filter(Document.dataset_id == dataset_id).all()
        segments = db.session.query(DocumentSegment).filter(DocumentSegment.dataset_id == dataset_id).all()
        kw_index = IndexBuilder.get_index(dataset, 'economy')
        if dataset.indexing_technique == 'high_quality':
            vector_index = IndexBuilder.get_default_high_quality_index(dataset)
            try:
                vector_index.delete_by_group_id(dataset.id)
            except Exception:
                logging.exception('Delete doc index failed when dataset deleted.')
        try:
            kw_index.delete()
        except Exception:
            logging.exception('Delete nodes index failed when dataset deleted.')
        for document in documents:
            db.session.delete(document)
        for segment in segments:
            db.session.delete(segment)
        db.session.query(DatasetProcessRule).filter(DatasetProcessRule.dataset_id == dataset_id).delete()
        db.session.query(DatasetQuery).filter(DatasetQuery.dataset_id == dataset_id).delete()
        db.session.query(AppDatasetJoin).filter(AppDatasetJoin.dataset_id == dataset_id).delete()
        db.session.commit()
        end_at = time.perf_counter()
        logging.info(click.style('Cleaned dataset when dataset deleted: {} latency: {}'.format(dataset_id, end_at - start_at), fg='green'))
    except Exception:
        logging.exception('Cleaned dataset when dataset deleted failed')