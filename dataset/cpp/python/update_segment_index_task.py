import datetime
import logging
import time
from typing import List, Optional

import click
from celery import shared_task
from langchain.schema import Document
from werkzeug.exceptions import NotFound

from core.index.index import IndexBuilder
from extensions.ext_database import db
from extensions.ext_redis import redis_client
from models.dataset import DocumentSegment


@shared_task(queue='dataset')
def update_segment_index_task(segment_id: str, keywords: Optional[List[str]] = None):
    """
    Async update segment index
    :param segment_id:
    :param keywords:
    Usage: update_segment_index_task.delay(segment_id)
    """
    logging.info(click.style('Start update segment index: {}'.format(segment_id), fg='green'))
    start_at = time.perf_counter()

    segment = db.session.query(DocumentSegment).filter(DocumentSegment.id == segment_id).first()
    if not segment:
        raise NotFound('Segment not found')

    if segment.status != 'updating':
        return

    indexing_cache_key = 'segment_{}_indexing'.format(segment.id)

    try:
        dataset = segment.dataset

        if not dataset:
            logging.info(click.style('Segment {} has no dataset, pass.'.format(segment.id), fg='cyan'))
            return

        dataset_document = segment.document

        if not dataset_document:
            logging.info(click.style('Segment {} has no document, pass.'.format(segment.id), fg='cyan'))
            return

        if not dataset_document.enabled or dataset_document.archived or dataset_document.indexing_status != 'completed':
            logging.info(click.style('Segment {} document status is invalid, pass.'.format(segment.id), fg='cyan'))
            return

        # update segment status to indexing
        update_params = {
            DocumentSegment.status: "indexing",
            DocumentSegment.indexing_at: datetime.datetime.utcnow()
        }
        DocumentSegment.query.filter_by(id=segment.id).update(update_params)
        db.session.commit()

        vector_index = IndexBuilder.get_index(dataset, 'high_quality')
        kw_index = IndexBuilder.get_index(dataset, 'economy')

        # delete from vector index
        if vector_index:
            vector_index.delete_by_ids([segment.index_node_id])

        # delete from keyword index
        kw_index.delete_by_ids([segment.index_node_id])

        # add new index
        document = Document(
            page_content=segment.content,
            metadata={
                "doc_id": segment.index_node_id,
                "doc_hash": segment.index_node_hash,
                "document_id": segment.document_id,
                "dataset_id": segment.dataset_id,
            }
        )

        # save vector index
        index = IndexBuilder.get_index(dataset, 'high_quality')
        if index:
            index.add_texts([document], duplicate_check=True)

        # save keyword index
        index = IndexBuilder.get_index(dataset, 'economy')
        if index:
            if keywords and len(keywords) > 0:
                index.create_segment_keywords(segment.index_node_id, keywords)
            else:
                index.add_texts([document])

        # update segment to completed
        update_params = {
            DocumentSegment.status: "completed",
            DocumentSegment.completed_at: datetime.datetime.utcnow()
        }
        DocumentSegment.query.filter_by(id=segment.id).update(update_params)
        db.session.commit()

        end_at = time.perf_counter()
        logging.info(click.style('Segment update index: {} latency: {}'.format(segment.id, end_at - start_at), fg='green'))
    except Exception as e:
        logging.exception("update segment index failed")
        segment.enabled = False
        segment.disabled_at = datetime.datetime.utcnow()
        segment.status = 'error'
        segment.error = str(e)
        db.session.commit()
    finally:
        redis_client.delete(indexing_cache_key)
