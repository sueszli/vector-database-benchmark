import uuid
from datetime import datetime
from flask import request
from flask_login import current_user
from flask_restful import Resource, reqparse, marshal
from werkzeug.exceptions import NotFound, Forbidden
import services
from controllers.console import api
from controllers.console.app.error import ProviderNotInitializeError
from controllers.console.datasets.error import InvalidActionError, NoFileUploadedError, TooManyFilesError
from controllers.console.setup import setup_required
from controllers.console.wraps import account_initialization_required
from core.model_providers.error import LLMBadRequestError, ProviderTokenNotInitError
from core.model_providers.model_factory import ModelFactory
from libs.login import login_required
from extensions.ext_database import db
from extensions.ext_redis import redis_client
from fields.segment_fields import segment_fields
from models.dataset import DocumentSegment
from services.dataset_service import DatasetService, DocumentService, SegmentService
from tasks.enable_segment_to_index_task import enable_segment_to_index_task
from tasks.disable_segment_from_index_task import disable_segment_from_index_task
from tasks.batch_create_segment_to_index_task import batch_create_segment_to_index_task
import pandas as pd

class DatasetDocumentSegmentListApi(Resource):

    @setup_required
    @login_required
    @account_initialization_required
    def get(self, dataset_id, document_id):
        if False:
            for i in range(10):
                print('nop')
        dataset_id = str(dataset_id)
        document_id = str(document_id)
        dataset = DatasetService.get_dataset(dataset_id)
        if not dataset:
            raise NotFound('Dataset not found.')
        try:
            DatasetService.check_dataset_permission(dataset, current_user)
        except services.errors.account.NoPermissionError as e:
            raise Forbidden(str(e))
        document = DocumentService.get_document(dataset_id, document_id)
        if not document:
            raise NotFound('Document not found.')
        parser = reqparse.RequestParser()
        parser.add_argument('last_id', type=str, default=None, location='args')
        parser.add_argument('limit', type=int, default=20, location='args')
        parser.add_argument('status', type=str, action='append', default=[], location='args')
        parser.add_argument('hit_count_gte', type=int, default=None, location='args')
        parser.add_argument('enabled', type=str, default='all', location='args')
        parser.add_argument('keyword', type=str, default=None, location='args')
        args = parser.parse_args()
        last_id = args['last_id']
        limit = min(args['limit'], 100)
        status_list = args['status']
        hit_count_gte = args['hit_count_gte']
        keyword = args['keyword']
        query = DocumentSegment.query.filter(DocumentSegment.document_id == str(document_id), DocumentSegment.tenant_id == current_user.current_tenant_id)
        if last_id is not None:
            last_segment = DocumentSegment.query.get(str(last_id))
            if last_segment:
                query = query.filter(DocumentSegment.position > last_segment.position)
            else:
                return ({'data': [], 'has_more': False, 'limit': limit}, 200)
        if status_list:
            query = query.filter(DocumentSegment.status.in_(status_list))
        if hit_count_gte is not None:
            query = query.filter(DocumentSegment.hit_count >= hit_count_gte)
        if keyword:
            query = query.where(DocumentSegment.content.ilike(f'%{keyword}%'))
        if args['enabled'].lower() != 'all':
            if args['enabled'].lower() == 'true':
                query = query.filter(DocumentSegment.enabled == True)
            elif args['enabled'].lower() == 'false':
                query = query.filter(DocumentSegment.enabled == False)
        total = query.count()
        segments = query.order_by(DocumentSegment.position).limit(limit + 1).all()
        has_more = False
        if len(segments) > limit:
            has_more = True
            segments = segments[:-1]
        return ({'data': marshal(segments, segment_fields), 'doc_form': document.doc_form, 'has_more': has_more, 'limit': limit, 'total': total}, 200)

class DatasetDocumentSegmentApi(Resource):

    @setup_required
    @login_required
    @account_initialization_required
    def patch(self, dataset_id, segment_id, action):
        if False:
            return 10
        dataset_id = str(dataset_id)
        dataset = DatasetService.get_dataset(dataset_id)
        if not dataset:
            raise NotFound('Dataset not found.')
        DatasetService.check_dataset_model_setting(dataset)
        if current_user.current_tenant.current_role not in ['admin', 'owner']:
            raise Forbidden()
        try:
            DatasetService.check_dataset_permission(dataset, current_user)
        except services.errors.account.NoPermissionError as e:
            raise Forbidden(str(e))
        if dataset.indexing_technique == 'high_quality':
            try:
                ModelFactory.get_embedding_model(tenant_id=current_user.current_tenant_id, model_provider_name=dataset.embedding_model_provider, model_name=dataset.embedding_model)
            except LLMBadRequestError:
                raise ProviderNotInitializeError(f'No Embedding Model available. Please configure a valid provider in the Settings -> Model Provider.')
            except ProviderTokenNotInitError as ex:
                raise ProviderNotInitializeError(ex.description)
        segment = DocumentSegment.query.filter(DocumentSegment.id == str(segment_id), DocumentSegment.tenant_id == current_user.current_tenant_id).first()
        if not segment:
            raise NotFound('Segment not found.')
        document_indexing_cache_key = 'document_{}_indexing'.format(segment.document_id)
        cache_result = redis_client.get(document_indexing_cache_key)
        if cache_result is not None:
            raise InvalidActionError('Document is being indexed, please try again later')
        indexing_cache_key = 'segment_{}_indexing'.format(segment.id)
        cache_result = redis_client.get(indexing_cache_key)
        if cache_result is not None:
            raise InvalidActionError('Segment is being indexed, please try again later')
        if action == 'enable':
            if segment.enabled:
                raise InvalidActionError('Segment is already enabled.')
            segment.enabled = True
            segment.disabled_at = None
            segment.disabled_by = None
            db.session.commit()
            redis_client.setex(indexing_cache_key, 600, 1)
            enable_segment_to_index_task.delay(segment.id)
            return ({'result': 'success'}, 200)
        elif action == 'disable':
            if not segment.enabled:
                raise InvalidActionError('Segment is already disabled.')
            segment.enabled = False
            segment.disabled_at = datetime.utcnow()
            segment.disabled_by = current_user.id
            db.session.commit()
            redis_client.setex(indexing_cache_key, 600, 1)
            disable_segment_from_index_task.delay(segment.id)
            return ({'result': 'success'}, 200)
        else:
            raise InvalidActionError()

class DatasetDocumentSegmentAddApi(Resource):

    @setup_required
    @login_required
    @account_initialization_required
    def post(self, dataset_id, document_id):
        if False:
            i = 10
            return i + 15
        dataset_id = str(dataset_id)
        dataset = DatasetService.get_dataset(dataset_id)
        if not dataset:
            raise NotFound('Dataset not found.')
        document_id = str(document_id)
        document = DocumentService.get_document(dataset_id, document_id)
        if not document:
            raise NotFound('Document not found.')
        if current_user.current_tenant.current_role not in ['admin', 'owner']:
            raise Forbidden()
        if dataset.indexing_technique == 'high_quality':
            try:
                ModelFactory.get_embedding_model(tenant_id=current_user.current_tenant_id, model_provider_name=dataset.embedding_model_provider, model_name=dataset.embedding_model)
            except LLMBadRequestError:
                raise ProviderNotInitializeError(f'No Embedding Model available. Please configure a valid provider in the Settings -> Model Provider.')
            except ProviderTokenNotInitError as ex:
                raise ProviderNotInitializeError(ex.description)
        try:
            DatasetService.check_dataset_permission(dataset, current_user)
        except services.errors.account.NoPermissionError as e:
            raise Forbidden(str(e))
        parser = reqparse.RequestParser()
        parser.add_argument('content', type=str, required=True, nullable=False, location='json')
        parser.add_argument('answer', type=str, required=False, nullable=True, location='json')
        parser.add_argument('keywords', type=list, required=False, nullable=True, location='json')
        args = parser.parse_args()
        SegmentService.segment_create_args_validate(args, document)
        segment = SegmentService.create_segment(args, document, dataset)
        return ({'data': marshal(segment, segment_fields), 'doc_form': document.doc_form}, 200)

class DatasetDocumentSegmentUpdateApi(Resource):

    @setup_required
    @login_required
    @account_initialization_required
    def patch(self, dataset_id, document_id, segment_id):
        if False:
            while True:
                i = 10
        dataset_id = str(dataset_id)
        dataset = DatasetService.get_dataset(dataset_id)
        if not dataset:
            raise NotFound('Dataset not found.')
        DatasetService.check_dataset_model_setting(dataset)
        document_id = str(document_id)
        document = DocumentService.get_document(dataset_id, document_id)
        if not document:
            raise NotFound('Document not found.')
        if dataset.indexing_technique == 'high_quality':
            try:
                ModelFactory.get_embedding_model(tenant_id=current_user.current_tenant_id, model_provider_name=dataset.embedding_model_provider, model_name=dataset.embedding_model)
            except LLMBadRequestError:
                raise ProviderNotInitializeError(f'No Embedding Model available. Please configure a valid provider in the Settings -> Model Provider.')
            except ProviderTokenNotInitError as ex:
                raise ProviderNotInitializeError(ex.description)
        segment_id = str(segment_id)
        segment = DocumentSegment.query.filter(DocumentSegment.id == str(segment_id), DocumentSegment.tenant_id == current_user.current_tenant_id).first()
        if not segment:
            raise NotFound('Segment not found.')
        if current_user.current_tenant.current_role not in ['admin', 'owner']:
            raise Forbidden()
        try:
            DatasetService.check_dataset_permission(dataset, current_user)
        except services.errors.account.NoPermissionError as e:
            raise Forbidden(str(e))
        parser = reqparse.RequestParser()
        parser.add_argument('content', type=str, required=True, nullable=False, location='json')
        parser.add_argument('answer', type=str, required=False, nullable=True, location='json')
        parser.add_argument('keywords', type=list, required=False, nullable=True, location='json')
        args = parser.parse_args()
        SegmentService.segment_create_args_validate(args, document)
        segment = SegmentService.update_segment(args, segment, document, dataset)
        return ({'data': marshal(segment, segment_fields), 'doc_form': document.doc_form}, 200)

    @setup_required
    @login_required
    @account_initialization_required
    def delete(self, dataset_id, document_id, segment_id):
        if False:
            print('Hello World!')
        dataset_id = str(dataset_id)
        dataset = DatasetService.get_dataset(dataset_id)
        if not dataset:
            raise NotFound('Dataset not found.')
        DatasetService.check_dataset_model_setting(dataset)
        document_id = str(document_id)
        document = DocumentService.get_document(dataset_id, document_id)
        if not document:
            raise NotFound('Document not found.')
        segment_id = str(segment_id)
        segment = DocumentSegment.query.filter(DocumentSegment.id == str(segment_id), DocumentSegment.tenant_id == current_user.current_tenant_id).first()
        if not segment:
            raise NotFound('Segment not found.')
        if current_user.current_tenant.current_role not in ['admin', 'owner']:
            raise Forbidden()
        try:
            DatasetService.check_dataset_permission(dataset, current_user)
        except services.errors.account.NoPermissionError as e:
            raise Forbidden(str(e))
        SegmentService.delete_segment(segment, document, dataset)
        return ({'result': 'success'}, 200)

class DatasetDocumentSegmentBatchImportApi(Resource):

    @setup_required
    @login_required
    @account_initialization_required
    def post(self, dataset_id, document_id):
        if False:
            while True:
                i = 10
        dataset_id = str(dataset_id)
        dataset = DatasetService.get_dataset(dataset_id)
        if not dataset:
            raise NotFound('Dataset not found.')
        document_id = str(document_id)
        document = DocumentService.get_document(dataset_id, document_id)
        if not document:
            raise NotFound('Document not found.')
        file = request.files['file']
        if 'file' not in request.files:
            raise NoFileUploadedError()
        if len(request.files) > 1:
            raise TooManyFilesError()
        if not file.filename.endswith('.csv'):
            raise ValueError('Invalid file type. Only CSV files are allowed')
        try:
            df = pd.read_csv(file)
            result = []
            for (index, row) in df.iterrows():
                if document.doc_form == 'qa_model':
                    data = {'content': row[0], 'answer': row[1]}
                else:
                    data = {'content': row[0]}
                result.append(data)
            if len(result) == 0:
                raise ValueError('The CSV file is empty.')
            job_id = str(uuid.uuid4())
            indexing_cache_key = 'segment_batch_import_{}'.format(str(job_id))
            redis_client.setnx(indexing_cache_key, 'waiting')
            batch_create_segment_to_index_task.delay(str(job_id), result, dataset_id, document_id, current_user.current_tenant_id, current_user.id)
        except Exception as e:
            return ({'error': str(e)}, 500)
        return ({'job_id': job_id, 'job_status': 'waiting'}, 200)

    @setup_required
    @login_required
    @account_initialization_required
    def get(self, job_id):
        if False:
            print('Hello World!')
        job_id = str(job_id)
        indexing_cache_key = 'segment_batch_import_{}'.format(job_id)
        cache_result = redis_client.get(indexing_cache_key)
        if cache_result is None:
            raise ValueError('The job is not exist.')
        return ({'job_id': job_id, 'job_status': cache_result.decode()}, 200)
api.add_resource(DatasetDocumentSegmentListApi, '/datasets/<uuid:dataset_id>/documents/<uuid:document_id>/segments')
api.add_resource(DatasetDocumentSegmentApi, '/datasets/<uuid:dataset_id>/segments/<uuid:segment_id>/<string:action>')
api.add_resource(DatasetDocumentSegmentAddApi, '/datasets/<uuid:dataset_id>/documents/<uuid:document_id>/segment')
api.add_resource(DatasetDocumentSegmentUpdateApi, '/datasets/<uuid:dataset_id>/documents/<uuid:document_id>/segments/<uuid:segment_id>')
api.add_resource(DatasetDocumentSegmentBatchImportApi, '/datasets/<uuid:dataset_id>/documents/<uuid:document_id>/segments/batch_import', '/datasets/batch_import_status/<uuid:job_id>')