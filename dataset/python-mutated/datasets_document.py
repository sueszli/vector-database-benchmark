from datetime import datetime
from typing import List
from flask import request, current_app
from flask_login import current_user
from libs.login import login_required
from flask_restful import Resource, fields, marshal, marshal_with, reqparse
from sqlalchemy import desc, asc
from werkzeug.exceptions import NotFound, Forbidden
import services
from controllers.console import api
from controllers.console.app.error import ProviderNotInitializeError, ProviderQuotaExceededError, ProviderModelCurrentlyNotSupportError
from controllers.console.datasets.error import DocumentAlreadyFinishedError, InvalidActionError, DocumentIndexingError, InvalidMetadataError, ArchivedDocumentImmutableError
from controllers.console.setup import setup_required
from controllers.console.wraps import account_initialization_required
from core.indexing_runner import IndexingRunner
from core.model_providers.error import ProviderTokenNotInitError, QuotaExceededError, ModelCurrentlyNotSupportError, LLMBadRequestError
from core.model_providers.model_factory import ModelFactory
from extensions.ext_redis import redis_client
from fields.document_fields import document_with_segments_fields, document_fields, dataset_and_document_fields, document_status_fields
from extensions.ext_database import db
from models.dataset import DatasetProcessRule, Dataset
from models.dataset import Document, DocumentSegment
from models.model import UploadFile
from services.dataset_service import DocumentService, DatasetService
from tasks.add_document_to_index_task import add_document_to_index_task
from tasks.remove_document_from_index_task import remove_document_from_index_task

class DocumentResource(Resource):

    def get_document(self, dataset_id: str, document_id: str) -> Document:
        if False:
            i = 10
            return i + 15
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
        if document.tenant_id != current_user.current_tenant_id:
            raise Forbidden('No permission.')
        return document

    def get_batch_documents(self, dataset_id: str, batch: str) -> List[Document]:
        if False:
            print('Hello World!')
        dataset = DatasetService.get_dataset(dataset_id)
        if not dataset:
            raise NotFound('Dataset not found.')
        try:
            DatasetService.check_dataset_permission(dataset, current_user)
        except services.errors.account.NoPermissionError as e:
            raise Forbidden(str(e))
        documents = DocumentService.get_batch_documents(dataset_id, batch)
        if not documents:
            raise NotFound('Documents not found.')
        return documents

class GetProcessRuleApi(Resource):

    @setup_required
    @login_required
    @account_initialization_required
    def get(self):
        if False:
            while True:
                i = 10
        req_data = request.args
        document_id = req_data.get('document_id')
        mode = DocumentService.DEFAULT_RULES['mode']
        rules = DocumentService.DEFAULT_RULES['rules']
        if document_id:
            document = Document.query.get_or_404(document_id)
            dataset = DatasetService.get_dataset(document.dataset_id)
            if not dataset:
                raise NotFound('Dataset not found.')
            try:
                DatasetService.check_dataset_permission(dataset, current_user)
            except services.errors.account.NoPermissionError as e:
                raise Forbidden(str(e))
            dataset_process_rule = db.session.query(DatasetProcessRule).filter(DatasetProcessRule.dataset_id == document.dataset_id).order_by(DatasetProcessRule.created_at.desc()).limit(1).one_or_none()
            if dataset_process_rule:
                mode = dataset_process_rule.mode
                rules = dataset_process_rule.rules_dict
        return {'mode': mode, 'rules': rules}

class DatasetDocumentListApi(Resource):

    @setup_required
    @login_required
    @account_initialization_required
    def get(self, dataset_id):
        if False:
            print('Hello World!')
        dataset_id = str(dataset_id)
        page = request.args.get('page', default=1, type=int)
        limit = request.args.get('limit', default=20, type=int)
        search = request.args.get('keyword', default=None, type=str)
        sort = request.args.get('sort', default='-created_at', type=str)
        fetch = request.args.get('fetch', default=False, type=bool)
        dataset = DatasetService.get_dataset(dataset_id)
        if not dataset:
            raise NotFound('Dataset not found.')
        try:
            DatasetService.check_dataset_permission(dataset, current_user)
        except services.errors.account.NoPermissionError as e:
            raise Forbidden(str(e))
        query = Document.query.filter_by(dataset_id=str(dataset_id), tenant_id=current_user.current_tenant_id)
        if search:
            search = f'%{search}%'
            query = query.filter(Document.name.like(search))
        if sort.startswith('-'):
            sort_logic = desc
            sort = sort[1:]
        else:
            sort_logic = asc
        if sort == 'hit_count':
            sub_query = db.select(DocumentSegment.document_id, db.func.sum(DocumentSegment.hit_count).label('total_hit_count')).group_by(DocumentSegment.document_id).subquery()
            query = query.outerjoin(sub_query, sub_query.c.document_id == Document.id).order_by(sort_logic(db.func.coalesce(sub_query.c.total_hit_count, 0)))
        elif sort == 'created_at':
            query = query.order_by(sort_logic(Document.created_at))
        else:
            query = query.order_by(desc(Document.created_at))
        paginated_documents = query.paginate(page=page, per_page=limit, max_per_page=100, error_out=False)
        documents = paginated_documents.items
        if fetch:
            for document in documents:
                completed_segments = DocumentSegment.query.filter(DocumentSegment.completed_at.isnot(None), DocumentSegment.document_id == str(document.id), DocumentSegment.status != 're_segment').count()
                total_segments = DocumentSegment.query.filter(DocumentSegment.document_id == str(document.id), DocumentSegment.status != 're_segment').count()
                document.completed_segments = completed_segments
                document.total_segments = total_segments
            data = marshal(documents, document_with_segments_fields)
        else:
            data = marshal(documents, document_fields)
        response = {'data': data, 'has_more': len(documents) == limit, 'limit': limit, 'total': paginated_documents.total, 'page': page}
        return response
    documents_and_batch_fields = {'documents': fields.List(fields.Nested(document_fields)), 'batch': fields.String}

    @setup_required
    @login_required
    @account_initialization_required
    @marshal_with(documents_and_batch_fields)
    def post(self, dataset_id):
        if False:
            for i in range(10):
                print('nop')
        dataset_id = str(dataset_id)
        dataset = DatasetService.get_dataset(dataset_id)
        if not dataset:
            raise NotFound('Dataset not found.')
        if current_user.current_tenant.current_role not in ['admin', 'owner']:
            raise Forbidden()
        try:
            DatasetService.check_dataset_permission(dataset, current_user)
        except services.errors.account.NoPermissionError as e:
            raise Forbidden(str(e))
        parser = reqparse.RequestParser()
        parser.add_argument('indexing_technique', type=str, choices=Dataset.INDEXING_TECHNIQUE_LIST, nullable=False, location='json')
        parser.add_argument('data_source', type=dict, required=False, location='json')
        parser.add_argument('process_rule', type=dict, required=False, location='json')
        parser.add_argument('duplicate', type=bool, nullable=False, location='json')
        parser.add_argument('original_document_id', type=str, required=False, location='json')
        parser.add_argument('doc_form', type=str, default='text_model', required=False, nullable=False, location='json')
        parser.add_argument('doc_language', type=str, default='English', required=False, nullable=False, location='json')
        args = parser.parse_args()
        if not dataset.indexing_technique and (not args['indexing_technique']):
            raise ValueError('indexing_technique is required.')
        DocumentService.document_create_args_validate(args)
        try:
            (documents, batch) = DocumentService.save_document_with_dataset_id(dataset, args, current_user)
        except ProviderTokenNotInitError as ex:
            raise ProviderNotInitializeError(ex.description)
        except QuotaExceededError:
            raise ProviderQuotaExceededError()
        except ModelCurrentlyNotSupportError:
            raise ProviderModelCurrentlyNotSupportError()
        return {'documents': documents, 'batch': batch}

class DatasetInitApi(Resource):

    @setup_required
    @login_required
    @account_initialization_required
    @marshal_with(dataset_and_document_fields)
    def post(self):
        if False:
            while True:
                i = 10
        if current_user.current_tenant.current_role not in ['admin', 'owner']:
            raise Forbidden()
        parser = reqparse.RequestParser()
        parser.add_argument('indexing_technique', type=str, choices=Dataset.INDEXING_TECHNIQUE_LIST, required=True, nullable=False, location='json')
        parser.add_argument('data_source', type=dict, required=True, nullable=True, location='json')
        parser.add_argument('process_rule', type=dict, required=True, nullable=True, location='json')
        parser.add_argument('doc_form', type=str, default='text_model', required=False, nullable=False, location='json')
        parser.add_argument('doc_language', type=str, default='English', required=False, nullable=False, location='json')
        args = parser.parse_args()
        if args['indexing_technique'] == 'high_quality':
            try:
                ModelFactory.get_embedding_model(tenant_id=current_user.current_tenant_id)
            except LLMBadRequestError:
                raise ProviderNotInitializeError(f'No Embedding Model available. Please configure a valid provider in the Settings -> Model Provider.')
            except ProviderTokenNotInitError as ex:
                raise ProviderNotInitializeError(ex.description)
        DocumentService.document_create_args_validate(args)
        try:
            (dataset, documents, batch) = DocumentService.save_document_without_dataset_id(tenant_id=current_user.current_tenant_id, document_data=args, account=current_user)
        except ProviderTokenNotInitError as ex:
            raise ProviderNotInitializeError(ex.description)
        except QuotaExceededError:
            raise ProviderQuotaExceededError()
        except ModelCurrentlyNotSupportError:
            raise ProviderModelCurrentlyNotSupportError()
        response = {'dataset': dataset, 'documents': documents, 'batch': batch}
        return response

class DocumentIndexingEstimateApi(DocumentResource):

    @setup_required
    @login_required
    @account_initialization_required
    def get(self, dataset_id, document_id):
        if False:
            i = 10
            return i + 15
        dataset_id = str(dataset_id)
        document_id = str(document_id)
        document = self.get_document(dataset_id, document_id)
        if document.indexing_status in ['completed', 'error']:
            raise DocumentAlreadyFinishedError()
        data_process_rule = document.dataset_process_rule
        data_process_rule_dict = data_process_rule.to_dict()
        response = {'tokens': 0, 'total_price': 0, 'currency': 'USD', 'total_segments': 0, 'preview': []}
        if document.data_source_type == 'upload_file':
            data_source_info = document.data_source_info_dict
            if data_source_info and 'upload_file_id' in data_source_info:
                file_id = data_source_info['upload_file_id']
                file = db.session.query(UploadFile).filter(UploadFile.tenant_id == document.tenant_id, UploadFile.id == file_id).first()
                if not file:
                    raise NotFound('File not found.')
                indexing_runner = IndexingRunner()
                try:
                    response = indexing_runner.file_indexing_estimate(current_user.current_tenant_id, [file], data_process_rule_dict, None, 'English', dataset_id)
                except LLMBadRequestError:
                    raise ProviderNotInitializeError(f'No Embedding Model available. Please configure a valid provider in the Settings -> Model Provider.')
                except ProviderTokenNotInitError as ex:
                    raise ProviderNotInitializeError(ex.description)
        return response

class DocumentBatchIndexingEstimateApi(DocumentResource):

    @setup_required
    @login_required
    @account_initialization_required
    def get(self, dataset_id, batch):
        if False:
            while True:
                i = 10
        dataset_id = str(dataset_id)
        batch = str(batch)
        dataset = DatasetService.get_dataset(dataset_id)
        if dataset is None:
            raise NotFound('Dataset not found.')
        documents = self.get_batch_documents(dataset_id, batch)
        response = {'tokens': 0, 'total_price': 0, 'currency': 'USD', 'total_segments': 0, 'preview': []}
        if not documents:
            return response
        data_process_rule = documents[0].dataset_process_rule
        data_process_rule_dict = data_process_rule.to_dict()
        info_list = []
        for document in documents:
            if document.indexing_status in ['completed', 'error']:
                raise DocumentAlreadyFinishedError()
            data_source_info = document.data_source_info_dict
            if data_source_info and 'upload_file_id' in data_source_info:
                file_id = data_source_info['upload_file_id']
                info_list.append(file_id)
            elif data_source_info and 'notion_workspace_id' in data_source_info and ('notion_page_id' in data_source_info):
                pages = []
                page = {'page_id': data_source_info['notion_page_id'], 'type': data_source_info['type']}
                pages.append(page)
                notion_info = {'workspace_id': data_source_info['notion_workspace_id'], 'pages': pages}
                info_list.append(notion_info)
        if dataset.data_source_type == 'upload_file':
            file_details = db.session.query(UploadFile).filter(UploadFile.tenant_id == current_user.current_tenant_id, UploadFile.id in info_list).all()
            if file_details is None:
                raise NotFound('File not found.')
            indexing_runner = IndexingRunner()
            try:
                response = indexing_runner.file_indexing_estimate(current_user.current_tenant_id, file_details, data_process_rule_dict, None, 'English', dataset_id)
            except LLMBadRequestError:
                raise ProviderNotInitializeError(f'No Embedding Model available. Please configure a valid provider in the Settings -> Model Provider.')
            except ProviderTokenNotInitError as ex:
                raise ProviderNotInitializeError(ex.description)
        elif dataset.data_source_type == 'notion_import':
            indexing_runner = IndexingRunner()
            try:
                response = indexing_runner.notion_indexing_estimate(current_user.current_tenant_id, info_list, data_process_rule_dict, None, 'English', dataset_id)
            except LLMBadRequestError:
                raise ProviderNotInitializeError(f'No Embedding Model available. Please configure a valid provider in the Settings -> Model Provider.')
            except ProviderTokenNotInitError as ex:
                raise ProviderNotInitializeError(ex.description)
        else:
            raise ValueError('Data source type not support')
        return response

class DocumentBatchIndexingStatusApi(DocumentResource):

    @setup_required
    @login_required
    @account_initialization_required
    def get(self, dataset_id, batch):
        if False:
            print('Hello World!')
        dataset_id = str(dataset_id)
        batch = str(batch)
        documents = self.get_batch_documents(dataset_id, batch)
        documents_status = []
        for document in documents:
            completed_segments = DocumentSegment.query.filter(DocumentSegment.completed_at.isnot(None), DocumentSegment.document_id == str(document.id), DocumentSegment.status != 're_segment').count()
            total_segments = DocumentSegment.query.filter(DocumentSegment.document_id == str(document.id), DocumentSegment.status != 're_segment').count()
            document.completed_segments = completed_segments
            document.total_segments = total_segments
            if document.is_paused:
                document.indexing_status = 'paused'
            documents_status.append(marshal(document, document_status_fields))
        data = {'data': documents_status}
        return data

class DocumentIndexingStatusApi(DocumentResource):

    @setup_required
    @login_required
    @account_initialization_required
    def get(self, dataset_id, document_id):
        if False:
            print('Hello World!')
        dataset_id = str(dataset_id)
        document_id = str(document_id)
        document = self.get_document(dataset_id, document_id)
        completed_segments = DocumentSegment.query.filter(DocumentSegment.completed_at.isnot(None), DocumentSegment.document_id == str(document_id), DocumentSegment.status != 're_segment').count()
        total_segments = DocumentSegment.query.filter(DocumentSegment.document_id == str(document_id), DocumentSegment.status != 're_segment').count()
        document.completed_segments = completed_segments
        document.total_segments = total_segments
        if document.is_paused:
            document.indexing_status = 'paused'
        return marshal(document, document_status_fields)

class DocumentDetailApi(DocumentResource):
    METADATA_CHOICES = {'all', 'only', 'without'}

    @setup_required
    @login_required
    @account_initialization_required
    def get(self, dataset_id, document_id):
        if False:
            print('Hello World!')
        dataset_id = str(dataset_id)
        document_id = str(document_id)
        document = self.get_document(dataset_id, document_id)
        metadata = request.args.get('metadata', 'all')
        if metadata not in self.METADATA_CHOICES:
            raise InvalidMetadataError(f'Invalid metadata value: {metadata}')
        if metadata == 'only':
            response = {'id': document.id, 'doc_type': document.doc_type, 'doc_metadata': document.doc_metadata}
        elif metadata == 'without':
            process_rules = DatasetService.get_process_rules(dataset_id)
            data_source_info = document.data_source_detail_dict
            response = {'id': document.id, 'position': document.position, 'data_source_type': document.data_source_type, 'data_source_info': data_source_info, 'dataset_process_rule_id': document.dataset_process_rule_id, 'dataset_process_rule': process_rules, 'name': document.name, 'created_from': document.created_from, 'created_by': document.created_by, 'created_at': document.created_at.timestamp(), 'tokens': document.tokens, 'indexing_status': document.indexing_status, 'completed_at': int(document.completed_at.timestamp()) if document.completed_at else None, 'updated_at': int(document.updated_at.timestamp()) if document.updated_at else None, 'indexing_latency': document.indexing_latency, 'error': document.error, 'enabled': document.enabled, 'disabled_at': int(document.disabled_at.timestamp()) if document.disabled_at else None, 'disabled_by': document.disabled_by, 'archived': document.archived, 'segment_count': document.segment_count, 'average_segment_length': document.average_segment_length, 'hit_count': document.hit_count, 'display_status': document.display_status, 'doc_form': document.doc_form}
        else:
            process_rules = DatasetService.get_process_rules(dataset_id)
            data_source_info = document.data_source_detail_dict_()
            response = {'id': document.id, 'position': document.position, 'data_source_type': document.data_source_type, 'data_source_info': data_source_info, 'dataset_process_rule_id': document.dataset_process_rule_id, 'dataset_process_rule': process_rules, 'name': document.name, 'created_from': document.created_from, 'created_by': document.created_by, 'created_at': document.created_at.timestamp(), 'tokens': document.tokens, 'indexing_status': document.indexing_status, 'completed_at': int(document.completed_at.timestamp()) if document.completed_at else None, 'updated_at': int(document.updated_at.timestamp()) if document.updated_at else None, 'indexing_latency': document.indexing_latency, 'error': document.error, 'enabled': document.enabled, 'disabled_at': int(document.disabled_at.timestamp()) if document.disabled_at else None, 'disabled_by': document.disabled_by, 'archived': document.archived, 'doc_type': document.doc_type, 'doc_metadata': document.doc_metadata, 'segment_count': document.segment_count, 'average_segment_length': document.average_segment_length, 'hit_count': document.hit_count, 'display_status': document.display_status, 'doc_form': document.doc_form}
        return (response, 200)

class DocumentProcessingApi(DocumentResource):

    @setup_required
    @login_required
    @account_initialization_required
    def patch(self, dataset_id, document_id, action):
        if False:
            while True:
                i = 10
        dataset_id = str(dataset_id)
        document_id = str(document_id)
        document = self.get_document(dataset_id, document_id)
        if current_user.current_tenant.current_role not in ['admin', 'owner']:
            raise Forbidden()
        if action == 'pause':
            if document.indexing_status != 'indexing':
                raise InvalidActionError('Document not in indexing state.')
            document.paused_by = current_user.id
            document.paused_at = datetime.utcnow()
            document.is_paused = True
            db.session.commit()
        elif action == 'resume':
            if document.indexing_status not in ['paused', 'error']:
                raise InvalidActionError('Document not in paused or error state.')
            document.paused_by = None
            document.paused_at = None
            document.is_paused = False
            db.session.commit()
        else:
            raise InvalidActionError()
        return ({'result': 'success'}, 200)

class DocumentDeleteApi(DocumentResource):

    @setup_required
    @login_required
    @account_initialization_required
    def delete(self, dataset_id, document_id):
        if False:
            return 10
        dataset_id = str(dataset_id)
        document_id = str(document_id)
        dataset = DatasetService.get_dataset(dataset_id)
        if dataset is None:
            raise NotFound('Dataset not found.')
        DatasetService.check_dataset_model_setting(dataset)
        document = self.get_document(dataset_id, document_id)
        try:
            DocumentService.delete_document(document)
        except services.errors.document.DocumentIndexingError:
            raise DocumentIndexingError('Cannot delete document during indexing.')
        return ({'result': 'success'}, 204)

class DocumentMetadataApi(DocumentResource):

    @setup_required
    @login_required
    @account_initialization_required
    def put(self, dataset_id, document_id):
        if False:
            while True:
                i = 10
        dataset_id = str(dataset_id)
        document_id = str(document_id)
        document = self.get_document(dataset_id, document_id)
        req_data = request.get_json()
        doc_type = req_data.get('doc_type')
        doc_metadata = req_data.get('doc_metadata')
        if current_user.current_tenant.current_role not in ['admin', 'owner']:
            raise Forbidden()
        if doc_type is None or doc_metadata is None:
            raise ValueError('Both doc_type and doc_metadata must be provided.')
        if doc_type not in DocumentService.DOCUMENT_METADATA_SCHEMA:
            raise ValueError('Invalid doc_type.')
        if not isinstance(doc_metadata, dict):
            raise ValueError('doc_metadata must be a dictionary.')
        metadata_schema = DocumentService.DOCUMENT_METADATA_SCHEMA[doc_type]
        document.doc_metadata = {}
        if doc_type == 'others':
            document.doc_metadata = doc_metadata
        else:
            for (key, value_type) in metadata_schema.items():
                value = doc_metadata.get(key)
                if value is not None and isinstance(value, value_type):
                    document.doc_metadata[key] = value
        document.doc_type = doc_type
        document.updated_at = datetime.utcnow()
        db.session.commit()
        return ({'result': 'success', 'message': 'Document metadata updated.'}, 200)

class DocumentStatusApi(DocumentResource):

    @setup_required
    @login_required
    @account_initialization_required
    def patch(self, dataset_id, document_id, action):
        if False:
            for i in range(10):
                print('nop')
        dataset_id = str(dataset_id)
        document_id = str(document_id)
        dataset = DatasetService.get_dataset(dataset_id)
        if dataset is None:
            raise NotFound('Dataset not found.')
        DatasetService.check_dataset_model_setting(dataset)
        document = self.get_document(dataset_id, document_id)
        if current_user.current_tenant.current_role not in ['admin', 'owner']:
            raise Forbidden()
        indexing_cache_key = 'document_{}_indexing'.format(document.id)
        cache_result = redis_client.get(indexing_cache_key)
        if cache_result is not None:
            raise InvalidActionError('Document is being indexed, please try again later')
        if action == 'enable':
            if document.enabled:
                raise InvalidActionError('Document already enabled.')
            document.enabled = True
            document.disabled_at = None
            document.disabled_by = None
            document.updated_at = datetime.utcnow()
            db.session.commit()
            redis_client.setex(indexing_cache_key, 600, 1)
            add_document_to_index_task.delay(document_id)
            return ({'result': 'success'}, 200)
        elif action == 'disable':
            if not document.completed_at or document.indexing_status != 'completed':
                raise InvalidActionError('Document is not completed.')
            if not document.enabled:
                raise InvalidActionError('Document already disabled.')
            document.enabled = False
            document.disabled_at = datetime.utcnow()
            document.disabled_by = current_user.id
            document.updated_at = datetime.utcnow()
            db.session.commit()
            redis_client.setex(indexing_cache_key, 600, 1)
            remove_document_from_index_task.delay(document_id)
            return ({'result': 'success'}, 200)
        elif action == 'archive':
            if document.archived:
                raise InvalidActionError('Document already archived.')
            document.archived = True
            document.archived_at = datetime.utcnow()
            document.archived_by = current_user.id
            document.updated_at = datetime.utcnow()
            db.session.commit()
            if document.enabled:
                redis_client.setex(indexing_cache_key, 600, 1)
                remove_document_from_index_task.delay(document_id)
            return ({'result': 'success'}, 200)
        elif action == 'un_archive':
            if not document.archived:
                raise InvalidActionError('Document is not archived.')
            if current_app.config['EDITION'] == 'CLOUD':
                documents_count = DocumentService.get_tenant_documents_count()
                total_count = documents_count + 1
                tenant_document_count = int(current_app.config['TENANT_DOCUMENT_COUNT'])
                if total_count > tenant_document_count:
                    raise ValueError(f'All your documents have overed limit {tenant_document_count}.')
            document.archived = False
            document.archived_at = None
            document.archived_by = None
            document.updated_at = datetime.utcnow()
            db.session.commit()
            redis_client.setex(indexing_cache_key, 600, 1)
            add_document_to_index_task.delay(document_id)
            return ({'result': 'success'}, 200)
        else:
            raise InvalidActionError()

class DocumentPauseApi(DocumentResource):

    @setup_required
    @login_required
    @account_initialization_required
    def patch(self, dataset_id, document_id):
        if False:
            i = 10
            return i + 15
        'pause document.'
        dataset_id = str(dataset_id)
        document_id = str(document_id)
        dataset = DatasetService.get_dataset(dataset_id)
        if not dataset:
            raise NotFound('Dataset not found.')
        document = DocumentService.get_document(dataset.id, document_id)
        if document is None:
            raise NotFound('Document Not Exists.')
        if DocumentService.check_archived(document):
            raise ArchivedDocumentImmutableError()
        try:
            DocumentService.pause_document(document)
        except services.errors.document.DocumentIndexingError:
            raise DocumentIndexingError('Cannot pause completed document.')
        return ({'result': 'success'}, 204)

class DocumentRecoverApi(DocumentResource):

    @setup_required
    @login_required
    @account_initialization_required
    def patch(self, dataset_id, document_id):
        if False:
            for i in range(10):
                print('nop')
        'recover document.'
        dataset_id = str(dataset_id)
        document_id = str(document_id)
        dataset = DatasetService.get_dataset(dataset_id)
        if not dataset:
            raise NotFound('Dataset not found.')
        document = DocumentService.get_document(dataset.id, document_id)
        if document is None:
            raise NotFound('Document Not Exists.')
        if DocumentService.check_archived(document):
            raise ArchivedDocumentImmutableError()
        try:
            DocumentService.recover_document(document)
        except services.errors.document.DocumentIndexingError:
            raise DocumentIndexingError('Document is not in paused status.')
        return ({'result': 'success'}, 204)

class DocumentLimitApi(DocumentResource):

    @setup_required
    @login_required
    @account_initialization_required
    def get(self):
        if False:
            i = 10
            return i + 15
        'get document limit'
        documents_count = DocumentService.get_tenant_documents_count()
        tenant_document_count = int(current_app.config['TENANT_DOCUMENT_COUNT'])
        return ({'documents_count': documents_count, 'documents_limit': tenant_document_count}, 200)
api.add_resource(GetProcessRuleApi, '/datasets/process-rule')
api.add_resource(DatasetDocumentListApi, '/datasets/<uuid:dataset_id>/documents')
api.add_resource(DatasetInitApi, '/datasets/init')
api.add_resource(DocumentIndexingEstimateApi, '/datasets/<uuid:dataset_id>/documents/<uuid:document_id>/indexing-estimate')
api.add_resource(DocumentBatchIndexingEstimateApi, '/datasets/<uuid:dataset_id>/batch/<string:batch>/indexing-estimate')
api.add_resource(DocumentBatchIndexingStatusApi, '/datasets/<uuid:dataset_id>/batch/<string:batch>/indexing-status')
api.add_resource(DocumentIndexingStatusApi, '/datasets/<uuid:dataset_id>/documents/<uuid:document_id>/indexing-status')
api.add_resource(DocumentDetailApi, '/datasets/<uuid:dataset_id>/documents/<uuid:document_id>')
api.add_resource(DocumentProcessingApi, '/datasets/<uuid:dataset_id>/documents/<uuid:document_id>/processing/<string:action>')
api.add_resource(DocumentDeleteApi, '/datasets/<uuid:dataset_id>/documents/<uuid:document_id>')
api.add_resource(DocumentMetadataApi, '/datasets/<uuid:dataset_id>/documents/<uuid:document_id>/metadata')
api.add_resource(DocumentStatusApi, '/datasets/<uuid:dataset_id>/documents/<uuid:document_id>/status/<string:action>')
api.add_resource(DocumentPauseApi, '/datasets/<uuid:dataset_id>/documents/<uuid:document_id>/processing/pause')
api.add_resource(DocumentRecoverApi, '/datasets/<uuid:dataset_id>/documents/<uuid:document_id>/processing/resume')
api.add_resource(DocumentLimitApi, '/datasets/limit')