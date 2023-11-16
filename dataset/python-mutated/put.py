"""
    eve.methods.put
    ~~~~~~~~~~~~~~~

    This module implements the PUT method.

    :copyright: (c) 2017 by Nicola Iarocci.
    :license: BSD, see LICENSE for more details.
"""
from cerberus.validator import DocumentError
from flask import abort
from flask import current_app as app
from werkzeug import exceptions
from eve.auth import auth_field_and_value, requires_auth
from eve.methods.common import build_response_document, get_document, marshal_write_response, oplog_push, parse
from eve.methods.common import payload as payload_
from eve.methods.common import pre_event, ratelimit, resolve_document_etag, resolve_embedded_fields, resolve_sub_resource_path, resolve_user_restricted_access, store_media_files, utcnow
from eve.methods.post import post_internal
from eve.utils import config, debug_error_message, parse_request
from eve.versioning import insert_versioning_documents, late_versioning_catch, resolve_document_version

@ratelimit()
@requires_auth('item')
@pre_event
def put(resource, payload=None, **lookup):
    if False:
        print('Hello World!')
    '\n    Default function for handling PUT requests, it has decorators for\n    rate limiting, authentication and for raising pre-request events.\n    After the decorators are applied forwards to call to :func:`put_internal`\n\n    .. versionchanged:: 0.5\n       Split into put() and put_internal().\n    '
    return put_internal(resource, payload, concurrency_check=True, skip_validation=False, **lookup)

def put_internal(resource, payload=None, concurrency_check=False, skip_validation=False, **lookup):
    if False:
        print('Hello World!')
    "Intended for internal put calls, this method is not rate limited,\n    authentication is not checked, pre-request events are not raised, and\n    concurrency checking is optional. Performs a document replacement.\n    Updates are first validated against the resource schema. If validation\n    passes, the document is replaced and an OK status update is returned.\n    If validation fails a set of validation issues is returned.\n\n    :param resource: the name of the resource to which the document belongs.\n    :param payload: alternative payload. When calling put() from your own code\n                    you can provide an alternative payload. This can be useful,\n                    for example, when you have a callback function hooked to a\n                    certain endpoint, and want to perform additional put()\n                    callsfrom there.\n\n                    Please be advised that in order to successfully use this\n                    option, a request context must be available.\n    :param concurrency_check: concurrency check switch (bool)\n    :param skip_validation: skip payload validation before write (bool)\n    :param **lookup: document lookup query.\n\n    .. versionchanged:: 0.6\n       Create document if it does not exist. Closes #634.\n       Allow restoring soft deleted documents via PUT\n\n    .. versionchanged:: 0.5\n       Back to resolving default values after validation as now the validator\n       can properly validate dependency even when some have default values. See\n       #353.\n       Original put() has been split into put() and put_internal().\n       You can now pass a pre-defined custom payload to the funcion.\n       ETAG is now stored with the document (#369).\n       Catching all HTTPExceptions and returning them to the caller, allowing\n       for eventual flask.abort() invocations in callback functions to go\n       through. Fixes #395.\n\n    .. versionchanged:: 0.4\n       Allow abort() to be invoked by callback functions.\n       Resolve default values before validation is performed. See #353.\n       Raise 'on_replace' instead of 'on_insert'. The callback function gets\n       the document (as opposed to a list of just 1 document) as an argument.\n       Support for document versioning.\n       Raise `on_replaced` after the document has been replaced\n\n    .. versionchanged:: 0.3\n       Support for media fields.\n       When IF_MATCH is disabled, no etag is included in the payload.\n       Support for new validation format introduced with Cerberus v0.5.\n\n    .. versionchanged:: 0.2\n       Use the new STATUS setting.\n       Use the new ISSUES setting.\n       Raise pre_<method> event.\n       explicitly resolve default values instead of letting them be resolved\n       by common.parse. This avoids a validation error when a read-only field\n       also has a default value.\n\n    .. versionchanged:: 0.1.1\n       auth.request_auth_value is now used to store the auth_field value.\n       Item-identifier wrapper stripped from both request and response payload.\n\n    .. versionadded:: 0.1.0\n    "
    resource_def = app.config['DOMAIN'][resource]
    schema = resource_def['schema']
    validator = app.validator(schema, resource=resource, allow_unknown=resource_def['allow_unknown'])
    if payload is None:
        payload = payload_()
    original = get_document(resource, concurrency_check, check_auth_value=False, force_auth_field_projection=True, **lookup)
    if not original:
        if config.UPSERT_ON_PUT:
            id = lookup[resource_def['id_field']]
            if schema[resource_def['id_field']].get('type', '') == 'objectid':
                id = str(id)
            payload[resource_def['id_field']] = id
            return post_internal(resource, payl=payload)
        abort(404)
    (auth_field, request_auth_value) = auth_field_and_value(resource)
    if auth_field and original.get(auth_field) != request_auth_value:
        abort(403)
    last_modified = None
    etag = None
    issues = {}
    object_id = original[resource_def['id_field']]
    response = {}
    if config.BANDWIDTH_SAVER is True:
        embedded_fields = []
    else:
        req = parse_request(resource)
        embedded_fields = resolve_embedded_fields(resource, req)
    try:
        document = parse(payload, resource)
        resolve_sub_resource_path(document, resource)
        if skip_validation:
            validation = True
        else:
            validation = validator.validate_replace(document, object_id, original)
            document = validator.document
        if validation:
            late_versioning_catch(original, resource)
            last_modified = utcnow()
            document[config.LAST_UPDATED] = last_modified
            document[config.DATE_CREATED] = original[config.DATE_CREATED]
            if resource_def['soft_delete'] is True:
                document[config.DELETED] = False
            if resource_def['id_field'] not in document:
                document[resource_def['id_field']] = object_id
            resolve_user_restricted_access(document, resource)
            store_media_files(document, resource, original)
            resolve_document_version(document, resource, 'PUT', original)
            getattr(app, 'on_replace')(resource, document, original)
            getattr(app, 'on_replace_%s' % resource)(document, original)
            resolve_document_etag(document, resource)
            try:
                app.data.replace(resource, object_id, document, original)
            except app.data.OriginalChangedError:
                if concurrency_check:
                    abort(412, description="Client and server etags don't match")
            oplog_push(resource, document, 'PUT')
            insert_versioning_documents(resource, document)
            getattr(app, 'on_replaced')(resource, document, original)
            getattr(app, 'on_replaced_%s' % resource)(document, original)
            build_response_document(document, resource, embedded_fields, document)
            response = document
            if config.IF_MATCH:
                etag = response[config.ETAG]
        else:
            issues = validator.errors
    except DocumentError as e:
        issues['validator exception'] = str(e)
    except exceptions.HTTPException as e:
        raise e
    except Exception as e:
        app.logger.exception(e)
        abort(400, description=debug_error_message('An exception occurred: %s' % e))
    if issues:
        response[config.ISSUES] = issues
        response[config.STATUS] = config.STATUS_ERR
        status = config.VALIDATION_ERROR_STATUS
    else:
        response[config.STATUS] = config.STATUS_OK
        status = 200
    response = marshal_write_response(response, resource)
    return (response, last_modified, etag, status)