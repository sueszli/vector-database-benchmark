import logging
import os
from typing import NamedTuple, Optional, Set
import botocore
from werkzeug.http import parse_dict_header
import localstack
from localstack import config
from localstack.aws.spec import ServiceCatalog, build_service_index_cache, load_service_index_cache
from localstack.constants import LOCALHOST_HOSTNAME, PATH_USER_REQUEST
from localstack.http import Request
from localstack.services.s3.utils import uses_host_addressing
from localstack.services.sqs.utils import is_sqs_queue_url
from localstack.utils.objects import singleton_factory
from localstack.utils.strings import to_bytes
from localstack.utils.urls import hostname_from_url
LOG = logging.getLogger(__name__)

class _ServiceIndicators(NamedTuple):
    """
    Encapsulates the different fields that might indicate which service a request is targeting.

    This class does _not_ contain any data which is parsed from the body of the request in order to defer or even avoid
    processing the body.
    """
    signing_name: Optional[str] = None
    target_prefix: Optional[str] = None
    operation: Optional[str] = None
    host: Optional[str] = None
    path: Optional[str] = None

def _extract_service_indicators(request: Request) -> _ServiceIndicators:
    if False:
        for i in range(10):
            print('nop')
    'Extracts all different fields that might indicate which service a request is targeting.'
    x_amz_target = request.headers.get('x-amz-target')
    authorization = request.headers.get('authorization')
    signing_name = None
    if authorization:
        try:
            (auth_type, auth_info) = authorization.split(None, 1)
            auth_type = auth_type.lower().strip()
            if auth_type == 'aws4-hmac-sha256':
                values = parse_dict_header(auth_info)
                (_, _, _, signing_name, _) = values['Credential'].split('/')
        except (ValueError, KeyError):
            LOG.debug('auth header could not be parsed for service routing: %s', authorization)
            pass
    if x_amz_target:
        if '.' in x_amz_target:
            (target_prefix, operation) = x_amz_target.split('.', 1)
        else:
            target_prefix = None
            operation = x_amz_target
    else:
        (target_prefix, operation) = (None, None)
    return _ServiceIndicators(signing_name, target_prefix, operation, request.host, request.path)
signing_name_path_prefix_rules = {'apigateway': {'/v2': 'apigatewayv2'}, 'appconfig': {'/configuration': 'appconfigdata'}, 'execute-api': {'/@connections': 'apigatewaymanagementapi', '/participant': 'connectparticipant', '*': 'iot'}, 'ses': {'/v2': 'sesv2', '/v1': 'pinpoint-email'}, 'greengrass': {'/greengrass/v2/': 'greengrassv2'}, 'cloudsearch': {'/2013-01-01': 'cloudsearchdomain'}, 's3': {'/v20180820': 's3control'}, 'iot1click': {'/projects': 'iot1click-projects', '/devices': 'iot1click-devices'}, 'es': {'/2015-01-01': 'es', '/2021-01-01': 'opensearch'}, 'sagemaker': {'/endpoints': 'sagemaker-runtime', '/human-loops': 'sagemaker-a2i-runtime'}}

def custom_signing_name_rules(signing_name: str, path: str) -> Optional[str]:
    if False:
        i = 10
        return i + 15
    '\n    Rules which are based on the signing name (in the auth header) and the request path.\n    '
    rules = signing_name_path_prefix_rules.get(signing_name)
    if not rules:
        if signing_name == 'servicecatalog':
            if path == '/':
                return 'servicecatalog'
            else:
                return 'servicecatalog-appregistry'
        return
    for (prefix, name) in rules.items():
        if path.startswith(prefix):
            return name
    return rules.get('*', signing_name)

def custom_host_addressing_rules(host: str) -> Optional[str]:
    if False:
        return 10
    '\n    Rules based on the host header of the request.\n    '
    if '.execute-api.' in host:
        return 'apigateway'
    if '.lambda-url.' in host:
        return 'lambda'

def custom_path_addressing_rules(path: str) -> Optional[str]:
    if False:
        while True:
            i = 10
    '\n    Rules which are only based on the request path.\n    '
    if is_sqs_queue_url(path):
        return 'sqs-query'
    if path.startswith('/2015-03-31/functions/'):
        return 'lambda'

def legacy_rules(request: Request) -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    '\n    *Legacy* rules which migrate routing logic which will become obsolete with the ASF Gateway.\n    All rules which are implemented here should be migrated to the new router once these services are migrated to ASF.\n\n    TODO: These custom rules should become obsolete by migrating these to use the http/router.py\n    '
    path = request.path
    method = request.method
    host = hostname_from_url(request.host)
    if '/%s/' % PATH_USER_REQUEST in request.path or (host.endswith(LOCALHOST_HOSTNAME) and 'execute-api' in host):
        return 'apigateway'
    if '.lambda-url.' in host:
        return 'lambda'
    if path.startswith('/shell') or path.startswith('/dynamodb/shell'):
        return 'dynamodb'
    if path == '/health' or path.startswith('/_localstack') or path.startswith('/_pods') or path.startswith('/_aws'):
        return None
    stripped = path.strip('/')
    if method in ['GET', 'HEAD'] and stripped:
        return 's3'
    if stripped and '/' not in stripped:
        if method == 'PUT':
            return 's3'
        if method == 'POST' and 'key' in request.values:
            return 's3'
    if 'aws-cli/' in str(request.user_agent):
        return 's3'
    values = request.values
    if any((value in values for value in ['AWSAccessKeyId', 'Signature', 'X-Amz-Algorithm', 'X-Amz-Credential', 'X-Amz-Date', 'X-Amz-Expires', 'X-Amz-SignedHeaders', 'X-Amz-Signature'])):
        return 's3'
    if method == 'POST' and 'delete' in values:
        data_bytes = to_bytes(request.data)
        if b'<Delete' in data_bytes and b'<Key>' in data_bytes:
            return 's3'
    if stripped.count('/') >= 1 and method == 'PUT':
        return 's3'
    auth_header = request.headers.get('Authorization') or ''
    if auth_header.startswith('AWS '):
        return 's3'
    if uses_host_addressing(request.headers):
        return 's3'

@singleton_factory
def get_service_catalog() -> ServiceCatalog:
    if False:
        print('Hello World!')
    'Loads the ServiceCatalog (which contains all the service specs), and potentially re-uses a cached index.'
    if not os.path.isdir(config.dirs.cache):
        return ServiceCatalog()
    try:
        ls_ver = localstack.__version__.replace('.', '_')
        botocore_ver = botocore.__version__.replace('.', '_')
        cache_file_name = f'service-catalog-{ls_ver}-{botocore_ver}.pickle'
        cache_file = os.path.join(config.dirs.cache, cache_file_name)
        if not os.path.exists(cache_file):
            LOG.debug('building service catalog index cache file %s', cache_file)
            index = build_service_index_cache(cache_file)
        else:
            LOG.debug('loading service catalog index cache file %s', cache_file)
            index = load_service_index_cache(cache_file)
        return ServiceCatalog(index)
    except Exception:
        LOG.exception('error while processing service catalog index cache, falling back to lazy-loaded index')
        return ServiceCatalog()

def resolve_conflicts(candidates: Set[str], request: Request):
    if False:
        while True:
            i = 10
    '\n    Some service definitions are overlapping to a point where they are _not_ distinguishable at all\n    (f.e. ``DescribeEndpints`` in timestream-query and timestream-write).\n    These conflicts need to be resolved manually.\n    '
    if candidates == {'timestream-query', 'timestream-write'}:
        return 'timestream-query'
    if candidates == {'docdb', 'neptune', 'rds'}:
        return 'rds'
    if candidates == {'sqs-query', 'sqs'}:
        content_type = request.headers.get('Content-Type')
        return 'sqs' if content_type == 'application/x-amz-json-1.0' else 'sqs-query'

def determine_aws_service_name(request: Request, services: ServiceCatalog=None) -> Optional[str]:
    if False:
        print('Hello World!')
    '\n    Tries to determine the name of the AWS service an incoming request is targeting.\n    :param request: to determine the target service name of\n    :param services: service catalog (can be handed in for caching purposes)\n    :return: service name string (or None if the targeting service could not be determined exactly)\n    '
    services = services or get_service_catalog()
    (signing_name, target_prefix, operation, host, path) = _extract_service_indicators(request)
    candidates = set()
    if signing_name:
        signing_name_candidates = services.by_signing_name(signing_name)
        if len(signing_name_candidates) == 1:
            return signing_name_candidates[0]
        custom_match = custom_signing_name_rules(signing_name, path)
        if custom_match:
            return custom_match
        candidates.update(signing_name_candidates)
    if target_prefix and operation:
        target_candidates = services.by_target_prefix(target_prefix)
        if len(target_candidates) == 1:
            return target_candidates[0]
        candidates.update(target_candidates)
        for service_name in list(candidates):
            service = services.get(service_name)
            if operation not in service.operation_names:
                candidates.remove(service_name)
    else:
        for service_name in list(candidates):
            service = services.get(service_name)
            if service.metadata.get('targetPrefix') is not None:
                candidates.remove(service_name)
    if len(candidates) == 1:
        return candidates.pop()
    if path and path != '/':
        custom_path_match = custom_path_addressing_rules(path)
        if custom_path_match:
            return custom_path_match
    if host:
        for (prefix, services_per_prefix) in services.endpoint_prefix_index.items():
            if host.startswith(f'{prefix}.') and '.s3.' not in host:
                if len(services_per_prefix) == 1:
                    return services_per_prefix[0]
                candidates.update(services_per_prefix)
        custom_host_match = custom_host_addressing_rules(host)
        if custom_host_match:
            return custom_host_match
    if request.shallow:
        return None
    values = request.values
    if 'Action' in values:
        query_candidates = [service for service in services.by_operation(values['Action']) if services.get(service).protocol in ('ec2', 'query')]
        if len(query_candidates) == 1:
            return query_candidates[0]
        if 'Version' in values:
            for service in list(query_candidates):
                service_model = services.get(service)
                if values['Version'] != service_model.api_version:
                    query_candidates.remove(service)
        if len(query_candidates) == 1:
            return query_candidates[0]
        candidates.update(query_candidates)
    resolved_conflict = resolve_conflicts(candidates, request)
    if resolved_conflict:
        return resolved_conflict
    legacy_match = legacy_rules(request)
    if legacy_match:
        return legacy_match
    if signing_name:
        return signing_name
    if candidates:
        return candidates.pop()
    return None