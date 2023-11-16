import json
import logging
import os
import re
import socket
from functools import lru_cache
from typing import Dict, Optional, Union
import boto3
from localstack import config
from localstack.aws.accounts import get_aws_account_id
from localstack.config import S3_VIRTUAL_HOSTNAME
from localstack.constants import APPLICATION_AMZ_JSON_1_0, APPLICATION_AMZ_JSON_1_1, APPLICATION_X_WWW_FORM_URLENCODED, AWS_REGION_US_EAST_1, ENV_DEV, HEADER_LOCALSTACK_ACCOUNT_ID, LOCALHOST, REGION_LOCAL
from localstack.utils.strings import is_string, is_string_or_bytes, to_str
LOG = logging.getLogger(__name__)
LOCAL_REGION = None
INITIAL_BOTO3_SESSION = None
CACHE_S3_HOSTNAME_DNS_STATUS = None

@lru_cache()
def get_valid_regions():
    if False:
        return 10
    valid_regions = set()
    for partition in set(boto3.Session().get_available_partitions()):
        for region in boto3.Session().get_available_regions('sns', partition):
            valid_regions.add(region)
    return valid_regions

def get_valid_regions_for_service(service_name):
    if False:
        print('Hello World!')
    regions = list(boto3.Session().get_available_regions(service_name))
    regions.extend(boto3.Session().get_available_regions('cloudwatch', partition_name='aws-us-gov'))
    regions.extend(boto3.Session().get_available_regions('cloudwatch', partition_name='aws-cn'))
    return regions

class Environment:

    def __init__(self, region=None, prefix=None):
        if False:
            while True:
                i = 10
        self.region = region or get_local_region()
        self.prefix = prefix

    def apply_json(self, j):
        if False:
            return 10
        if isinstance(j, str):
            j = json.loads(j)
        self.__dict__.update(j)

    @staticmethod
    def from_string(s):
        if False:
            for i in range(10):
                print('nop')
        parts = s.split(':')
        if len(parts) == 1:
            if s in PREDEFINED_ENVIRONMENTS:
                return PREDEFINED_ENVIRONMENTS[s]
            parts = [get_local_region(), s]
        if len(parts) > 2:
            raise Exception('Invalid environment string "%s"' % s)
        region = parts[0]
        prefix = parts[1]
        return Environment(region=region, prefix=prefix)

    @staticmethod
    def from_json(j):
        if False:
            return 10
        if not isinstance(j, dict):
            j = j.to_dict()
        result = Environment()
        result.apply_json(j)
        return result

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return '%s:%s' % (self.region, self.prefix)
PREDEFINED_ENVIRONMENTS = {ENV_DEV: Environment(region=REGION_LOCAL, prefix=ENV_DEV)}

def get_environment(env=None, region_name=None):
    if False:
        return 10
    "\n    Return an Environment object based on the input arguments.\n\n    Parameter `env` can be either of:\n        * None (or empty), in which case the rules below are applied to (env = os.environ['ENV'] or ENV_DEV)\n        * an Environment object (then this object is returned)\n        * a string '<region>:<name>', which corresponds to Environment(region='<region>', prefix='<prefix>')\n        * the predefined string 'dev' (ENV_DEV), which implies Environment(region='local', prefix='dev')\n        * a string '<name>', which implies Environment(region=DEFAULT_REGION, prefix='<name>')\n\n    Additionally, parameter `region_name` can be used to override DEFAULT_REGION.\n    "
    if not env:
        if 'ENV' in os.environ:
            env = os.environ['ENV']
        else:
            env = ENV_DEV
    elif not is_string(env) and (not isinstance(env, Environment)):
        raise Exception('Invalid environment: %s' % env)
    if is_string(env):
        env = Environment.from_string(env)
    if region_name:
        env.region = region_name
    if not env.region:
        raise Exception('Invalid region in environment: "%s"' % env)
    return env

def is_local_env(env):
    if False:
        i = 10
        return i + 15
    return not env or env.region == REGION_LOCAL or env.prefix == ENV_DEV

def get_region():
    if False:
        while True:
            i = 10
    from localstack.utils.aws.request_context import get_region_from_request_context
    region = get_region_from_request_context()
    if region:
        return region
    return get_local_region()

def get_partition(region_name: str=None):
    if False:
        while True:
            i = 10
    region_name = region_name or get_region()
    return boto3.session.Session().get_partition_for_region(region_name)

def get_local_region():
    if False:
        return 10
    global LOCAL_REGION
    if LOCAL_REGION is None:
        LOCAL_REGION = get_boto3_region() or ''
    return AWS_REGION_US_EAST_1 or LOCAL_REGION

def get_boto3_region() -> str:
    if False:
        print('Hello World!')
    'Return the region name, as determined from the environment when creating a new boto3 session'
    return boto3.session.Session().region_name

def get_local_service_url(service_name_or_port: Union[str, int]) -> str:
    if False:
        return 10
    'Return the local service URL for the given service name or port.'
    if isinstance(service_name_or_port, int):
        return f'{config.get_protocol()}://{LOCALHOST}:{service_name_or_port}'
    return config.internal_service_url()

def get_s3_hostname():
    if False:
        i = 10
        return i + 15
    global CACHE_S3_HOSTNAME_DNS_STATUS
    if CACHE_S3_HOSTNAME_DNS_STATUS is None:
        try:
            assert socket.gethostbyname(S3_VIRTUAL_HOSTNAME)
            CACHE_S3_HOSTNAME_DNS_STATUS = True
        except socket.error:
            CACHE_S3_HOSTNAME_DNS_STATUS = False
    if CACHE_S3_HOSTNAME_DNS_STATUS:
        return S3_VIRTUAL_HOSTNAME
    return LOCALHOST

def fix_account_id_in_arns(response, colon_delimiter=':', existing=None, replace=None):
    if False:
        print('Hello World!')
    'Fix the account ID in the ARNs returned in the given Flask response or string'
    existing = existing or ['123456789', '1234567890', '123456789012', get_aws_account_id()]
    existing = existing if isinstance(existing, list) else [existing]
    replace = replace or get_aws_account_id()
    is_str_obj = is_string_or_bytes(response)
    content = to_str(response if is_str_obj else response._content)
    replace = 'arn{col}aws{col}\\1{col}\\2{col}{acc}{col}'.format(col=colon_delimiter, acc=replace)
    for acc_id in existing:
        regex = 'arn{col}aws{col}([^:%]+){col}([^:%]*){col}{acc}{col}'.format(col=colon_delimiter, acc=acc_id)
        content = re.sub(regex, replace, content)
    if not is_str_obj:
        response._content = content
        response.headers['Content-Length'] = len(response._content)
        return response
    return content

def inject_test_credentials_into_env(env):
    if False:
        for i in range(10):
            print('nop')
    if 'AWS_ACCESS_KEY_ID' not in env and 'AWS_SECRET_ACCESS_KEY' not in env:
        env['AWS_ACCESS_KEY_ID'] = 'test'
        env['AWS_SECRET_ACCESS_KEY'] = 'test'

def inject_region_into_env(env, region):
    if False:
        return 10
    env['AWS_REGION'] = region

def extract_region_from_auth_header(headers: Dict[str, str], use_default=True) -> str:
    if False:
        i = 10
        return i + 15
    auth = headers.get('Authorization') or ''
    region = re.sub('.*Credential=[^/]+/[^/]+/([^/]+)/.*', '\\1', auth)
    if region == auth:
        region = None
    if use_default:
        region = region or get_region()
    return region

def extract_access_key_id_from_auth_header(headers: Dict[str, str]) -> Optional[str]:
    if False:
        while True:
            i = 10
    auth = headers.get('Authorization') or ''
    if auth.startswith('AWS4-'):
        access_id = re.findall('.*Credential=([^/]+)/[^/]+/[^/]+/.*', auth)
        if len(access_id):
            return access_id[0]
    elif auth.startswith('AWS '):
        access_id = auth.removeprefix('AWS ').split(':')
        if len(access_id):
            return access_id[0]

def mock_aws_request_headers(service: str, aws_access_key_id: str, region_name: str, internal: bool=False) -> Dict[str, str]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a mock set of headers that resemble SigV4 signing method.\n    '
    ctype = APPLICATION_AMZ_JSON_1_0
    if service == 'kinesis':
        ctype = APPLICATION_AMZ_JSON_1_1
    elif service in ['sns', 'sqs', 'sts', 'cloudformation']:
        ctype = APPLICATION_X_WWW_FORM_URLENCODED
    headers = {'Content-Type': ctype, 'Accept-Encoding': 'identity', 'X-Amz-Date': '20160623T103251Z', 'Authorization': 'AWS4-HMAC-SHA256 ' + f'Credential={aws_access_key_id}/20160623/{region_name}/{service}/aws4_request, ' + 'SignedHeaders=content-type;host;x-amz-date;x-amz-target, Signature=1234'}
    if internal:
        headers[HEADER_LOCALSTACK_ACCOUNT_ID] = get_aws_account_id()
    return headers