import base64
import hashlib
import logging
import re
from re import Match
from typing import Optional
from urllib.parse import urlparse
from localstack import config
from localstack.aws.api import RequestContext
from localstack.aws.chain import Handler, HandlerChain
from localstack.http import Response
from localstack.http.proxy import forward
from localstack.http.request import Request, get_full_raw_path, get_raw_path, restore_payload
from localstack.utils.aws.aws_responses import calculate_crc32
from localstack.utils.aws.request_context import extract_region_from_headers
from localstack.utils.run import to_str
from localstack.utils.strings import to_bytes
LOG = logging.getLogger(__name__)

class ArnPartitionRewriteHandler(Handler):
    """
    Intercepts requests and responses and tries to adjust the partitions in ARNs within the
    intercepted requests.
    For incoming requests, the default partition is set ("aws").
    For outgoing responses, the partition is adjusted based on the region in the ARN, or by the
    default region if the ARN does not contain a region.
    This listener is used to support other partitions than the default "aws" partition (f.e.
    aws-us-gov) without
    rewriting all the cases where the ARN is parsed or constructed within LocalStack or moto.
    In other words, this listener makes sure that internally the ARNs are always in the partition
    "aws", while the client gets ARNs with the proper partition.

    There are multiple operation modes you can choose for the rewriting:

    - "request": only the requests gets rewritten (to DEFAULT_INBOUND_PARTITION)
    - "response": only the response gets rewritten (to original partition based on the region)
    - "bidirectional": both request and response are rewritten as described above
    - "internal-guard": both request and response are rewritten, but the response is also rewritten to DEFAULT_INBOUND_PARTITION (!)

    Default behavior for external clients is "bidirectional".
    Default behavior for internal clients is "internal-guard".
    Behavior can be overwritten by setting the "LS-INTERNAL-REWRITE-MODE" header

    """
    DEFAULT_INBOUND_PARTITION = 'aws'

    class InvalidRegionException(Exception):
        """An exception indicating that a region could not be matched to a partition."""
    arn_regex = re.compile('arn:(?P<Partition>(aws|aws-cn|aws-iso|aws-iso-b|aws-us-gov)*):(?P<Service>[\\w-]*):(?P<Region>[\\w-]*):(?P<AccountID>[\\w-]*):(?P<ResourcePath>((?P<ResourceType>[\\w-]*)[:/])?(?P<ResourceID>[\\w\\-/*]*))')
    arn_regex_encoded = re.compile('arn%3A(?P<Partition>(aws|aws-cn|aws-iso|aws-iso-b|aws-us-gov)*)%3A(?P<Service>[\\w-]*)%3A(?P<Region>[\\w-]*)%3A(?P<AccountID>[\\w-]*)%3A(?P<ResourcePath>((?P<ResourceType>[\\w-]*)((%3A)|(%2F)))?(?P<ResourceID>(\\w|-|(%2F)|(%2A))*))')

    def __call__(self, chain: HandlerChain, context: RequestContext, response: Response):
        if False:
            return 10
        request = context.request
        if request.headers.pop('LS-INTERNAL-REWRITE-HANDLER', None):
            return
        request_region = extract_region_from_headers(request.headers)
        rewrite_mode = request.headers.pop('LS-INTERNAL-REWRITE-MODE', None)
        if rewrite_mode is None and context.is_internal_call:
            rewrite_mode = 'internal-guard'
        else:
            rewrite_mode = 'bidirectional'
        if rewrite_mode in {'request', 'bidirectional', 'internal-guard'}:
            request = self.modify_request(request)
        result_response = forward(request=request, forward_base_url=config.internal_service_url(), forward_path=get_raw_path(request), headers=request.headers)
        match rewrite_mode:
            case 'response' | 'bidirectional':
                self.modify_response_revert(result_response, request_region=request_region)
            case 'internal-guard':
                self.modify_response_guard(result_response)
        response.update_from(result_response)
        chain.terminate()

    def modify_request(self, request: Request) -> Request:
        if False:
            for i in range(10):
                print('nop')
        '\n        Modifies the request by rewriting ARNs to default partition\n\n\n        :param request: Request\n        :return: New request with rewritten data\n        '
        full_forward_rewritten_path = self._adjust_partition(get_full_raw_path(request), self.DEFAULT_INBOUND_PARTITION, encoded=True)
        parsed_forward_rewritten_path = urlparse(full_forward_rewritten_path)
        body_is_encoded = request.mimetype == 'application/x-www-form-urlencoded'
        forward_rewritten_body = self._adjust_partition(restore_payload(request), self.DEFAULT_INBOUND_PARTITION, encoded=body_is_encoded)
        forward_rewritten_headers = self._adjust_partition(dict(request.headers), self.DEFAULT_INBOUND_PARTITION)
        if 'Content-MD5' in forward_rewritten_headers:
            md = hashlib.md5(forward_rewritten_body).digest()
            content_md5 = base64.b64encode(md).decode('utf-8')
            forward_rewritten_headers['Content-MD5'] = content_md5
        forward_rewritten_headers['LS-INTERNAL-REWRITE-HANDLER'] = '1'
        return Request(method=request.method, path=parsed_forward_rewritten_path.path, query_string=parsed_forward_rewritten_path.query, headers=forward_rewritten_headers, body=forward_rewritten_body, raw_path=parsed_forward_rewritten_path.path)

    def modify_response_revert(self, response: Response, request_region: str):
        if False:
            return 10
        '\n        Modifies the supplied response by rewriting the ARNs back based on the regions in the arn or the supplied region\n\n        :param response: Response to be modified\n        :param request_region: Region the original request was meant for\n        '
        response.headers = self._adjust_partition(dict(response.headers), request_region=request_region)
        response.data = self._adjust_partition(response.data, request_region=request_region)
        self._post_process_response_headers(response)

    def modify_response_guard(self, response: Response):
        if False:
            return 10
        '\n        Modifies the supplied response by rewriting the ARNs to default partition\n\n        :param response: Response to be modified\n        :param request_region: Region the original request was meant for\n        '
        response.headers = self._adjust_partition(dict(response.headers), static_partition=self.DEFAULT_INBOUND_PARTITION)
        response.data = self._adjust_partition(response.data, static_partition=self.DEFAULT_INBOUND_PARTITION)
        self._post_process_response_headers(response)

    def _adjust_partition(self, source, static_partition: str=None, request_region: str=None, encoded: bool=False):
        if False:
            print('Hello World!')
        if isinstance(source, dict):
            result = {}
            for (k, v) in source.items():
                result[k] = self._adjust_partition(v, static_partition, request_region, encoded=encoded)
            return result
        if isinstance(source, list):
            result = []
            for v in source:
                result.append(self._adjust_partition(v, static_partition, request_region, encoded=encoded))
            return result
        elif isinstance(source, bytes):
            try:
                decoded = to_str(source)
                adjusted = self._adjust_partition(decoded, static_partition, request_region, encoded=encoded)
                return to_bytes(adjusted)
            except UnicodeDecodeError:
                return source
        elif not isinstance(source, str):
            return source
        regex = self.arn_regex if not encoded else self.arn_regex_encoded
        return regex.sub(lambda m: self._adjust_match(m, static_partition, request_region, encoded=encoded), source)

    def _adjust_match(self, match: Match, static_partition: str=None, request_region: str=None, encoded: bool=False):
        if False:
            print('Hello World!')
        region = match.group('Region')
        partition = self._partition_lookup(region, request_region) if static_partition is None else static_partition
        service = match.group('Service')
        account_id = match.group('AccountID')
        resource_path = match.group('ResourcePath')
        separator = ':' if not encoded else '%3A'
        return f'arn{separator}{partition}{separator}{service}{separator}{region}{separator}{account_id}{separator}{resource_path}'

    def _partition_lookup(self, region: str, request_region: str=None):
        if False:
            return 10
        try:
            partition = self._get_partition_for_region(region)
        except ArnPartitionRewriteHandler.InvalidRegionException:
            try:
                partition = self._get_partition_for_region(request_region)
            except self.InvalidRegionException:
                partition = config.ARN_PARTITION_FALLBACK
        return partition

    @staticmethod
    def _get_partition_for_region(region: Optional[str]) -> str:
        if False:
            while True:
                i = 10
        if region and region.startswith('us-gov-'):
            return 'aws-us-gov'
        elif region and region.startswith('us-iso-'):
            return 'aws-iso'
        elif region and region.startswith('us-isob-'):
            return 'aws-iso-b'
        elif region and region.startswith('cn-'):
            return 'aws-cn'
        elif region and re.match('^(us|eu|ap|sa|ca|me|af)-\\w+-\\d+$', region):
            return 'aws'
        else:
            raise ArnPartitionRewriteHandler.InvalidRegionException(f'Region ({region}) could not be matched to a partition.')

    @staticmethod
    def _post_process_response_headers(response: Response) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Adjust potential content lengths and checksums after modifying the response.'
        if response.headers and response.data:
            if 'Content-Length' in response.headers:
                response.headers['Content-Length'] = str(len(to_bytes(response.data)))
            if 'x-amz-crc32' in response.headers:
                response.headers['x-amz-crc32'] = calculate_crc32(response.data)