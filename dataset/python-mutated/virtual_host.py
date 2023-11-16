import copy
import logging
from urllib.parse import urlsplit, urlunsplit
from localstack import config
from localstack.constants import LOCALHOST_HOSTNAME
from localstack.http import Request, Response
from localstack.http.proxy import Proxy
from localstack.http.request import get_raw_path
from localstack.runtime import hooks
from localstack.services.edge import ROUTER
from localstack.services.s3.utils import S3_VIRTUAL_HOST_FORWARDED_HEADER
LOG = logging.getLogger(__name__)
AWS_REGION_REGEX = '(?:us-gov|us|ap|ca|cn|eu|sa)-[a-z]+-\\d'
VHOST_REGEX_PATTERN = f"<regex('.*'):bucket>.s3.<regex('(?:{AWS_REGION_REGEX}\\.)?'):region><domain>"
PATH_WITH_REGION_PATTERN = f"s3.<regex('{AWS_REGION_REGEX}\\.'):region><domain>"

class S3VirtualHostProxyHandler:
    """
    A dispatcher Handler which can be used in a ``Router[Handler]`` that proxies incoming requests to a virtual host
    addressed S3 bucket to a path addressed URL, to allow easy routing matching the ASF specs.
    """

    def __call__(self, request: Request, **kwargs) -> Response:
        if False:
            while True:
                i = 10
        rewritten_url = self._rewrite_url(request=request, **kwargs)
        LOG.debug(f'Rewritten original host url: {request.url} to path-style url: {rewritten_url}')
        forward_to_url = urlsplit(rewritten_url)
        copied_headers = copy.copy(request.headers)
        copied_headers['Host'] = forward_to_url.netloc
        copied_headers[S3_VIRTUAL_HOST_FORWARDED_HEADER] = request.headers['host']
        with self._create_proxy() as proxy:
            forwarded = proxy.forward(request=request, forward_path=forward_to_url.path, headers=copied_headers)
        forwarded.headers.pop('date', None)
        forwarded.headers.pop('server', None)
        return forwarded

    def _create_proxy(self) -> Proxy:
        if False:
            for i in range(10):
                print('nop')
        '\n        Factory for creating proxy instance used when proxying s3 calls.\n\n        :return: a proxy instance\n        '
        return Proxy(forward_base_url=config.internal_service_url(), preserve_host=False)

    @staticmethod
    def _rewrite_url(request: Request, domain: str, bucket: str, region: str, **kwargs) -> str:
        if False:
            while True:
                i = 10
        "\n        Rewrites the url so that it can be forwarded to moto. Used for vhost-style and for any url that contains the region.\n\n        For vhost style: removes the bucket-name from the host-name and adds it as path\n        E.g. https://bucket.s3.localhost.localstack.cloud:4566 -> https://s3.localhost.localstack.cloud:4566/bucket\n        E.g. https://bucket.s3.amazonaws.com -> https://s3.localhost.localstack.cloud:4566/bucket\n\n        If the region is contained in the host-name we remove it (for now) as moto cannot handle the region correctly\n\n        :param url: the original url\n        :param domain: the domain name (anything after s3.<region>., may include a port)\n        :param bucket: the bucket name\n        :param region: the region name (includes the '.' at the end)\n        :return: re-written url as string\n        "
        splitted = urlsplit(request.url)
        raw_path = get_raw_path(request)
        if splitted.netloc.startswith(f'{bucket}.'):
            netloc = splitted.netloc.replace(f'{bucket}.', '')
            path = f'{bucket}{raw_path}'
        else:
            netloc = splitted.netloc
            path = raw_path
        if region:
            netloc = netloc.replace(f'{region}', '')
        host = domain
        edge_host = f'{LOCALHOST_HOSTNAME}:{config.GATEWAY_LISTEN[0].port}'
        if host != edge_host:
            netloc = netloc.replace(host, edge_host)
        return urlunsplit((splitted.scheme, netloc, path, splitted.query, splitted.fragment))

def add_s3_vhost_rules(router, s3_proxy_handler):
    if False:
        print('Hello World!')
    router.add(path='/', host=VHOST_REGEX_PATTERN, endpoint=s3_proxy_handler, defaults={'path': '/'})
    router.add(path='/<path:path>', host=VHOST_REGEX_PATTERN, endpoint=s3_proxy_handler)
    router.add(path="/<regex('.+'):bucket>", host=PATH_WITH_REGION_PATTERN, endpoint=s3_proxy_handler, defaults={'path': '/'})
    router.add(path="/<regex('.+'):bucket>/<path:path>", host=PATH_WITH_REGION_PATTERN, endpoint=s3_proxy_handler)

@hooks.on_infra_ready(should_load=config.LEGACY_V2_S3_PROVIDER)
def register_virtual_host_routes():
    if False:
        return 10
    '\n    Registers the S3 virtual host handler into the edge router.\n\n    '
    s3_proxy_handler = S3VirtualHostProxyHandler()
    add_s3_vhost_rules(ROUTER, s3_proxy_handler)