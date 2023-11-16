import logging
import re
from functools import wraps
from typing import Callable, Dict, Optional, Union
from urllib.parse import urlparse
from werkzeug.datastructures import Headers
from localstack.aws.api.s3 import BucketName, ErrorDocument, GetObjectOutput, NoSuchKey, NoSuchWebsiteConfiguration, ObjectKey, RoutingRule, RoutingRules
from localstack.aws.connect import connect_to
from localstack.aws.protocol.serializer import gen_amzn_requestid
from localstack.http import Request, Response, Router
from localstack.http.dispatcher import Handler
LOG = logging.getLogger(__name__)
STATIC_WEBSITE_HOST_REGEX = '<regex(".*"):bucket_name>.s3-website.<regex(".*"):domain>'
_leading_whitespace_re = re.compile('(^[ \t]*)(?:[ \t\n])', re.MULTILINE)

class NoSuchKeyFromErrorDocument(NoSuchKey):
    code: str = 'NoSuchKey'
    sender_fault: bool = False
    status_code: int = 404
    Key: Optional[ObjectKey]
    ErrorDocumentKey: Optional[ObjectKey]

class S3WebsiteHostingHandler:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.s3_client = connect_to().s3

    def __call__(self, request: Request, bucket_name: str, domain: str=None, path: str=None) -> Response:
        if False:
            while True:
                i = 10
        '\n        Tries to serve the key, and if an Exception is encountered, returns a generic response\n        This will allow to easily extend it to 403 exceptions\n        :param request: router Request object\n        :param bucket_name: str, bucket name\n        :param domain: str, domain name\n        :param path: the path of the request\n        :return: Response object\n        '
        if request.method != 'GET':
            return Response(_create_405_error_string(request.method, request_id=gen_amzn_requestid()), status=405)
        try:
            return self._serve_object(request, bucket_name, path)
        except (NoSuchKeyFromErrorDocument, NoSuchWebsiteConfiguration) as e:
            resource_name = e.Key if hasattr(e, 'Key') else e.BucketName
            response_body = _create_404_error_string(code=e.code, message=e.message, resource_name=resource_name, request_id=gen_amzn_requestid(), from_error_document=getattr(e, 'ErrorDocumentKey', None))
            return Response(response_body, status=e.status_code)
        except self.s3_client.exceptions.ClientError as e:
            error = e.response['Error']
            if error['Code'] not in ('NoSuchKey', 'NoSuchBucket', 'NoSuchWebsiteConfiguration'):
                raise
            resource_name = error.get('Key', error.get('BucketName'))
            response_body = _create_404_error_string(code=error['Code'], message=error['Message'], resource_name=resource_name, request_id=gen_amzn_requestid(), from_error_document=getattr(e, 'ErrorDocumentKey', None))
            return Response(response_body, status=e.response['ResponseMetadata']['HTTPStatusCode'])
        except Exception:
            LOG.exception('Exception encountered while trying to serve s3-website at %s', request.url)
            return Response(_create_500_error_string(), status=500)

    def _serve_object(self, request: Request, bucket_name: BucketName, path: str=None) -> Response:
        if False:
            print('Hello World!')
        '\n        Serves the S3 Object as a website handler. It will match routing rules set in the configuration first,\n        and redirect the request if necessary. They are specific case for handling configured index, see the docs:\n        https://docs.aws.amazon.com/AmazonS3/latest/userguide/IndexDocumentSupport.html\n        https://docs.aws.amazon.com/AmazonS3/latest/userguide/CustomErrorDocSupport.html\n        https://docs.aws.amazon.com/AmazonS3/latest/userguide/how-to-page-redirect.html\n        :param request: Request object received by the router\n        :param bucket_name: bucket name contained in the host name\n        :param path: path of the request, corresponds to the S3 Object key\n        :return: Response object, either the Object, a redirection or an error\n        '
        website_config = self.s3_client.get_bucket_website(Bucket=bucket_name)
        headers = {}
        redirection = website_config.get('RedirectAllRequestsTo')
        if redirection:
            parsed_url = urlparse(request.url)
            redirect_to = request.url.replace(parsed_url.netloc, redirection['HostName'])
            if (protocol := redirection.get('Protocol')):
                redirect_to = redirect_to.replace(parsed_url.scheme, protocol)
            headers['Location'] = redirect_to
            return Response('', status=301, headers=headers)
        object_key = path
        routing_rules = website_config.get('RoutingRules')
        if object_key and routing_rules and (rule := self._find_matching_rule(routing_rules, object_key=object_key)):
            redirect_response = self._get_redirect_from_routing_rule(request, rule)
            return redirect_response
        is_folder = request.url[-1] == '/'
        if not object_key or is_folder:
            index_key = website_config['IndexDocument']['Suffix']
            object_key = f'{object_key}{index_key}' if object_key else index_key
        try:
            s3_object = self.s3_client.get_object(Bucket=bucket_name, Key=object_key)
        except self.s3_client.exceptions.NoSuchKey:
            if not is_folder:
                index_key = website_config['IndexDocument']['Suffix']
                try:
                    self.s3_client.head_object(Bucket=bucket_name, Key=f'{object_key}/{index_key}')
                    return Response('', status=302, headers={'Location': f'/{object_key}/'})
                except self.s3_client.exceptions.ClientError:
                    pass
            if routing_rules and (rule := self._find_matching_rule(routing_rules, object_key=object_key, error_code=404)):
                redirect_response = self._get_redirect_from_routing_rule(request, rule)
                return redirect_response
            if (error_document := website_config.get('ErrorDocument')):
                return self._return_error_document(error_document=error_document, bucket=bucket_name, missing_key=object_key)
            else:
                raise
        if (website_redirect_location := s3_object.get('WebsiteRedirectLocation')):
            headers['Location'] = website_redirect_location
            return Response('', status=301, headers=headers)
        if self._check_if_headers(request.headers, s3_object=s3_object):
            return Response('', status=304)
        headers = self._get_response_headers_from_object(s3_object)
        return Response(s3_object['Body'], headers=headers)

    def _return_error_document(self, error_document: ErrorDocument, bucket: BucketName, missing_key: ObjectKey) -> Response:
        if False:
            while True:
                i = 10
        '\n        Try to retrieve the configured ErrorDocument and return the response with its body\n        https://docs.aws.amazon.com/AmazonS3/latest/userguide/CustomErrorDocSupport.html\n        :param error_document: the ErrorDocument from the bucket WebsiteConfiguration\n        :param bucket: the bucket name\n        :param missing_key: the missing key not found in the bucket\n        :return: a Response, either a redirection or containing the Body of the ErrorDocument\n        :raises NoSuchKeyFromErrorDocument if the ErrorDocument is not found\n        '
        headers = {}
        error_key = error_document['Key']
        try:
            s3_object = self.s3_client.get_object(Bucket=bucket, Key=error_key)
            if (website_redirect_location := s3_object.get('WebsiteRedirectLocation')):
                headers['Location'] = website_redirect_location
                return Response('', status=301, headers=headers)
            headers = self._get_response_headers_from_object(s3_object)
            return Response(s3_object['Body'], status=404, headers=headers)
        except self.s3_client.exceptions.NoSuchKey:
            raise NoSuchKeyFromErrorDocument('The specified key does not exist.', Key=missing_key, ErrorDocumentKey=error_key)

    @staticmethod
    def _get_response_headers_from_object(get_object_response: GetObjectOutput) -> Dict[str, str]:
        if False:
            i = 10
            return i + 15
        '\n        Only return some headers from the S3 Object\n        :param get_object_response: the response from S3.GetObject\n        :return: headers from the object to be part of the response\n        '
        response_headers = {}
        if (content_type := get_object_response.get('ContentType')):
            response_headers['Content-Type'] = content_type
        if (etag := get_object_response.get('ETag')):
            response_headers['etag'] = etag
        return response_headers

    @staticmethod
    def _check_if_headers(headers: Headers, s3_object: GetObjectOutput) -> bool:
        if False:
            print('Hello World!')
        etag = s3_object.get('ETag')
        if 'if-none-match' in headers and etag and (etag in headers['if-none-match']):
            return True

    @staticmethod
    def _find_matching_rule(routing_rules: RoutingRules, object_key: ObjectKey, error_code: int=None) -> Union[RoutingRule, None]:
        if False:
            return 10
        '\n        Iterate over the routing rules set in the configuration, and return the first that match the key name and/or the\n        error code (in the 4XX range).\n        :param routing_rules: RoutingRules part of WebsiteConfiguration\n        :param object_key: ObjectKey\n        :param error_code: error code of the Response in the 4XX range\n        :return: a RoutingRule if matched, or None\n        '
        for rule in routing_rules:
            if (condition := rule.get('Condition')):
                prefix = condition.get('KeyPrefixEquals')
                return_http_code = condition.get('HttpErrorCodeReturnedEquals')
                if prefix and return_http_code:
                    if object_key.startswith(prefix) and error_code == int(return_http_code):
                        return rule
                    else:
                        continue
                elif prefix and object_key.startswith(prefix):
                    return rule
                elif return_http_code and error_code == int(return_http_code):
                    return rule
            else:
                return rule

    @staticmethod
    def _get_redirect_from_routing_rule(request: Request, routing_rule: RoutingRule) -> Response:
        if False:
            i = 10
            return i + 15
        '\n        Return a redirect Response object created with the different parameters set in the RoutingRule\n        :param request: the original Request object received from the router\n        :param routing_rule: a RoutingRule from the WebsiteConfiguration\n        :return: a redirect Response\n        '
        parsed_url = urlparse(request.url)
        redirect_to = request.url
        redirect = routing_rule['Redirect']
        if (host_name := redirect.get('HostName')):
            redirect_to = redirect_to.replace(parsed_url.netloc, host_name)
        if (protocol := redirect.get('Protocol')):
            redirect_to = redirect_to.replace(parsed_url.scheme, protocol)
        if (redirect_to_key := redirect.get('ReplaceKeyWith')):
            redirect_to = redirect_to.replace(parsed_url.path, f'/{redirect_to_key}')
        elif 'ReplaceKeyPrefixWith' in redirect:
            matched_prefix = routing_rule['Condition'].get('KeyPrefixEquals', '')
            redirect_to = redirect_to.replace(matched_prefix, redirect.get('ReplaceKeyPrefixWith'), 1)
        return Response('', headers={'Location': redirect_to}, status=redirect.get('HttpRedirectCode', 301))

def register_website_hosting_routes(router: Router[Handler], handler: S3WebsiteHostingHandler=None):
    if False:
        print('Hello World!')
    '\n    Registers the S3 website hosting handler into the given router.\n    :param handler: an S3WebsiteHosting handler\n    :param router: the router to add the handlers into.\n    '
    handler = handler or S3WebsiteHostingHandler()
    router.add(path='/', host=STATIC_WEBSITE_HOST_REGEX, endpoint=handler)
    router.add(path='/<path:path>', host=STATIC_WEBSITE_HOST_REGEX, endpoint=handler)

def _flatten_html_response(fn: Callable[[...], str]):
    if False:
        i = 10
        return i + 15

    @wraps(fn)
    def wrapper(*args, **kwargs) -> str:
        if False:
            while True:
                i = 10
        r = fn(*args, **kwargs)
        return re.sub(_leading_whitespace_re, '', r)
    return wrapper

@_flatten_html_response
def _create_404_error_string(code: str, message: str, resource_name: str, request_id: str, from_error_document: str=None) -> str:
    if False:
        print('Hello World!')
    resource_key = 'Key' if 'Key' in code else 'BucketName'
    return f'<html>\n    <head><title>404 Not Found</title></head>\n    <body>\n        <h1>404 Not Found</h1>\n        <ul>\n            <li>Code: {code}</li>\n            <li>Message: {message}</li>\n            <li>{resource_key}: {resource_name}</li>\n            <li>RequestId: {request_id}</li>\n            <li>HostId: h6t23Wl2Ndijztq+COn9kvx32omFVRLLtwk36D6+2/CIYSey+Uox6kBxRgcnAASsgnGwctU6zzU=</li>\n        </ul>\n        {_create_nested_404_error_string(from_error_document)}\n        <hr/>\n    </body>\n</html>\n'

def _create_nested_404_error_string(error_document_key: str) -> str:
    if False:
        i = 10
        return i + 15
    if not error_document_key:
        return ''
    return f'<h3>An Error Occurred While Attempting to Retrieve a Custom Error Document</h3>\n        <ul>\n            <li>Code: NoSuchKey</li>\n            <li>Message: The specified key does not exist.</li>\n            <li>Key: {error_document_key}</li>\n        </ul>\n    '

@_flatten_html_response
def _create_405_error_string(method: str, request_id: str) -> str:
    if False:
        return 10
    return f'<html>\n    <head><title>405 Method Not Allowed</title></head>\n    <body>\n        <h1>405 Method Not Allowed</h1>\n        <ul>\n            <li>Code: MethodNotAllowed</li>\n            <li>Message: The specified method is not allowed against this resource.</li>\n            <li>Method: {method.upper()}</li>\n            <li>ResourceType: OBJECT</li>\n            <li>RequestId: {request_id}</li>\n            <li>HostId: h6t23Wl2Ndijztq+COn9kvx32omFVRLLtwk36D6+2/CIYSey+Uox6kBxRgcnAASsgnGwctU6zzU=</li>\n        </ul>\n        <hr/>\n    </body>\n</html>\n'

@_flatten_html_response
def _create_500_error_string() -> str:
    if False:
        while True:
            i = 10
    return '<html>\n        <head><title>500 Service Error</title></head>\n        <body>\n            <h1>500 Service Error</h1>\n            <hr/>\n        </body>\n    </html>\n    '