import base64
import email.message
import hashlib
import io
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Callable, Dict, List, Optional, Pattern, Tuple
from urllib.request import Request
from spack.oci.image import Digest
from spack.oci.opener import OCIAuthHandler

class MockHTTPResponse(io.IOBase):
    """This is a mock HTTP response, which implements part of http.client.HTTPResponse"""

    def __init__(self, status, reason, headers=None, body=None):
        if False:
            i = 10
            return i + 15
        self.msg = None
        self.version = 11
        self.url = None
        self.headers = email.message.EmailMessage()
        self.status = status
        self.code = status
        self.reason = reason
        self.debuglevel = 0
        self._body = body
        if headers is not None:
            for (key, value) in headers.items():
                self.headers[key] = value

    @classmethod
    def with_json(cls, status, reason, headers=None, body=None):
        if False:
            print('Hello World!')
        'Create a mock HTTP response with JSON string as body'
        body = io.BytesIO(json.dumps(body).encode('utf-8'))
        return cls(status, reason, headers, body)

    def read(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return self._body.read(*args, **kwargs)

    def getheader(self, name, default=None):
        if False:
            print('Hello World!')
        self.headers.get(name, default)

    def getheaders(self):
        if False:
            return 10
        return self.headers.items()

    def fileno(self):
        if False:
            return 10
        return 0

    def getcode(self):
        if False:
            while True:
                i = 10
        return self.status

    def info(self):
        if False:
            return 10
        return self.headers

class MiddlewareError(Exception):
    """Thrown in a handler to return a response early."""

    def __init__(self, response: MockHTTPResponse):
        if False:
            for i in range(10):
                print('nop')
        self.response = response

class Router:
    """This class is a small router for requests to the OCI registry.

    It is used to dispatch requests to a handler, and middleware can be
    used to transform requests, as well as return responses early
    (e.g. for authentication)."""

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.routes: List[Tuple[str, Pattern, Callable]] = []
        self.middleware: List[Callable[[Request], Request]] = []

    def handle(self, req: Request) -> MockHTTPResponse:
        if False:
            for i in range(10):
                print('nop')
        'Dispatch a request to a handler.'
        result = urllib.parse.urlparse(req.full_url)
        try:
            for handler in self.middleware:
                req = handler(req)
        except MiddlewareError as e:
            return e.response
        for (method, path_regex, handler) in self.routes:
            if method != req.get_method():
                continue
            match = re.fullmatch(path_regex, result.path)
            if not match:
                continue
            return handler(req, **match.groupdict())
        return MockHTTPResponse(404, 'Not found')

    def register(self, method, path: str, handler: Callable):
        if False:
            print('Hello World!')
        self.routes.append((method, re.compile(path), handler))

    def add_middleware(self, handler: Callable[[Request], Request]):
        if False:
            while True:
                i = 10
        self.middleware.append(handler)

class DummyServer:

    def __init__(self, domain: str) -> None:
        if False:
            i = 10
            return i + 15
        self.domain = domain
        self.requests: List[Tuple[str, str]] = []
        self.router = Router()
        self.router.add_middleware(self.log_request)

    def handle(self, req: Request) -> MockHTTPResponse:
        if False:
            while True:
                i = 10
        return self.router.handle(req)

    def log_request(self, req: Request):
        if False:
            i = 10
            return i + 15
        path = urllib.parse.urlparse(req.full_url).path
        self.requests.append((req.get_method(), path))
        return req

    def clear_log(self):
        if False:
            return 10
        self.requests = []

class InMemoryOCIRegistry(DummyServer):
    """This implements the basic OCI registry API, but in memory.

    It supports two types of blob uploads:
    1. POST + PUT: the client first starts a session with POST, then does a large PUT request
    2. POST: the client does a single POST request with the whole blob

    Option 2 is not supported by all registries, so we allow to disable it,
    with allow_single_post=False.

    A third option is to use the chunked upload, but this is not implemented here, because
    it's typically a major performance hit in upload speed, so we're not using it in Spack."""

    def __init__(self, domain: str, allow_single_post: bool=True) -> None:
        if False:
            return 10
        super().__init__(domain)
        self.router.register('GET', '/v2/', self.index)
        self.router.register('HEAD', '/v2/(?P<name>.+)/blobs/(?P<digest>.+)', self.head_blob)
        self.router.register('POST', '/v2/(?P<name>.+)/blobs/uploads/', self.start_session)
        self.router.register('PUT', '/upload', self.put_session)
        self.router.register('PUT', '/v2/(?P<name>.+)/manifests/(?P<ref>.+)', self.put_manifest)
        self.router.register('GET', '/v2/(?P<name>.+)/manifests/(?P<ref>.+)', self.get_manifest)
        self.router.register('GET', '/v2/(?P<name>.+)/blobs/(?P<digest>.+)', self.get_blob)
        self.router.register('GET', '/v2/(?P<name>.+)/tags/list', self.list_tags)
        self.allow_single_post = allow_single_post
        self.sessions: Dict[str, str] = {}
        self.blobs: Dict[str, bytes] = {}
        self.manifests: Dict[Tuple[str, str], Dict] = {}

    def index(self, req: Request):
        if False:
            return 10
        return MockHTTPResponse.with_json(200, 'OK', body={})

    def head_blob(self, req: Request, name: str, digest: str):
        if False:
            print('Hello World!')
        if digest in self.blobs:
            return MockHTTPResponse(200, 'OK', headers={'Content-Length': '1234'})
        return MockHTTPResponse(404, 'Not found')

    def get_blob(self, req: Request, name: str, digest: str):
        if False:
            print('Hello World!')
        if digest in self.blobs:
            return MockHTTPResponse(200, 'OK', body=io.BytesIO(self.blobs[digest]))
        return MockHTTPResponse(404, 'Not found')

    def start_session(self, req: Request, name: str):
        if False:
            for i in range(10):
                print('nop')
        id = str(uuid.uuid4())
        self.sessions[id] = name
        result = urllib.parse.urlparse(req.full_url)
        query = urllib.parse.parse_qs(result.query)
        if self.allow_single_post and 'digest' in query:
            return self.handle_upload(req, name=name, digest=Digest.from_string(query['digest'][0]))
        return MockHTTPResponse(202, 'Accepted', headers={'Location': f'/upload?uuid={id}'})

    def put_session(self, req: Request):
        if False:
            i = 10
            return i + 15
        result = urllib.parse.urlparse(req.full_url)
        query = urllib.parse.parse_qs(result.query)
        assert 'uuid' in query and len(query['uuid']) == 1
        assert 'digest' in query and len(query['digest']) == 1
        id = query['uuid'][0]
        assert id in self.sessions
        (name, digest) = (self.sessions[id], Digest.from_string(query['digest'][0]))
        response = self.handle_upload(req, name=name, digest=digest)
        del self.sessions[id]
        return response

    def put_manifest(self, req: Request, name: str, ref: str):
        if False:
            print('Hello World!')
        content_type = req.get_header('Content-type')
        assert content_type in ('application/vnd.oci.image.manifest.v1+json', 'application/vnd.oci.image.index.v1+json')
        index_or_manifest = json.loads(self._require_data(req))
        if content_type == 'application/vnd.oci.image.manifest.v1+json':
            for layer in index_or_manifest['layers']:
                assert layer['digest'] in self.blobs, 'Missing blob while uploading manifest'
        else:
            for manifest in index_or_manifest['manifests']:
                assert (name, manifest['digest']) in self.manifests, 'Missing manifest while uploading index'
        self.manifests[name, ref] = index_or_manifest
        return MockHTTPResponse(201, 'Created', headers={'Location': f'/v2/{name}/manifests/{ref}'})

    def get_manifest(self, req: Request, name: str, ref: str):
        if False:
            while True:
                i = 10
        if (name, ref) not in self.manifests:
            return MockHTTPResponse(404, 'Not found')
        manifest_or_index = self.manifests[name, ref]
        return MockHTTPResponse.with_json(200, 'OK', headers={'Content-type': manifest_or_index['mediaType']}, body=manifest_or_index)

    def _require_data(self, req: Request) -> bytes:
        if False:
            i = 10
            return i + 15
        "Extract request.data, it's type remains a mystery"
        assert req.data is not None
        if hasattr(req.data, 'read'):
            return req.data.read()
        elif isinstance(req.data, bytes):
            return req.data
        raise ValueError('req.data should be bytes or have a read() method')

    def handle_upload(self, req: Request, name: str, digest: Digest):
        if False:
            while True:
                i = 10
        'Verify the digest, save the blob, return created status'
        data = self._require_data(req)
        assert hashlib.sha256(data).hexdigest() == digest.digest
        self.blobs[str(digest)] = data
        return MockHTTPResponse(201, 'Created', headers={'Location': f'/v2/{name}/blobs/{digest}'})

    def list_tags(self, req: Request, name: str):
        if False:
            print('Hello World!')
        tags = [_tag for (_name, _tag) in self.manifests.keys() if _name == name and ':' not in _tag]
        tags.sort()
        return MockHTTPResponse.with_json(200, 'OK', body={'tags': tags})

class DummyServerUrllibHandler(urllib.request.BaseHandler):
    """Glue between urllib and DummyServer, routing requests to
    the correct mock server for a given domain."""

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.servers: Dict[str, DummyServer] = {}

    def add_server(self, domain: str, api: DummyServer):
        if False:
            i = 10
            return i + 15
        self.servers[domain] = api
        return self

    def https_open(self, req: Request):
        if False:
            print('Hello World!')
        domain = urllib.parse.urlparse(req.full_url).netloc
        if domain not in self.servers:
            return MockHTTPResponse(404, 'Not found')
        return self.servers[domain].handle(req)

class InMemoryOCIRegistryWithAuth(InMemoryOCIRegistry):
    """This is another in-memory OCI registry, but it requires authentication."""

    def __init__(self, domain, token: Optional[str], realm: str, allow_single_post: bool=True) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(domain, allow_single_post)
        self.token = token
        self.realm = realm
        self.router.add_middleware(self.authenticate)

    def authenticate(self, req: Request):
        if False:
            print('Hello World!')
        authorization = req.get_header('Authorization')
        if authorization is None:
            raise MiddlewareError(self.unauthorized())
        assert authorization.startswith('Bearer ')
        token = authorization[7:]
        if token != self.token:
            raise MiddlewareError(self.unauthorized())
        return req

    def unauthorized(self):
        if False:
            i = 10
            return i + 15
        return MockHTTPResponse(401, 'Unauthorized', {'www-authenticate': f'Bearer realm="{self.realm}",service="{self.domain}",scope="repository:spack-registry:pull,push"'})

class MockBearerTokenServer(DummyServer):
    """Simulates a basic server that hands out bearer tokens
    at the /login endpoint for the following services:
    public.example.com, which doesn't require Basic Auth
    private.example.com, which requires Basic Auth, with user:pass
    """

    def __init__(self, domain: str) -> None:
        if False:
            while True:
                i = 10
        super().__init__(domain)
        self.router.register('GET', '/login', self.login)

    def login(self, req: Request):
        if False:
            i = 10
            return i + 15
        url = urllib.parse.urlparse(req.full_url)
        query_params = urllib.parse.parse_qs(url.query)
        assert query_params['client_id'] == ['spack']
        assert len(query_params['service']) == 1
        assert query_params['scope'] == ['repository:spack-registry:pull,push']
        service = query_params['service'][0]
        if service == 'public.example.com':
            return self.public_auth(req)
        elif service == 'private.example.com':
            return self.private_auth(req)
        return MockHTTPResponse(404, 'Not found')

    def public_auth(self, req: Request):
        if False:
            for i in range(10):
                print('nop')
        assert req.get_header('Authorization') is None
        return MockHTTPResponse.with_json(200, 'OK', body={'token': 'public_token'})

    def private_auth(self, req: Request):
        if False:
            i = 10
            return i + 15
        auth_value = req.get_header('Authorization')
        if auth_value is None or not auth_value.startswith('Basic ') or base64.b64decode(auth_value[6:]) != b'user:pass':
            return MockHTTPResponse(401, 'Unauthorized')
        return MockHTTPResponse.with_json(200, 'OK', body={'token': 'private_token'})

def create_opener(*servers: DummyServer, credentials_provider=None):
    if False:
        return 10
    'Creates a mock opener, that can be used to fake requests to a list\n    of servers.'
    opener = urllib.request.OpenerDirector()
    handler = DummyServerUrllibHandler()
    for server in servers:
        handler.add_server(server.domain, server)
    opener.add_handler(handler)
    opener.add_handler(urllib.request.HTTPDefaultErrorHandler())
    opener.add_handler(urllib.request.HTTPErrorProcessor())
    if credentials_provider is not None:
        opener.add_handler(OCIAuthHandler(credentials_provider))
    return opener