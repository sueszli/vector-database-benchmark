from prometheus_client import REGISTRY, CollectorRegistry, generate_latest
from twisted.web.resource import Resource
from twisted.web.server import Request
CONTENT_TYPE_LATEST = 'text/plain; version=0.0.4; charset=utf-8'

class MetricsResource(Resource):
    """
    Twisted ``Resource`` that serves prometheus metrics.
    """
    isLeaf = True

    def __init__(self, registry: CollectorRegistry=REGISTRY):
        if False:
            for i in range(10):
                print('nop')
        self.registry = registry

    def render_GET(self, request: Request) -> bytes:
        if False:
            while True:
                i = 10
        request.setHeader(b'Content-Type', CONTENT_TYPE_LATEST.encode('ascii'))
        response = generate_latest(self.registry)
        request.setHeader(b'Content-Length', str(len(response)))
        return response