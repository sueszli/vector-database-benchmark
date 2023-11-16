import logging
from ray._private.utils import get_or_create_event_loop
from packaging.version import Version
import ray.dashboard.optional_utils as dashboard_optional_utils
from ray.dashboard.optional_deps import aiohttp, aiohttp_cors, hdrs
logger = logging.getLogger(__name__)
routes = dashboard_optional_utils.DashboardAgentRouteTable

class HttpServerAgent:

    def __init__(self, ip, listen_port):
        if False:
            i = 10
            return i + 15
        self.ip = ip
        self.listen_port = listen_port
        self.http_host = None
        self.http_port = None
        self.http_session = None
        self.runner = None
        if Version(aiohttp.__version__) < Version('4.0.0'):
            self.http_session = aiohttp.ClientSession(loop=get_or_create_event_loop())
        else:
            self.http_session = aiohttp.ClientSession()

    async def start(self, modules):
        for c in modules:
            dashboard_optional_utils.DashboardAgentRouteTable.bind(c)
        app = aiohttp.web.Application()
        app.add_routes(routes=routes.bound_routes())
        cors = aiohttp_cors.setup(app, defaults={'*': aiohttp_cors.ResourceOptions(allow_credentials=True, expose_headers='*', allow_methods='*', allow_headers=('Content-Type', 'X-Header'))})
        for route in list(app.router.routes()):
            cors.add(route)
        self.runner = aiohttp.web.AppRunner(app)
        await self.runner.setup()
        try:
            site = aiohttp.web.TCPSite(self.runner, '127.0.0.1' if self.ip == '127.0.0.1' else '0.0.0.0', self.listen_port)
            await site.start()
        except OSError as e:
            logger.error(f'Agent port #{self.listen_port} already in use. Failed to start agent. Ensure port #{self.listen_port} is available, and then try again.')
            raise e
        (self.http_host, self.http_port, *_) = site._server.sockets[0].getsockname()
        logger.info('Dashboard agent http address: %s:%s', self.http_host, self.http_port)
        dump_routes = [r for r in app.router.routes() if r.method != hdrs.METH_HEAD]
        for r in dump_routes:
            logger.info(r)
        logger.info('Registered %s routes.', len(dump_routes))

    async def cleanup(self):
        await self.runner.cleanup()
        await self.http_session.close()