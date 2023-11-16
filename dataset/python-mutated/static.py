from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from argparse import Namespace
from bokeh.settings import settings
from bokeh.util.logconfig import basicConfig
from ...application import Application
from ..subcommand import Subcommand
from ..util import report_server_init_errors
from .serve import base_serve_args
__all__ = ('Static',)

class Static(Subcommand):
    """ Subcommand to launch the Bokeh static server. """
    name = 'static'
    help = "Serve bokehjs' static assets (JavaScript, CSS, images, fonts, etc.)"
    args = base_serve_args

    def invoke(self, args: Namespace) -> None:
        if False:
            while True:
                i = 10
        '\n\n        '
        basicConfig(format=args.log_format, filename=args.log_file)
        log_level = settings.py_log_level(args.log_level)
        if log_level is None:
            log_level = logging.INFO
        logging.getLogger('bokeh').setLevel(log_level)
        if args.use_config is not None:
            log.info(f'Using override config file: {args.use_config}')
            settings.load_config(args.use_config)
        from bokeh.server.server import Server
        applications: dict[str, Application] = {}
        _allowed_keys = ['port', 'address']
        server_kwargs = {key: getattr(args, key) for key in _allowed_keys if getattr(args, key, None) is not None}
        with report_server_init_errors(**server_kwargs):
            server = Server(applications, **server_kwargs)
            address_string = ''
            if server.address is not None and server.address != '':
                address_string = ' address ' + server.address
            log.info(f'Starting Bokeh static server at {server.port}{address_string}')
            log.debug(f'Serving static files from: {settings.bokehjs_path()}')
            server.run_until_shutdown()