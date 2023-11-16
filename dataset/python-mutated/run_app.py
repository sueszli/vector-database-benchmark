import argparse
import asyncio
import logging
import os
import signal
import sys
import warnings
from feeluown.app import AppMode, create_app, create_config
from feeluown.plugin import plugins_mgr
from feeluown.utils.utils import is_port_inuse, win32_is_port_binded
from feeluown.fuoexec import fuoexec_load_rcfile, fuoexec_init
from feeluown.utils import aio
from feeluown.utils.dispatch import Signal
from .base import ensure_dirs, setup_config, setup_logger
logger = logging.getLogger(__name__)

def run_app(args: argparse.Namespace):
    if False:
        i = 10
        return i + 15
    (args, config) = before_start_app(args)
    if sys.version_info.major == 3 and sys.version_info.minor >= 11:
        runner = asyncio.runners.Runner()
        try:
            runner.run(start_app(args, config))
        finally:
            runner.close()
    else:
        aio.run(start_app(args, config))

def before_start_app(args):
    if False:
        for i in range(10):
            print('nop')
    "\n    Prepare things that app depends on and initialize things which don't depend on app.\n    "
    config = create_config()
    plugins_mgr.light_scan()
    plugins_mgr.init_plugins_config(config)
    fuoexec_load_rcfile(config)
    setup_config(args, config)
    precheck(args, config)
    ensure_dirs()
    setup_logger(config)
    if AppMode.cli in AppMode(config.MODE):
        warnings.filterwarnings('ignore')
    if AppMode.gui in AppMode(config.MODE):
        if sys.platform == 'win32':
            os.environ.setdefault('QT_AUTO_SCREEN_SCALE_FACTOR', '1')
        elif sys.platform == 'darwin':
            os.environ.setdefault('QT_EVENT_DISPATCHER_CORE_FOUNDATION', '1')
        try:
            import PyQt5.QtWebEngineWidgets
        except ImportError:
            logger.info('import QtWebEngineWidgets failed')
        from feeluown.utils.compat import DefaultQEventLoopPolicy
        asyncio.set_event_loop_policy(DefaultQEventLoopPolicy())
    return (args, config)

async def start_app(args, config, sentinal=None):
    """
    The param sentinal is currently only used for unittest.
    """
    Signal.setup_aio_support()
    app = create_app(args, config)
    fuoexec_init(app)
    app.initialize()
    app.initialized.emit(app)

    def sighanlder(signum, _):
        if False:
            return 10
        logger.info('Signal %d is received', signum)
        app.exit()
    signal.signal(signal.SIGTERM, sighanlder)
    signal.signal(signal.SIGINT, sighanlder)
    if sentinal is None:
        sentinal: asyncio.Future = asyncio.Future()

    def shutdown(_):
        if False:
            for i in range(10):
                print('nop')
        if not sentinal.done():
            sentinal.set_result(0)
    app.about_to_shutdown.connect(shutdown, weak=False)
    app.load_and_apply_state()
    app.run()
    app.started.emit(app)
    await sentinal
    Signal.teardown_aio_support()

def precheck(args, config):
    if False:
        while True:
            i = 10
    err_msg = ''
    if AppMode.cli in AppMode(config.MODE):
        if args.cmd not in ('show', 'play', 'search'):
            err_msg = f"Run {args.cmd} failed, can't connect to fuo server."
    if AppMode.server in AppMode(config.MODE):
        if sys.platform == 'win32':
            host = '0.0.0.0' if config.ALLOW_LAN_CONNECT else '127.0.0.1'
            used = win32_is_port_binded(host, config.RPC_PORT) or win32_is_port_binded(host, config.PUBSUB_PORT)
        else:
            used = is_port_inuse(config.RPC_PORT) or is_port_inuse(config.PUBSUB_PORT)
        if used:
            err_msg = f'App fails to start services because either port {config.RPC_PORT} or {config.PUBSUB_PORT} was already in use. Please check if there was another FeelUOwn instance.'
    if err_msg:
        if AppMode.gui in AppMode(config.MODE):
            from PyQt5.QtWidgets import QMessageBox, QApplication
            qapp = QApplication([])
            w = QMessageBox()
            w.setText(err_msg)
            w.finished.connect(lambda _: QApplication.quit())
            w.show()
            qapp.exec()
        else:
            print(err_msg)
        sys.exit(1)