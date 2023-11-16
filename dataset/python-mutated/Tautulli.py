import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
from future.builtins import str
import appdirs
import argparse
import datetime
import locale
import pytz
import signal
import shutil
import time
import threading
import tzlocal
import plexpy
from plexpy import common, config, database, helpers, logger, webstart
if common.PLATFORM == 'Windows':
    from plexpy import windows
elif common.PLATFORM == 'Darwin':
    from plexpy import macos
signal.signal(signal.SIGINT, plexpy.sig_handler)
signal.signal(signal.SIGTERM, plexpy.sig_handler)

def main():
    if False:
        while True:
            i = 10
    '\n    Tautulli application entry point. Parses arguments, setups encoding and\n    initializes the application.\n    '
    if hasattr(sys, 'frozen') and hasattr(sys, '_MEIPASS'):
        plexpy.FROZEN = True
        plexpy.FULL_PATH = os.path.abspath(sys.executable)
        plexpy.PROG_DIR = sys._MEIPASS
    else:
        plexpy.FULL_PATH = os.path.abspath(__file__)
        plexpy.PROG_DIR = os.path.dirname(plexpy.FULL_PATH)
    plexpy.ARGS = sys.argv[1:]
    plexpy.SYS_PLATFORM = sys.platform
    plexpy.SYS_ENCODING = None
    try:
        locale.setlocale(locale.LC_ALL, '')
        (plexpy.SYS_LANGUAGE, plexpy.SYS_ENCODING) = locale.getdefaultlocale()
    except (locale.Error, IOError):
        pass
    if not plexpy.SYS_ENCODING or plexpy.SYS_ENCODING in ('ANSI_X3.4-1968', 'US-ASCII', 'ASCII'):
        plexpy.SYS_ENCODING = 'UTF-8'
    parser = argparse.ArgumentParser(description='A Python based monitoring and tracking tool for Plex Media Server.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase console logging verbosity')
    parser.add_argument('-q', '--quiet', action='store_true', help='Turn off console logging')
    parser.add_argument('-d', '--daemon', action='store_true', help='Run as a daemon')
    parser.add_argument('-p', '--port', type=int, help='Force Tautulli to run on a specified port')
    parser.add_argument('--dev', action='store_true', help='Start Tautulli in the development environment')
    parser.add_argument('--datadir', help='Specify a directory where to store your data files')
    parser.add_argument('--config', help='Specify a config file to use')
    parser.add_argument('--nolaunch', action='store_true', help='Prevent browser from launching on startup')
    parser.add_argument('--pidfile', help='Create a pid file (only relevant when running as a daemon)')
    parser.add_argument('--nofork', action='store_true', help='Start Tautulli as a service, do not fork when restarting')
    args = parser.parse_args()
    if args.verbose:
        plexpy.VERBOSE = True
    if args.quiet:
        plexpy.QUIET = True
    logger.initLogger(console=not plexpy.QUIET, log_dir=False, verbose=True)
    try:
        plexpy.SYS_TIMEZONE = tzlocal.get_localzone()
    except (pytz.UnknownTimeZoneError, LookupError, ValueError) as e:
        logger.error('Could not determine system timezone: %s' % e)
        plexpy.SYS_TIMEZONE = pytz.UTC
    plexpy.SYS_UTC_OFFSET = datetime.datetime.now(plexpy.SYS_TIMEZONE).strftime('%z')
    if helpers.bool_true(os.getenv('TAUTULLI_DOCKER', False)):
        plexpy.DOCKER = True
        plexpy.DOCKER_MOUNT = not os.path.isfile('/config/DOCKER')
    if helpers.bool_true(os.getenv('TAUTULLI_SNAP', False)):
        plexpy.SNAP = True
    if args.dev:
        plexpy.DEV = True
        logger.debug('Tautulli is running in the dev environment.')
    if args.daemon:
        if sys.platform == 'win32':
            logger.warn('Daemonizing not supported under Windows, starting normally')
        else:
            plexpy.DAEMON = True
            plexpy.QUIET = True
    if args.nofork:
        plexpy.NOFORK = True
        logger.info('Tautulli is running as a service, it will not fork when restarted.')
    if args.pidfile:
        plexpy.PIDFILE = str(args.pidfile)
        if os.path.exists(plexpy.PIDFILE):
            try:
                with open(plexpy.PIDFILE, 'r') as fp:
                    pid = int(fp.read())
            except IOError as e:
                raise SystemExit('Unable to read PID file: %s', e)
            try:
                os.kill(pid, 0)
            except OSError:
                logger.warn("PID file '%s' already exists, but PID %d is not running. Ignoring PID file." % (plexpy.PIDFILE, pid))
            else:
                raise SystemExit("PID file '%s' already exists. Exiting." % plexpy.PIDFILE)
        if plexpy.DAEMON:
            plexpy.CREATEPID = True
            try:
                with open(plexpy.PIDFILE, 'w') as fp:
                    fp.write('pid\n')
            except IOError as e:
                raise SystemExit('Unable to write PID file: %s', e)
        else:
            logger.warn('Not running in daemon mode. PID file creation disabled.')
    if args.datadir:
        plexpy.DATA_DIR = args.datadir
    elif plexpy.FROZEN:
        plexpy.DATA_DIR = appdirs.user_data_dir('Tautulli', False)
    else:
        plexpy.DATA_DIR = plexpy.PROG_DIR
    if plexpy.SNAP:
        snap_common = os.environ['SNAP_COMMON']
        old_data_dir = os.path.join(snap_common, 'Tautulli')
        if os.path.exists(old_data_dir) and os.listdir(old_data_dir):
            plexpy.SNAP_MIGRATE = True
            logger.info('Migrating Snap user data.')
            shutil.move(old_data_dir, plexpy.DATA_DIR)
    if args.config:
        config_file = args.config
    else:
        config_file = os.path.join(plexpy.DATA_DIR, config.FILENAME)
    if not os.path.exists(plexpy.DATA_DIR):
        try:
            os.makedirs(plexpy.DATA_DIR)
        except OSError:
            raise SystemExit('Could not create data directory: ' + plexpy.DATA_DIR + '. Exiting....')
    test_file = os.path.join(plexpy.DATA_DIR, '.TEST')
    try:
        with open(test_file, 'w'):
            pass
    except IOError:
        raise SystemExit('Cannot write to the data directory: ' + plexpy.DATA_DIR + '. Exiting...')
    finally:
        try:
            os.remove(test_file)
        except OSError:
            pass
    plexpy.DB_FILE = os.path.join(plexpy.DATA_DIR, database.FILENAME)
    if os.path.isfile(os.path.join(plexpy.DATA_DIR, 'plexpy.db')) and (not os.path.isfile(os.path.join(plexpy.DATA_DIR, plexpy.DB_FILE))):
        try:
            os.rename(os.path.join(plexpy.DATA_DIR, 'plexpy.db'), plexpy.DB_FILE)
        except OSError as e:
            raise SystemExit('Unable to rename plexpy.db to tautulli.db: %s', e)
    if plexpy.DAEMON:
        plexpy.daemonize()
    plexpy.initialize(config_file)
    plexpy.start()
    if args.port:
        plexpy.HTTP_PORT = args.port
        logger.info('Using forced web server port: %i', plexpy.HTTP_PORT)
    else:
        plexpy.HTTP_PORT = int(plexpy.CONFIG.HTTP_PORT)
    webstart.start()
    if common.PLATFORM == 'Windows':
        if plexpy.CONFIG.SYS_TRAY_ICON:
            plexpy.WIN_SYS_TRAY_ICON = windows.WindowsSystemTray()
            plexpy.WIN_SYS_TRAY_ICON.start()
        windows.set_startup()
    elif common.PLATFORM == 'Darwin':
        macos.set_startup()
    if plexpy.CONFIG.LAUNCH_BROWSER and (not args.nolaunch) and (not plexpy.DEV):
        plexpy.launch_browser(plexpy.CONFIG.HTTP_HOST, plexpy.HTTP_PORT, plexpy.HTTP_ROOT)
    if common.PLATFORM == 'Darwin' and plexpy.CONFIG.SYS_TRAY_ICON:
        if not macos.HAS_PYOBJC:
            logger.warn('The pyobjc module is missing. Install this module to enable the MacOS menu bar icon.')
            plexpy.CONFIG.SYS_TRAY_ICON = False
        if plexpy.CONFIG.SYS_TRAY_ICON:
            thread = threading.Thread(target=wait)
            thread.daemon = True
            thread.start()
            plexpy.MAC_SYS_TRAY_ICON = macos.MacOSSystemTray()
            plexpy.MAC_SYS_TRAY_ICON.start()
        else:
            wait()
    else:
        wait()

def wait():
    if False:
        return 10
    logger.info('Tautulli is ready!')
    while True:
        if not plexpy.SIGNAL:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                plexpy.SIGNAL = 'shutdown'
        else:
            logger.info('Received signal: %s', plexpy.SIGNAL)
            if plexpy.SIGNAL == 'shutdown':
                plexpy.shutdown()
            elif plexpy.SIGNAL == 'restart':
                plexpy.shutdown(restart=True)
            elif plexpy.SIGNAL == 'checkout':
                plexpy.shutdown(restart=True, checkout=True)
            elif plexpy.SIGNAL == 'reset':
                plexpy.shutdown(restart=True, reset=True)
            elif plexpy.SIGNAL == 'update':
                plexpy.shutdown(restart=True, update=True)
            else:
                logger.error('Unknown signal. Shutting down...')
                plexpy.shutdown()
            plexpy.SIGNAL = None
if __name__ == '__main__':
    main()