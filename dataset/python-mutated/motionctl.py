import errno
import logging
import os.path
import re
import signal
import subprocess
import time
from tornado.ioloop import IOLoop
import mediafiles
import powerctl
import settings
import update
import utils
_MOTION_CONTROL_TIMEOUT = 5
_started = False
_motion_binary_cache = None
_motion_detected = {}

def find_motion():
    if False:
        for i in range(10):
            print('nop')
    global _motion_binary_cache
    if _motion_binary_cache:
        return _motion_binary_cache
    if settings.MOTION_BINARY:
        if os.path.exists(settings.MOTION_BINARY):
            binary = settings.MOTION_BINARY
        else:
            return (None, None)
    else:
        try:
            binary = subprocess.check_output(['which', 'motion'], stderr=utils.DEV_NULL).strip()
        except subprocess.CalledProcessError:
            return (None, None)
    try:
        help = subprocess.check_output(binary + ' -h || true', shell=True)
    except subprocess.CalledProcessError:
        return (None, None)
    result = re.findall('motion Version ([^,]+)', help, re.IGNORECASE)
    version = result and result[0] or ''
    logging.debug('using motion version %s' % version)
    _motion_binary_cache = (binary, version)
    return _motion_binary_cache

def start(deferred=False):
    if False:
        print('Hello World!')
    import config
    import mjpgclient
    if deferred:
        io_loop = IOLoop.instance()
        io_loop.add_callback(start, deferred=False)
    global _started
    _started = True
    enabled_local_motion_cameras = config.get_enabled_local_motion_cameras()
    if running() or not enabled_local_motion_cameras:
        return
    logging.debug('starting motion')
    program = find_motion()
    if not program[0]:
        raise Exception('motion executable could not be found')
    (program, version) = program
    logging.debug('starting motion binary "%s"' % program)
    motion_config_path = os.path.join(settings.CONF_PATH, 'motion.conf')
    motion_log_path = os.path.join(settings.LOG_PATH, 'motion.log')
    motion_pid_path = os.path.join(settings.RUN_PATH, 'motion.pid')
    args = [program, '-n', '-c', motion_config_path, '-d']
    if settings.LOG_LEVEL <= logging.DEBUG:
        args.append('9')
    elif settings.LOG_LEVEL <= logging.WARN:
        args.append('5')
    elif settings.LOG_LEVEL <= logging.ERROR:
        args.append('4')
    else:
        args.append('1')
    log_file = open(motion_log_path, 'w')
    process = subprocess.Popen(args, stdout=log_file, stderr=log_file, close_fds=True, cwd=settings.CONF_PATH)
    for i in xrange(20):
        time.sleep(0.1)
        exit_code = process.poll()
        if exit_code is not None and exit_code != 0:
            raise Exception('motion failed to start')
    pid = process.pid
    with open(motion_pid_path, 'w') as f:
        f.write(str(pid) + '\n')
    _disable_initial_motion_detection()
    if not settings.MJPG_CLIENT_IDLE_TIMEOUT:
        logging.debug('creating default mjpg clients for local cameras')
        for camera in enabled_local_motion_cameras:
            mjpgclient.get_jpg(camera['@id'])

def stop(invalidate=False):
    if False:
        while True:
            i = 10
    import mjpgclient
    global _started
    _started = False
    if not running():
        return
    logging.debug('stopping motion')
    mjpgclient.close_all(invalidate=invalidate)
    pid = _get_pid()
    if pid is not None:
        try:
            os.kill(pid, signal.SIGTERM)
            for i in xrange(50):
                os.waitpid(pid, os.WNOHANG)
                time.sleep(0.1)
            os.kill(pid, signal.SIGKILL)
            for i in xrange(20):
                time.sleep(0.1)
                os.waitpid(pid, os.WNOHANG)
            if settings.ENABLE_REBOOT:
                logging.error('could not terminate the motion process')
                powerctl.reboot()
            else:
                raise Exception('could not terminate the motion process')
        except OSError as e:
            if e.errno not in (errno.ESRCH, errno.ECHILD):
                raise

def running():
    if False:
        for i in range(10):
            print('nop')
    pid = _get_pid()
    if pid is None:
        return False
    try:
        os.waitpid(pid, os.WNOHANG)
        os.kill(pid, 0)
        return True
    except OSError as e:
        if e.errno not in (errno.ESRCH, errno.ECHILD):
            raise
    return False

def started():
    if False:
        i = 10
        return i + 15
    return _started

def get_motion_detection(camera_id, callback):
    if False:
        while True:
            i = 10
    from tornado.httpclient import HTTPRequest, AsyncHTTPClient
    motion_camera_id = camera_id_to_motion_camera_id(camera_id)
    if motion_camera_id is None:
        error = 'could not find motion camera id for camera with id %s' % camera_id
        logging.error(error)
        return callback(error=error)
    url = 'http://127.0.0.1:%(port)s/%(id)s/detection/status' % {'port': settings.MOTION_CONTROL_PORT, 'id': motion_camera_id}

    def on_response(response):
        if False:
            i = 10
            return i + 15
        if response.error:
            return callback(error=utils.pretty_http_error(response))
        enabled = bool(response.body.lower().count('active'))
        logging.debug('motion detection is %(what)s for camera with id %(id)s' % {'what': ['disabled', 'enabled'][enabled], 'id': camera_id})
        callback(enabled)
    request = HTTPRequest(url, connect_timeout=_MOTION_CONTROL_TIMEOUT, request_timeout=_MOTION_CONTROL_TIMEOUT)
    http_client = AsyncHTTPClient()
    http_client.fetch(request, callback=on_response)

def set_motion_detection(camera_id, enabled):
    if False:
        return 10
    from tornado.httpclient import HTTPRequest, AsyncHTTPClient
    motion_camera_id = camera_id_to_motion_camera_id(camera_id)
    if motion_camera_id is None:
        return logging.error('could not find motion camera id for camera with id %s' % camera_id)
    if not enabled:
        _motion_detected[camera_id] = False
    logging.debug('%(what)s motion detection for camera with id %(id)s' % {'what': ['disabling', 'enabling'][enabled], 'id': camera_id})
    url = 'http://127.0.0.1:%(port)s/%(id)s/detection/%(enabled)s' % {'port': settings.MOTION_CONTROL_PORT, 'id': motion_camera_id, 'enabled': ['pause', 'start'][enabled]}

    def on_response(response):
        if False:
            i = 10
            return i + 15
        if response.error:
            logging.error('failed to %(what)s motion detection for camera with id %(id)s: %(msg)s' % {'what': ['disable', 'enable'][enabled], 'id': camera_id, 'msg': utils.pretty_http_error(response)})
        else:
            logging.debug('successfully %(what)s motion detection for camera with id %(id)s' % {'what': ['disabled', 'enabled'][enabled], 'id': camera_id})
    request = HTTPRequest(url, connect_timeout=_MOTION_CONTROL_TIMEOUT, request_timeout=_MOTION_CONTROL_TIMEOUT)
    http_client = AsyncHTTPClient()
    http_client.fetch(request, on_response)

def take_snapshot(camera_id):
    if False:
        for i in range(10):
            print('nop')
    from tornado.httpclient import HTTPRequest, AsyncHTTPClient
    motion_camera_id = camera_id_to_motion_camera_id(camera_id)
    if motion_camera_id is None:
        return logging.error('could not find motion camera id for camera with id %s' % camera_id)
    logging.debug('taking snapshot for camera with id %(id)s' % {'id': camera_id})
    url = 'http://127.0.0.1:%(port)s/%(id)s/action/snapshot' % {'port': settings.MOTION_CONTROL_PORT, 'id': motion_camera_id}

    def on_response(response):
        if False:
            print('Hello World!')
        if response.error:
            logging.error('failed to take snapshot for camera with id %(id)s: %(msg)s' % {'id': camera_id, 'msg': utils.pretty_http_error(response)})
        else:
            logging.debug('successfully took snapshot for camera with id %(id)s' % {'id': camera_id})
    request = HTTPRequest(url, connect_timeout=_MOTION_CONTROL_TIMEOUT, request_timeout=_MOTION_CONTROL_TIMEOUT)
    http_client = AsyncHTTPClient()
    http_client.fetch(request, on_response)

def is_motion_detected(camera_id):
    if False:
        print('Hello World!')
    return _motion_detected.get(camera_id, False)

def set_motion_detected(camera_id, motion_detected):
    if False:
        for i in range(10):
            print('nop')
    if motion_detected:
        logging.debug('marking motion detected for camera with id %s' % camera_id)
    else:
        logging.debug('clearing motion detected for camera with id %s' % camera_id)
    _motion_detected[camera_id] = motion_detected

def camera_id_to_motion_camera_id(camera_id):
    if False:
        for i in range(10):
            print('nop')
    import config
    main_config = config.get_main()
    cameras = main_config.get('camera', [])
    camera_filename = 'camera-%d.conf' % camera_id
    for (i, camera) in enumerate(cameras):
        if camera != camera_filename:
            continue
        return i + 1
    return None

def motion_camera_id_to_camera_id(motion_camera_id):
    if False:
        for i in range(10):
            print('nop')
    import config
    main_config = config.get_main()
    cameras = main_config.get('camera', [])
    try:
        return int(re.search('camera-(\\d+).conf', cameras[int(motion_camera_id) - 1]).group(1))
    except IndexError:
        return None

def is_motion_pre42():
    if False:
        i = 10
        return i + 15
    (binary, version) = find_motion()
    if not binary:
        return False
    return update.compare_versions(version, '4.2') < 0

def has_h264_omx_support():
    if False:
        i = 10
        return i + 15
    (binary, version, codecs) = mediafiles.find_ffmpeg()
    if not binary:
        return False
    return 'h264_omx' in codecs.get('h264', {}).get('encoders', set())

def has_h264_v4l2m2m_support():
    if False:
        for i in range(10):
            print('nop')
    (binary, version, codecs) = mediafiles.find_ffmpeg()
    if not binary:
        return False
    return 'h264_v4l2m2m' in codecs.get('h264', {}).get('encoders', set())

def has_h264_nvenc_support():
    if False:
        print('Hello World!')
    (binary, version, codecs) = mediafiles.find_ffmpeg()
    if not binary:
        return False
    return 'h264_nvenc' in codecs.get('h264', {}).get('encoders', set())

def has_h264_nvmpi_support():
    if False:
        while True:
            i = 10
    (binary, version, codecs) = mediafiles.find_ffmpeg()
    if not binary:
        return False
    return 'h264_nvmpi' in codecs.get('h264', {}).get('encoders', set())

def has_hevc_nvmpi_support():
    if False:
        for i in range(10):
            print('nop')
    (binary, version, codecs) = mediafiles.find_ffmpeg()
    if not binary:
        return False
    return 'hevc_nvmpi' in codecs.get('hevc', {}).get('encoders', set())

def has_hevc_nvenc_support():
    if False:
        return 10
    (binary, version, codecs) = mediafiles.find_ffmpeg()
    if not binary:
        return False
    return 'hevc_nvenc' in codecs.get('hevc', {}).get('encoders', set())

def has_h264_qsv_support():
    if False:
        for i in range(10):
            print('nop')
    (binary, version, codecs) = mediafiles.find_ffmpeg()
    if not binary:
        return False
    return 'h264_qsv' in codecs.get('h264', {}).get('encoders', set())

def has_hevc_qsv_support():
    if False:
        print('Hello World!')
    (binary, version, codecs) = mediafiles.find_ffmpeg()
    if not binary:
        return False
    return 'hevc_qsv' in codecs.get('hevc', {}).get('encoders', set())

def resolution_is_valid(width, height):
    if False:
        return 10
    if width % 8:
        return False
    if height % 8:
        return False
    return True

def _disable_initial_motion_detection():
    if False:
        print('Hello World!')
    import config
    for camera_id in config.get_camera_ids():
        camera_config = config.get_camera(camera_id)
        if not utils.is_local_motion_camera(camera_config):
            continue
        if not camera_config['@motion_detection']:
            logging.debug('motion detection disabled by config for camera with id %s' % camera_id)
            set_motion_detection(camera_id, False)

def _get_pid():
    if False:
        for i in range(10):
            print('nop')
    motion_pid_path = os.path.join(settings.RUN_PATH, 'motion.pid')
    try:
        with open(motion_pid_path, 'r') as f:
            return int(f.readline().strip())
    except (IOError, ValueError):
        return None