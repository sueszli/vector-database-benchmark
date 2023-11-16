import fcntl
import logging
import os.path
import pipes
import re
import stat
import subprocess
import time
import utils
_resolutions_cache = {}
_ctrls_cache = {}
_ctrl_values_cache = {}
_DEV_V4L_BY_ID = '/dev/v4l/by-id/'
_V4L2_TIMEOUT = 10

def find_v4l2_ctl():
    if False:
        for i in range(10):
            print('nop')
    try:
        return subprocess.check_output(['which', 'v4l2-ctl'], stderr=utils.DEV_NULL).strip()
    except subprocess.CalledProcessError:
        return None

def list_devices():
    if False:
        return 10
    global _resolutions_cache, _ctrls_cache, _ctrl_values_cache
    logging.debug('listing V4L2 devices')
    try:
        output = ''
        started = time.time()
        p = subprocess.Popen(['v4l2-ctl', '--list-devices'], stdout=subprocess.PIPE, bufsize=1)
        fd = p.stdout.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
        while True:
            try:
                data = p.stdout.read(1024)
                if not data:
                    break
            except IOError:
                data = ''
                time.sleep(0.01)
            output += data
            if len(output) > 10240:
                logging.warn('v4l2-ctl command returned more than 10k of output')
                break
            if time.time() - started > _V4L2_TIMEOUT:
                logging.warn('v4l2-ctl command ran for more than %s seconds' % _V4L2_TIMEOUT)
                break
    except subprocess.CalledProcessError:
        logging.debug('failed to list devices (probably no devices installed)')
        return []
    try:
        p.kill()
    except:
        pass
    name = None
    devices = []
    for line in output.split('\n'):
        if line.startswith('\t'):
            device = line.strip()
            persistent_device = find_persistent_device(device)
            devices.append((device, persistent_device, name))
            logging.debug('found device %(name)s: %(device)s, %(persistent_device)s' % {'name': name, 'device': device, 'persistent_device': persistent_device})
        else:
            name = line.split('(')[0].strip()
    _resolutions_cache = {}
    _ctrls_cache = {}
    _ctrl_values_cache = {}
    return devices

def list_resolutions(device):
    if False:
        i = 10
        return i + 15
    import motionctl
    global _resolutions_cache
    device = utils.make_str(device)
    if device in _resolutions_cache:
        return _resolutions_cache[device]
    logging.debug('listing resolutions of device %(device)s...' % {'device': device})
    resolutions = set()
    output = ''
    started = time.time()
    cmd = 'v4l2-ctl -d %(device)s --list-formats-ext | grep -vi stepwise | grep -oE "[0-9]+x[0-9]+" || true' % {'device': pipes.quote(device)}
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, bufsize=1)
    fd = p.stdout.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    while True:
        try:
            data = p.stdout.read(1024)
            if not data:
                break
        except IOError:
            data = ''
            time.sleep(0.01)
        output += data
        if len(output) > 10240:
            logging.warn('v4l2-ctl command returned more than 10k of output')
            break
        if time.time() - started > _V4L2_TIMEOUT:
            logging.warn('v4l2-ctl command ran for more than %s seconds' % _V4L2_TIMEOUT)
            break
    try:
        p.kill()
    except:
        pass
    for pair in output.split('\n'):
        pair = pair.strip()
        if not pair:
            continue
        (width, height) = pair.split('x')
        width = int(width)
        height = int(height)
        if (width, height) in resolutions:
            continue
        if width < 96 or height < 96:
            continue
        if not motionctl.resolution_is_valid(width, height):
            continue
        resolutions.add((width, height))
        logging.debug('found resolution %(width)sx%(height)s for device %(device)s' % {'device': device, 'width': width, 'height': height})
    if not resolutions:
        logging.debug('no resolutions found for device %(device)s, using common values' % {'device': device})
        resolutions = utils.COMMON_RESOLUTIONS
        resolutions = [r for r in resolutions if motionctl.resolution_is_valid(*r)]
    resolutions = list(sorted(resolutions, key=lambda r: (r[0], r[1])))
    _resolutions_cache[device] = resolutions
    return resolutions

def device_present(device):
    if False:
        for i in range(10):
            print('nop')
    device = utils.make_str(device)
    try:
        st = os.stat(device)
        return stat.S_ISCHR(st.st_mode)
    except:
        return False

def find_persistent_device(device):
    if False:
        while True:
            i = 10
    device = utils.make_str(device)
    try:
        devs_by_id = os.listdir(_DEV_V4L_BY_ID)
    except OSError:
        return device
    for p in devs_by_id:
        p = os.path.join(_DEV_V4L_BY_ID, p)
        if os.path.realpath(p) == device:
            return p
    return device

def list_ctrls(device):
    if False:
        for i in range(10):
            print('nop')
    global _ctrls_cache
    device = utils.make_str(device)
    if device in _ctrls_cache:
        return _ctrls_cache[device]
    output = ''
    started = time.time()
    p = subprocess.Popen('v4l2-ctl -d %(device)s --list-ctrls' % {'device': pipes.quote(device)}, shell=True, stdout=subprocess.PIPE, bufsize=1)
    fd = p.stdout.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    while True:
        try:
            data = p.stdout.read(1024)
            if not data:
                break
        except IOError:
            data = ''
            time.sleep(0.01)
        output += data
        if len(output) > 10240:
            logging.warn('v4l2-ctl command returned more than 10k of output')
            break
        if time.time() - started > 3:
            logging.warn('v4l2-ctl command ran for more than 3 seconds')
            break
    try:
        p.kill()
    except:
        pass
    controls = {}
    for line in output.split('\n'):
        if not line:
            continue
        match = re.match('^\\s*(\\w+)\\s+([a-f0-9x\\s]+)?\\(\\w+\\)\\s*:\\s*(.+)\\s*', line)
        if not match:
            continue
        (control, _, properties) = match.groups()
        properties = dict([v.split('=', 1) for v in properties.split(' ') if v.count('=')])
        controls[control] = properties
    _ctrls_cache[device] = controls
    return controls