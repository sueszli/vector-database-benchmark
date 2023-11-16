import hashlib
import logging
import os
import settings
import subprocess
from config import additional_config
LOCAL_TIME_FILE = settings.LOCAL_TIME_FILE

def get_time_zone():
    if False:
        i = 10
        return i + 15
    return _get_time_zone_symlink() or _get_time_zone_md5() or 'UTC'

def _get_time_zone_symlink():
    if False:
        while True:
            i = 10
    f = settings.LOCAL_TIME_FILE
    if not f:
        return None
    for i in xrange(8):
        try:
            f = os.readlink(f)
        except OSError:
            break
    if f and f.startswith('/usr/share/zoneinfo/'):
        f = f[20:]
    else:
        f = None
    time_zone = f or None
    if time_zone:
        logging.debug('found time zone by symlink method: %s' % time_zone)
    return time_zone

def _get_time_zone_md5():
    if False:
        print('Hello World!')
    if settings.LOCAL_TIME_FILE:
        return None
    try:
        output = subprocess.check_output('find * -type f | xargs md5sum', shell=True, cwd='/usr/share/zoneinfo')
    except Exception as e:
        logging.error('getting md5 of zoneinfo files failed: %s' % e)
        return None
    lines = [l for l in output.split('\n') if l]
    lines = [l.split(None, 1) for l in lines]
    time_zone_by_md5 = dict(lines)
    try:
        with open(settings.LOCAL_TIME_FILE, 'r') as f:
            data = f.read()
    except Exception as e:
        logging.error('failed to read local time file: %s' % e)
        return None
    md5 = hashlib.md5(data).hexdigest()
    time_zone = time_zone_by_md5.get(md5)
    if time_zone:
        logging.debug('found time zone by md5 method: %s' % time_zone)
    return time_zone

def _set_time_zone(time_zone):
    if False:
        i = 10
        return i + 15
    time_zone = time_zone or 'UTC'
    zoneinfo_file = '/usr/share/zoneinfo/' + time_zone
    if not os.path.exists(zoneinfo_file):
        logging.error('%s file does not exist' % zoneinfo_file)
        return False
    logging.debug('linking "%s" to "%s"' % (settings.LOCAL_TIME_FILE, zoneinfo_file))
    try:
        os.remove(settings.LOCAL_TIME_FILE)
    except:
        pass
    try:
        os.symlink(zoneinfo_file, settings.LOCAL_TIME_FILE)
        return True
    except Exception as e:
        logging.error('failed to link "%s" to "%s": %s' % (settings.LOCAL_TIME_FILE, zoneinfo_file, e))
        return False

@additional_config
def timeZone():
    if False:
        for i in range(10):
            print('nop')
    if not LOCAL_TIME_FILE:
        return
    import pytz
    timezones = pytz.common_timezones
    return {'label': 'Time Zone', 'description': 'selecting the right timezone assures a correct timestamp displayed on pictures and movies', 'type': 'choices', 'choices': [(t, t) for t in timezones], 'section': 'general', 'reboot': True, 'get': get_time_zone, 'set': _set_time_zone}