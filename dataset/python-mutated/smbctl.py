import datetime
import logging
import os
import re
import subprocess
import time
import utils
from tornado.ioloop import IOLoop
import config
import settings

def start():
    if False:
        for i in range(10):
            print('nop')
    io_loop = IOLoop.instance()
    io_loop.add_timeout(datetime.timedelta(seconds=settings.MOUNT_CHECK_INTERVAL), _check_mounts)

def stop():
    if False:
        return 10
    _umount_all()

def find_mount_cifs():
    if False:
        print('Hello World!')
    try:
        return subprocess.check_output(['which', 'mount.cifs'], stderr=utils.DEV_NULL).strip()
    except subprocess.CalledProcessError:
        return None

def make_mount_point(server, share, username):
    if False:
        return 10
    server = re.sub('[^a-zA-Z0-9]', '_', server).lower()
    share = re.sub('[^a-zA-Z0-9]', '_', share).lower()
    if username:
        username = re.sub('[^a-zA-Z0-9]', '_', username).lower()
        mount_point = os.path.join(settings.SMB_MOUNT_ROOT, 'motioneye_%s_%s_%s' % (server, share, username))
    else:
        mount_point = os.path.join(settings.SMB_MOUNT_ROOT, 'motioneye_%s_%s' % (server, share))
    return mount_point

def list_mounts():
    if False:
        while True:
            i = 10
    logging.debug('listing smb mounts...')
    mounts = []
    with open('/proc/mounts', 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            target = parts[0]
            mount_point = parts[1]
            fstype = parts[2]
            opts = ' '.join(parts[3:])
            if fstype != 'cifs':
                continue
            if not _is_motioneye_mount(mount_point):
                continue
            match = re.match('//([^/]+)/(.+)', target)
            if not match:
                continue
            if len(match.groups()) != 2:
                continue
            (server, share) = match.groups()
            share = share.replace('\\040', ' ')
            match = re.search('username=([a-z][-\\w]*)', opts)
            if match:
                username = match.group(1)
            else:
                username = ''
            match = re.search('vers=([\\w.]+)', opts)
            if match:
                smb_ver = match.group(1)
            else:
                smb_ver = '1.0'
            logging.debug('found smb mount "//%s/%s" at "%s"' % (server, share, mount_point))
            mounts.append({'server': server.lower(), 'share': share.lower(), 'smb_ver': smb_ver, 'username': username, 'mount_point': mount_point})
    return mounts

def update_mounts():
    if False:
        print('Hello World!')
    network_shares = config.get_network_shares()
    mounts = list_mounts()
    mounts = dict((((m['server'], m['share'], m['smb_ver'], m['username'] or ''), False) for m in mounts))
    should_stop = False
    should_start = True
    for network_share in network_shares:
        key = (network_share['server'].lower(), network_share['share'].lower(), network_share['smb_ver'], network_share['username'].lower() or '')
        if key in mounts:
            mounts[key] = True
        else:
            should_stop = True
            if not _mount(network_share['server'], network_share['share'], network_share['smb_ver'], network_share['username'], network_share['password']):
                should_start = False
    for ((server, share, smb_ver, username), required) in mounts.items():
        if not required:
            _umount(server, share, username)
            should_stop = True
    return (should_stop, should_start)

def test_share(server, share, smb_ver, username, password, root_directory):
    if False:
        print('Hello World!')
    mounts = list_mounts()
    mounts = dict((((m['server'], m['share'], m['smb_ver'], m['username'] or ''), m['mount_point']) for m in mounts))
    key = (server, share, smb_ver, username or '')
    mounted = False
    mount_point = mounts.get(key)
    if not mount_point:
        mount_point = _mount(server, share, smb_ver, username, password)
        if not mount_point:
            raise Exception('cannot mount network share')
        mounted = True

    def maybe_umount():
        if False:
            i = 10
            return i + 15
        if mounted:
            time.sleep(1)
            _umount(server, share, username)
    path = os.path.join(mount_point, root_directory)
    if os.path.exists(path):
        return maybe_umount()
    try:
        os.makedirs(path)
    except:
        raise Exception('cannot create root directory')
    finally:
        maybe_umount()

def _mount(server, share, smb_ver, username, password):
    if False:
        for i in range(10):
            print('nop')
    mount_point = make_mount_point(server, share, username)
    logging.debug('making sure mount point "%s" exists' % mount_point)
    if not os.path.exists(mount_point):
        os.makedirs(mount_point)
    if username:
        opts = 'username=%s,password=%s' % (username, password)
        sec_types = [None, 'ntlm', 'ntlmv2', 'ntlmv2i', 'ntlmsspi', 'none']
    else:
        opts = 'guest'
        sec_types = [None, 'none', 'ntlm', 'ntlmv2', 'ntlmv2i', 'ntlmsspi']
    opts += ',vers=%s' % smb_ver
    for sec in sec_types:
        if sec:
            actual_opts = opts + ',sec=' + sec
        else:
            actual_opts = opts
        try:
            logging.debug('mounting "//%s/%s" at "%s" (sec=%s)' % (server, share, mount_point, sec))
            subprocess.check_call(['mount.cifs', '//%s/%s' % (server, share), mount_point, '-o', actual_opts])
            break
        except subprocess.CalledProcessError:
            pass
    else:
        logging.error('failed to mount smb share "//%s/%s" at "%s"' % (server, share, mount_point))
        return None
    try:
        path = os.path.join(mount_point, '.motioneye_' + str(int(time.time())))
        os.mkdir(path)
        os.rmdir(path)
        logging.debug('directory at "%s" is writable' % mount_point)
    except:
        logging.error('directory at "%s" is not writable' % mount_point)
        return None
    return mount_point

def _umount(server, share, username):
    if False:
        for i in range(10):
            print('nop')
    mount_point = make_mount_point(server, share, username)
    logging.debug('unmounting "//%s/%s" from "%s"' % (server, share, mount_point))
    try:
        subprocess.check_call(['umount', mount_point])
    except subprocess.CalledProcessError:
        logging.error('failed to unmount smb share "//%s/%s" from "%s"' % (server, share, mount_point))
        return False
    try:
        os.rmdir(mount_point)
    except Exception as e:
        logging.error('failed to remove smb mount point "%s": %s' % (mount_point, e))
        return False
    return True

def _is_motioneye_mount(mount_point):
    if False:
        return 10
    mount_point_root = os.path.join(settings.SMB_MOUNT_ROOT, 'motioneye_')
    return bool(re.match('^' + mount_point_root + '\\w+$', mount_point))

def _umount_all():
    if False:
        i = 10
        return i + 15
    for mount in list_mounts():
        _umount(mount['server'], mount['share'], mount['username'])

def _check_mounts():
    if False:
        i = 10
        return i + 15
    import motionctl
    logging.debug('checking SMB mounts...')
    (stop, start) = update_mounts()
    if stop:
        motionctl.stop()
    if start:
        motionctl.start()
    io_loop = IOLoop.instance()
    io_loop.add_timeout(datetime.timedelta(seconds=settings.MOUNT_CHECK_INTERVAL), _check_mounts)