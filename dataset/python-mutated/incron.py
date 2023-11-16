"""
Work with incron
"""
import logging
import os
import salt.utils.data
import salt.utils.files
import salt.utils.functools
import salt.utils.stringutils
log = logging.getLogger(__name__)
TAG = '# Line managed by Salt, do not edit'
_INCRON_SYSTEM_TAB = '/etc/incron.d/'
_MASK_TYPES = ['IN_ACCESS', 'IN_ATTRIB', 'IN_CLOSE_WRITE', 'IN_CLOSE_NOWRITE', 'IN_CREATE', 'IN_DELETE', 'IN_DELETE_SELF', 'IN_MODIFY', 'IN_MOVE_SELF', 'IN_MOVED_FROM', 'IN_MOVED_TO', 'IN_OPEN', 'IN_ALL_EVENTS', 'IN_MOVE', 'IN_CLOSE', 'IN_DONT_FOLLOW', 'IN_ONESHOT', 'IN_ONLYDIR', 'IN_NO_LOOP']

def _needs_change(old, new):
    if False:
        while True:
            i = 10
    if old != new:
        if new == 'random':
            if old == '*':
                return True
        elif new is not None:
            return True
    return False

def _render_tab(lst):
    if False:
        while True:
            i = 10
    '\n    Takes a tab list structure and renders it to a list for applying it to\n    a file\n    '
    ret = []
    for pre in lst['pre']:
        ret.append('{}\n'.format(pre))
    for cron in lst['crons']:
        ret.append('{} {} {}\n'.format(cron['path'], cron['mask'], cron['cmd']))
    return ret

def _get_incron_cmdstr(path):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a format string, to be used to build an incrontab command.\n    '
    return 'incrontab {}'.format(path)

def write_incron_file(user, path):
    if False:
        while True:
            i = 10
    "\n    Writes the contents of a file to a user's incrontab\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' incron.write_incron_file root /tmp/new_incron\n    "
    return __salt__['cmd.retcode'](_get_incron_cmdstr(path), runas=user, python_shell=False) == 0

def write_incron_file_verbose(user, path):
    if False:
        return 10
    "\n    Writes the contents of a file to a user's incrontab and return error message on error\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' incron.write_incron_file_verbose root /tmp/new_incron\n    "
    return __salt__['cmd.run_all'](_get_incron_cmdstr(path), runas=user, python_shell=False)

def _write_incron_lines(user, lines):
    if False:
        for i in range(10):
            print('nop')
    "\n    Takes a list of lines to be committed to a user's incrontab and writes it\n    "
    if user == 'system':
        ret = {}
        ret['retcode'] = _write_file(_INCRON_SYSTEM_TAB, 'salt', ''.join(lines))
        return ret
    else:
        path = salt.utils.files.mkstemp()
        with salt.utils.files.fopen(path, 'wb') as fp_:
            fp_.writelines(salt.utils.data.encode(lines))
        if user != 'root':
            __salt__['cmd.run']('chown {} {}'.format(user, path), python_shell=False)
        ret = __salt__['cmd.run_all'](_get_incron_cmdstr(path), runas=user, python_shell=False)
        os.remove(path)
        return ret

def _write_file(folder, filename, data):
    if False:
        print('Hello World!')
    '\n    Writes a file to disk\n    '
    path = os.path.join(folder, filename)
    if not os.path.exists(folder):
        msg = '{} cannot be written. {} does not exist'.format(filename, folder)
        log.error(msg)
        raise AttributeError(str(msg))
    with salt.utils.files.fopen(path, 'w') as fp_:
        fp_.write(salt.utils.stringutils.to_str(data))
    return 0

def _read_file(folder, filename):
    if False:
        while True:
            i = 10
    '\n    Reads and returns the contents of a file\n    '
    path = os.path.join(folder, filename)
    try:
        with salt.utils.files.fopen(path, 'rb') as contents:
            return salt.utils.data.decode(contents.readlines())
    except OSError:
        return ''

def raw_system_incron():
    if False:
        i = 10
        return i + 15
    "\n    Return the contents of the system wide incrontab\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' incron.raw_system_incron\n    "
    log.debug('read_file %s', _read_file(_INCRON_SYSTEM_TAB, 'salt'))
    return ''.join(_read_file(_INCRON_SYSTEM_TAB, 'salt'))

def raw_incron(user):
    if False:
        while True:
            i = 10
    "\n    Return the contents of the user's incrontab\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' incron.raw_incron root\n    "
    cmd = 'incrontab -l {}'.format(user)
    return __salt__['cmd.run_stdout'](cmd, rstrip=False, runas=user, python_shell=False)

def list_tab(user):
    if False:
        return 10
    "\n    Return the contents of the specified user's incrontab\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' incron.list_tab root\n    "
    if user == 'system':
        data = raw_system_incron()
    else:
        data = raw_incron(user)
        log.debug('user data %s', data)
    ret = {'crons': [], 'pre': []}
    flag = False
    for line in data.splitlines():
        if len(line.split()) > 3:
            comps = line.split()
            path = comps[0]
            mask = comps[1]
            cmd = ' '.join(comps[2:])
            dat = {'path': path, 'mask': mask, 'cmd': cmd}
            ret['crons'].append(dat)
        else:
            ret['pre'].append(line)
    return ret
ls = salt.utils.functools.alias_function(list_tab, 'ls')

def set_job(user, path, mask, cmd):
    if False:
        print('Hello World!')
    '\n    Sets an incron job up for a specified user.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' incron.set_job root \'/root\' \'IN_MODIFY\' \'echo "$$ $@ $# $% $&"\'\n    '
    mask = str(mask).upper()
    for item in mask.split(','):
        if item not in _MASK_TYPES:
            return 'Invalid mask type: {}'.format(item)
    updated = False
    arg_mask = mask.split(',')
    arg_mask.sort()
    lst = list_tab(user)
    updated_crons = []
    for (item, cron) in enumerate(lst['crons']):
        if path == cron['path']:
            if cron['cmd'] == cmd:
                cron_mask = cron['mask'].split(',')
                cron_mask.sort()
                if cron_mask == arg_mask:
                    return 'present'
                if any([x in cron_mask for x in arg_mask]):
                    updated = True
                else:
                    updated_crons.append(cron)
            else:
                updated_crons.append(cron)
        else:
            updated_crons.append(cron)
    cron = {'cmd': cmd, 'path': path, 'mask': mask}
    updated_crons.append(cron)
    lst['crons'] = updated_crons
    comdat = _write_incron_lines(user, _render_tab(lst))
    if comdat['retcode']:
        return comdat['stderr']
    if updated:
        return 'updated'
    else:
        return 'new'

def rm_job(user, path, mask, cmd):
    if False:
        return 10
    "\n    Remove a incron job for a specified user. If any of the day/time params are\n    specified, the job will only be removed if the specified params match.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' incron.rm_job root /path\n    "
    mask = str(mask).upper()
    for item in mask.split(','):
        if item not in _MASK_TYPES:
            return 'Invalid mask type: {}'.format(item)
    lst = list_tab(user)
    ret = 'absent'
    rm_ = None
    for (ind, val) in enumerate(lst['crons']):
        if rm_ is not None:
            break
        if path == val['path']:
            if cmd == val['cmd']:
                if mask == val['mask']:
                    rm_ = ind
    if rm_ is not None:
        lst['crons'].pop(rm_)
        ret = 'removed'
    comdat = _write_incron_lines(user, _render_tab(lst))
    if comdat['retcode']:
        return comdat['stderr']
    return ret
rm = salt.utils.functools.alias_function(rm_job, 'rm')