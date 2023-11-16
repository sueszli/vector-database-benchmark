"""
Wrapper module for at(1)

Also, a 'tag' feature has been added to more
easily tag jobs.

:platform:      linux,openbsd,freebsd

.. versionchanged:: 2017.7.0
"""
import datetime
import re
import time
import salt.utils.data
import salt.utils.path
import salt.utils.platform
from salt.exceptions import CommandNotFoundError
BSD = ('OpenBSD', 'FreeBSD')
__virtualname__ = 'at'

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Most everything has the ability to support at(1)\n    '
    if salt.utils.platform.is_windows() or salt.utils.platform.is_sunos():
        return (False, 'The at module could not be loaded: unsupported platform')
    if salt.utils.path.which('at') is None:
        return (False, 'The at module could not be loaded: at command not found')
    return __virtualname__

def _cmd(binary, *args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Wrapper to run at(1) or return None.\n    '
    binary = salt.utils.path.which(binary)
    if not binary:
        raise CommandNotFoundError(f'{binary}: command not found')
    cmd = [binary] + list(args)
    return __salt__['cmd.run_stdout']([binary] + list(args), python_shell=False)

def atq(tag=None):
    if False:
        i = 10
        return i + 15
    "\n    List all queued and running jobs or only those with\n    an optional 'tag'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' at.atq\n        salt '*' at.atq [tag]\n        salt '*' at.atq [job number]\n    "
    jobs = []
    if __grains__['os_family'] == 'RedHat':
        output = _cmd('at', '-l')
    else:
        output = _cmd('atq')
    if output is None:
        return "'at.atq' is not available."
    if output == '':
        return {'jobs': jobs}
    job_kw_regex = re.compile('^### SALT: (\\w+)')
    for line in output.splitlines():
        job_tag = ''
        if __grains__['os_family'] == 'RedHat':
            (job, spec) = line.split('\t')
            specs = spec.split()
        elif __grains__['os'] == 'OpenBSD':
            if line.startswith(' Rank'):
                continue
            else:
                tmp = line.split()
                timestr = ' '.join(tmp[1:5])
                job = tmp[6]
                specs = datetime.datetime(*time.strptime(timestr, '%b %d, %Y %H:%M')[0:5]).isoformat().split('T')
                specs.append(tmp[7])
                specs.append(tmp[5])
        elif __grains__['os'] == 'FreeBSD':
            if line.startswith('Date'):
                continue
            else:
                tmp = line.split()
                timestr = ' '.join(tmp[1:6])
                job = tmp[8]
                specs = datetime.datetime(*time.strptime(timestr, '%b %d %H:%M:%S %Z %Y')[0:5]).isoformat().split('T')
                specs.append(tmp[7])
                specs.append(tmp[6])
        else:
            (job, spec) = line.split('\t')
            tmp = spec.split()
            timestr = ' '.join(tmp[0:5])
            specs = datetime.datetime(*time.strptime(timestr)[0:5]).isoformat().split('T')
            specs.append(tmp[5])
            specs.append(tmp[6])
        atc_out = _cmd('at', '-c', job)
        for line in atc_out.splitlines():
            tmp = job_kw_regex.match(line)
            if tmp:
                job_tag = tmp.groups()[0]
        if __grains__['os'] in BSD:
            job = str(job)
        else:
            job = int(job)
        if tag:
            if tag == job_tag or tag == job:
                jobs.append({'job': job, 'date': specs[0], 'time': specs[1], 'queue': specs[2], 'user': specs[3], 'tag': job_tag})
        else:
            jobs.append({'job': job, 'date': specs[0], 'time': specs[1], 'queue': specs[2], 'user': specs[3], 'tag': job_tag})
    return {'jobs': jobs}

def atrm(*args):
    if False:
        print('Hello World!')
    "\n    Remove jobs from the queue.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' at.atrm <jobid> <jobid> .. <jobid>\n        salt '*' at.atrm all\n        salt '*' at.atrm all [tag]\n    "
    if not salt.utils.path.which('at'):
        return "'at.atrm' is not available."
    if not args:
        return {'jobs': {'removed': [], 'tag': None}}
    args = salt.utils.data.stringify(args)
    if args[0] == 'all':
        if len(args) > 1:
            opts = list(list(map(str, [j['job'] for j in atq(args[1])['jobs']])))
            ret = {'jobs': {'removed': opts, 'tag': args[1]}}
        else:
            opts = list(list(map(str, [j['job'] for j in atq()['jobs']])))
            ret = {'jobs': {'removed': opts, 'tag': None}}
    else:
        opts = list(list(map(str, [i['job'] for i in atq()['jobs'] if str(i['job']) in args])))
        ret = {'jobs': {'removed': opts, 'tag': None}}
    output = _cmd('at', '-d', ' '.join(opts))
    if output is None:
        return "'at.atrm' is not available."
    return ret

def at(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add a job to the queue.\n\n    The \'timespec\' follows the format documented in the\n    at(1) manpage.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' at.at <timespec> <cmd> [tag=<tag>] [runas=<user>]\n        salt \'*\' at.at 12:05am \'/sbin/reboot\' tag=reboot\n        salt \'*\' at.at \'3:05am +3 days\' \'bin/myscript\' tag=nightly runas=jim\n        salt \'*\' at.at \'"22:02"\' \'bin/myscript\' tag=nightly runas=jim\n    '
    if len(args) < 2:
        return {'jobs': []}
    binary = salt.utils.path.which('at')
    if not binary:
        return "'at.at' is not available."
    if 'tag' in kwargs:
        stdin = '### SALT: {}\n{}'.format(kwargs['tag'], ' '.join(args[1:]))
    else:
        stdin = ' '.join(args[1:])
    cmd = [binary, args[0]]
    cmd_kwargs = {'stdin': stdin, 'python_shell': False}
    if 'runas' in kwargs:
        cmd_kwargs['runas'] = kwargs['runas']
    output = __salt__['cmd.run'](cmd, **cmd_kwargs)
    if output is None:
        return "'at.at' is not available."
    if output.endswith('Garbled time'):
        return {'jobs': [], 'error': 'invalid timespec'}
    if output.startswith('warning: commands'):
        output = output.splitlines()[1]
    if output.startswith('commands will be executed'):
        output = output.splitlines()[1]
    output = output.split()[1]
    if __grains__['os'] in BSD:
        return atq(str(output))
    else:
        return atq(int(output))

def atc(jobid):
    if False:
        print('Hello World!')
    "\n    Print the at(1) script that will run for the passed job\n    id. This is mostly for debugging so the output will\n    just be text.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' at.atc <jobid>\n    "
    output = _cmd('at', '-c', str(jobid))
    if output is None:
        return "'at.atc' is not available."
    elif output == '':
        return {'error': f"invalid job id '{jobid}'"}
    return output

def _atq(**kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Return match jobs list\n    '
    jobs = []
    runas = kwargs.get('runas', None)
    tag = kwargs.get('tag', None)
    hour = kwargs.get('hour', None)
    minute = kwargs.get('minute', None)
    day = kwargs.get('day', None)
    month = kwargs.get('month', None)
    year = kwargs.get('year', None)
    if year and len(str(year)) == 2:
        year = f'20{year}'
    jobinfo = atq()['jobs']
    if not jobinfo:
        return {'jobs': jobs}
    for job in jobinfo:
        if not runas:
            pass
        elif runas == job['user']:
            pass
        else:
            continue
        if not tag:
            pass
        elif tag == job['tag']:
            pass
        else:
            continue
        if not hour:
            pass
        elif f'{int(hour):02d}' == job['time'].split(':')[0]:
            pass
        else:
            continue
        if not minute:
            pass
        elif f'{int(minute):02d}' == job['time'].split(':')[1]:
            pass
        else:
            continue
        if not day:
            pass
        elif f'{int(day):02d}' == job['date'].split('-')[2]:
            pass
        else:
            continue
        if not month:
            pass
        elif f'{int(month):02d}' == job['date'].split('-')[1]:
            pass
        else:
            continue
        if not year:
            pass
        elif year == job['date'].split('-')[0]:
            pass
        else:
            continue
        jobs.append(job)
    if not jobs:
        note = 'No match jobs or time format error'
        return {'jobs': jobs, 'note': note}
    return {'jobs': jobs}

def jobcheck(**kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Check the job from queue.\n    The kwargs dict include 'hour minute day month year tag runas'\n    Other parameters will be ignored.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' at.jobcheck runas=jam day=13\n        salt '*' at.jobcheck day=13 month=12 year=13 tag=rose\n    "
    if not kwargs:
        return {'error': 'You have given a condition'}
    return _atq(**kwargs)