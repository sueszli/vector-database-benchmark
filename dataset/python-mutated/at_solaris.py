"""
Wrapper for at(1) on Solaris-like systems

.. note::
    we try to mirror the generic at module
    where possible

:maintainer:    jorge schrauwen <sjorge@blackdot.be>
:maturity:      new
:platform:      solaris,illumos,smartso

.. versionadded:: 2017.7.0
"""
import datetime
import logging
import re
import time
import salt.utils.files
import salt.utils.path
import salt.utils.platform
import salt.utils.stringutils
log = logging.getLogger(__name__)
__virtualname__ = 'at'

def __virtual__():
    if False:
        i = 10
        return i + 15
    "\n    We only deal with Solaris' specific version of at\n    "
    if not salt.utils.platform.is_sunos():
        return (False, 'The at module could not be loaded: unsupported platform')
    if not salt.utils.path.which('at') or not salt.utils.path.which('atq') or (not salt.utils.path.which('atrm')):
        return (False, 'The at module could not be loaded: at command not found')
    return __virtualname__

def atq(tag=None):
    if False:
        while True:
            i = 10
    "\n    List all queued and running jobs or only those with\n    an optional 'tag'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' at.atq\n        salt '*' at.atq [tag]\n        salt '*' at.atq [job number]\n    "
    jobs = []
    res = __salt__['cmd.run_all']('atq')
    if res['retcode'] > 0:
        return {'error': res['stderr']}
    if res['stdout'] == 'no files in queue.':
        return {'jobs': jobs}
    job_kw_regex = re.compile('^### SALT: (\\w+)')
    for line in res['stdout'].splitlines():
        job_tag = ''
        if line.startswith(' Rank'):
            continue
        tmp = line.split()
        timestr = ' '.join(tmp[1:5])
        job = tmp[6]
        specs = datetime.datetime(*time.strptime(timestr, '%b %d, %Y %H:%M')[0:5]).isoformat().split('T')
        specs.append(tmp[7])
        specs.append(tmp[5])
        job = str(job)
        atjob_file = f'/var/spool/cron/atjobs/{job}'
        if __salt__['file.file_exists'](atjob_file):
            with salt.utils.files.fopen(atjob_file, 'r') as atjob:
                for line in atjob:
                    line = salt.utils.stringutils.to_unicode(line)
                    tmp = job_kw_regex.match(line)
                    if tmp:
                        job_tag = tmp.groups()[0]
        if not tag:
            jobs.append({'job': job, 'date': specs[0], 'time': specs[1], 'queue': specs[2], 'user': specs[3], 'tag': job_tag})
        elif tag and tag in [job_tag, job]:
            jobs.append({'job': job, 'date': specs[0], 'time': specs[1], 'queue': specs[2], 'user': specs[3], 'tag': job_tag})
    return {'jobs': jobs}

def atrm(*args):
    if False:
        print('Hello World!')
    "\n    Remove jobs from the queue.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' at.atrm <jobid> <jobid> .. <jobid>\n        salt '*' at.atrm all\n        salt '*' at.atrm all [tag]\n    "
    if not args:
        return {'jobs': {'removed': [], 'tag': None}}
    if args[0] == 'all':
        if len(args) > 1:
            opts = list(list(map(str, [j['job'] for j in atq(args[1])['jobs']])))
            ret = {'jobs': {'removed': opts, 'tag': args[1]}}
        else:
            opts = list(list(map(str, [j['job'] for j in atq()['jobs']])))
            ret = {'jobs': {'removed': opts, 'tag': None}}
    else:
        opts = list(list(map(str, [i['job'] for i in atq()['jobs'] if i['job'] in args])))
        ret = {'jobs': {'removed': opts, 'tag': None}}
    for job in ret['jobs']['removed']:
        res_job = __salt__['cmd.run_all'](f'atrm {job}')
        if res_job['retcode'] > 0:
            if 'failed' not in ret['jobs']:
                ret['jobs']['failed'] = {}
            ret['jobs']['failed'][job] = res_job['stderr']
    if 'failed' in ret['jobs']:
        for job in ret['jobs']['failed']:
            ret['jobs']['removed'].remove(job)
    return ret

def at(*args, **kwargs):
    if False:
        return 10
    "\n    Add a job to the queue.\n\n    The 'timespec' follows the format documented in the\n    at(1) manpage.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' at.at <timespec> <cmd> [tag=<tag>] [runas=<user>]\n        salt '*' at.at 12:05am '/sbin/reboot' tag=reboot\n        salt '*' at.at '3:05am +3 days' 'bin/myscript' tag=nightly runas=jim\n    "
    if len(args) < 2:
        return {'jobs': []}
    if 'tag' in kwargs:
        stdin = '### SALT: {}\n{}'.format(kwargs['tag'], ' '.join(args[1:]))
    else:
        stdin = ' '.join(args[1:])
    cmd_kwargs = {'stdin': stdin, 'python_shell': False}
    if 'runas' in kwargs:
        cmd_kwargs['runas'] = kwargs['runas']
    res = __salt__['cmd.run_all'](f'at "{args[0]}"', **cmd_kwargs)
    if res['retcode'] > 0:
        if 'bad time specification' in res['stderr']:
            return {'jobs': [], 'error': 'invalid timespec'}
        return {'jobs': [], 'error': res['stderr']}
    else:
        jobid = res['stderr'].splitlines()[1]
        jobid = str(jobid.split()[1])
        return atq(jobid)

def atc(jobid):
    if False:
        print('Hello World!')
    "\n    Print the at(1) script that will run for the passed job\n    id. This is mostly for debugging so the output will\n    just be text.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' at.atc <jobid>\n    "
    atjob_file = f'/var/spool/cron/atjobs/{jobid}'
    if __salt__['file.file_exists'](atjob_file):
        with salt.utils.files.fopen(atjob_file, 'r') as rfh:
            return ''.join([salt.utils.stringutils.to_unicode(x) for x in rfh.readlines()])
    else:
        return {'error': f"invalid job id '{jobid}'"}

def _atq(**kwargs):
    if False:
        for i in range(10):
            print('nop')
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
        while True:
            i = 10
    "\n    Check the job from queue.\n    The kwargs dict include 'hour minute day month year tag runas'\n    Other parameters will be ignored.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' at.jobcheck runas=jam day=13\n        salt '*' at.jobcheck day=13 month=12 year=13 tag=rose\n    "
    if not kwargs:
        return {'error': 'You have given a condition'}
    return _atq(**kwargs)