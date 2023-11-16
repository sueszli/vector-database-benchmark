"""
Functions for creating and working with job IDs
"""
import datetime
import hashlib
import os
from calendar import month_abbr as months
import salt.utils.stringutils
LAST_JID_DATETIME = None

def _utc_now():
    if False:
        return 10
    '\n    Helper method so tests do not have to patch the built-in method.\n    '
    return datetime.datetime.utcnow()

def gen_jid(opts):
    if False:
        print('Hello World!')
    '\n    Generate a jid\n    '
    global LAST_JID_DATETIME
    jid_dt = _utc_now()
    if not opts.get('unique_jid', False):
        return '{:%Y%m%d%H%M%S%f}'.format(jid_dt)
    if LAST_JID_DATETIME and LAST_JID_DATETIME >= jid_dt:
        jid_dt = LAST_JID_DATETIME + datetime.timedelta(microseconds=1)
    LAST_JID_DATETIME = jid_dt
    return '{:%Y%m%d%H%M%S%f}_{}'.format(jid_dt, os.getpid())

def is_jid(jid):
    if False:
        return 10
    '\n    Returns True if the passed in value is a job id\n    '
    if not isinstance(jid, str):
        return False
    if len(jid) != 20 and (len(jid) <= 21 or jid[20] != '_'):
        return False
    try:
        int(jid[:20])
        return True
    except ValueError:
        return False

def jid_to_time(jid):
    if False:
        print('Hello World!')
    '\n    Convert a salt job id into the time when the job was invoked\n    '
    jid = str(jid)
    if len(jid) != 20 and (len(jid) <= 21 or jid[20] != '_'):
        return ''
    year = jid[:4]
    month = jid[4:6]
    day = jid[6:8]
    hour = jid[8:10]
    minute = jid[10:12]
    second = jid[12:14]
    micro = jid[14:20]
    ret = '{}, {} {} {}:{}:{}.{}'.format(year, months[int(month)], day, hour, minute, second, micro)
    return ret

def format_job_instance(job):
    if False:
        for i in range(10):
            print('nop')
    '\n    Format the job instance correctly\n    '
    ret = {'Function': job.get('fun', 'unknown-function'), 'Arguments': list(job.get('arg', [])), 'Target': job.get('tgt', 'unknown-target'), 'Target-type': job.get('tgt_type', 'list'), 'User': job.get('user', 'root')}
    if 'metadata' in job:
        ret['Metadata'] = job.get('metadata', {})
    elif 'kwargs' in job:
        if 'metadata' in job['kwargs']:
            ret['Metadata'] = job['kwargs'].get('metadata', {})
    return ret

def format_jid_instance(jid, job):
    if False:
        for i in range(10):
            print('nop')
    '\n    Format the jid correctly\n    '
    ret = format_job_instance(job)
    ret.update({'StartTime': jid_to_time(jid)})
    return ret

def format_jid_instance_ext(jid, job):
    if False:
        while True:
            i = 10
    '\n    Format the jid correctly with jid included\n    '
    ret = format_job_instance(job)
    ret.update({'JID': jid, 'StartTime': jid_to_time(jid)})
    return ret

def jid_dir(jid, job_dir=None, hash_type='sha256'):
    if False:
        return 10
    '\n    Return the jid_dir for the given job id\n    '
    if not isinstance(jid, str):
        jid = str(jid)
    jhash = getattr(hashlib, hash_type)(salt.utils.stringutils.to_bytes(jid)).hexdigest()
    parts = []
    if job_dir is not None:
        parts.append(job_dir)
    parts.extend([jhash[:2], jhash[2:]])
    return os.path.join(*parts)