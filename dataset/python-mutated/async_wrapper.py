from __future__ import annotations
import errno
import json
import shlex
import shutil
import os
import subprocess
import sys
import traceback
import signal
import time
import syslog
import multiprocessing
from ansible.module_utils.common.text.converters import to_text, to_bytes
PY3 = sys.version_info[0] == 3
syslog.openlog('ansible-%s' % os.path.basename(__file__))
syslog.syslog(syslog.LOG_NOTICE, 'Invoked with %s' % ' '.join(sys.argv[1:]))
(ipc_watcher, ipc_notifier) = multiprocessing.Pipe()
job_path = ''

def notice(msg):
    if False:
        for i in range(10):
            print('nop')
    syslog.syslog(syslog.LOG_NOTICE, msg)

def end(res=None, exit_msg=0):
    if False:
        i = 10
        return i + 15
    if res is not None:
        print(json.dumps(res))
    sys.stdout.flush()
    sys.exit(exit_msg)

def daemonize_self():
    if False:
        return 10
    try:
        pid = os.fork()
        if pid > 0:
            end()
    except OSError:
        e = sys.exc_info()[1]
        end({'msg': 'fork #1 failed: %d (%s)\n' % (e.errno, e.strerror), 'failed': True}, 1)
    os.setsid()
    os.umask(int('022', 8))
    try:
        pid = os.fork()
        if pid > 0:
            end()
    except OSError:
        e = sys.exc_info()[1]
        end({'msg': 'fork #2 failed: %d (%s)\n' % (e.errno, e.strerror), 'failed': True}, 1)
    dev_null = open('/dev/null', 'w')
    os.dup2(dev_null.fileno(), sys.stdin.fileno())
    os.dup2(dev_null.fileno(), sys.stdout.fileno())
    os.dup2(dev_null.fileno(), sys.stderr.fileno())

def _filter_non_json_lines(data):
    if False:
        print('Hello World!')
    "\n    Used to filter unrelated output around module JSON output, like messages from\n    tcagetattr, or where dropbear spews MOTD on every single command (which is nuts).\n\n    Filters leading lines before first line-starting occurrence of '{', and filter all\n    trailing lines after matching close character (working from the bottom of output).\n    "
    warnings = []
    lines = data.splitlines()
    for (start, line) in enumerate(lines):
        line = line.strip()
        if line.startswith(u'{'):
            break
    else:
        raise ValueError('No start of json char found')
    lines = lines[start:]
    for (reverse_end_offset, line) in enumerate(reversed(lines)):
        if line.strip().endswith(u'}'):
            break
    else:
        raise ValueError('No end of json char found')
    if reverse_end_offset > 0:
        trailing_junk = lines[len(lines) - reverse_end_offset:]
        warnings.append('Module invocation had junk after the JSON data: %s' % '\n'.join(trailing_junk))
    lines = lines[:len(lines) - reverse_end_offset]
    return ('\n'.join(lines), warnings)

def _get_interpreter(module_path):
    if False:
        return 10
    with open(module_path, 'rb') as module_fd:
        head = module_fd.read(1024)
        if head[0:2] != b'#!':
            return None
        return head[2:head.index(b'\n')].strip().split(b' ')

def _make_temp_dir(path):
    if False:
        i = 10
        return i + 15
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def jwrite(info):
    if False:
        i = 10
        return i + 15
    jobfile = job_path + '.tmp'
    tjob = open(jobfile, 'w')
    try:
        tjob.write(json.dumps(info))
    except (IOError, OSError) as e:
        notice('failed to write to %s: %s' % (jobfile, str(e)))
        raise e
    finally:
        tjob.close()
        os.rename(jobfile, job_path)

def _run_module(wrapped_cmd, jid):
    if False:
        i = 10
        return i + 15
    jwrite({'started': 1, 'finished': 0, 'ansible_job_id': jid})
    result = {}
    ipc_notifier.send(True)
    ipc_notifier.close()
    outdata = ''
    filtered_outdata = ''
    stderr = ''
    try:
        cmd = [to_bytes(c, errors='surrogate_or_strict') for c in shlex.split(wrapped_cmd)]
        interpreter = _get_interpreter(cmd[0])
        if interpreter:
            cmd = interpreter + cmd
        script = subprocess.Popen(cmd, shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (outdata, stderr) = script.communicate()
        if PY3:
            outdata = outdata.decode('utf-8', 'surrogateescape')
            stderr = stderr.decode('utf-8', 'surrogateescape')
        (filtered_outdata, json_warnings) = _filter_non_json_lines(outdata)
        result = json.loads(filtered_outdata)
        if json_warnings:
            module_warnings = result.get('warnings', [])
            if not isinstance(module_warnings, list):
                module_warnings = [module_warnings]
            module_warnings.extend(json_warnings)
            result['warnings'] = module_warnings
        if stderr:
            result['stderr'] = stderr
        jwrite(result)
    except (OSError, IOError):
        e = sys.exc_info()[1]
        result = {'failed': 1, 'cmd': wrapped_cmd, 'msg': to_text(e), 'outdata': outdata, 'stderr': stderr}
        result['ansible_job_id'] = jid
        jwrite(result)
    except (ValueError, Exception):
        result = {'failed': 1, 'cmd': wrapped_cmd, 'data': outdata, 'stderr': stderr, 'msg': traceback.format_exc()}
        result['ansible_job_id'] = jid
        jwrite(result)

def main():
    if False:
        while True:
            i = 10
    if len(sys.argv) < 5:
        end({'failed': True, 'msg': 'usage: async_wrapper <jid> <time_limit> <modulescript> <argsfile> [-preserve_tmp]  Humans, do not call directly!'}, 1)
    jid = '%s.%d' % (sys.argv[1], os.getpid())
    time_limit = sys.argv[2]
    wrapped_module = sys.argv[3]
    argsfile = sys.argv[4]
    if '-tmp-' not in os.path.dirname(wrapped_module):
        preserve_tmp = True
    elif len(sys.argv) > 5:
        preserve_tmp = sys.argv[5] == '-preserve_tmp'
    else:
        preserve_tmp = False
    if argsfile != '_':
        cmd = '%s %s' % (wrapped_module, argsfile)
    else:
        cmd = wrapped_module
    step = 5
    async_dir = os.environ.get('ANSIBLE_ASYNC_DIR', '~/.ansible_async')
    jobdir = os.path.expanduser(async_dir)
    global job_path
    job_path = os.path.join(jobdir, jid)
    try:
        _make_temp_dir(jobdir)
    except Exception as e:
        end({'failed': 1, 'msg': 'could not create directory: %s - %s' % (jobdir, to_text(e)), 'exception': to_text(traceback.format_exc())}, 1)
    try:
        pid = os.fork()
        if pid:
            ipc_notifier.close()
            retries = 25
            while retries > 0:
                if ipc_watcher.poll(0.1):
                    break
                else:
                    retries = retries - 1
                    continue
            notice('Return async_wrapper task started.')
            end({'failed': 0, 'started': 1, 'finished': 0, 'ansible_job_id': jid, 'results_file': job_path, '_ansible_suppress_tmpdir_delete': not preserve_tmp}, 0)
        else:
            ipc_watcher.close()
            daemonize_self()
            notice('Starting module and watcher')
            sub_pid = os.fork()
            if sub_pid:
                ipc_watcher.close()
                ipc_notifier.close()
                remaining = int(time_limit)
                os.setpgid(sub_pid, sub_pid)
                notice('Start watching %s (%s)' % (sub_pid, remaining))
                time.sleep(step)
                while os.waitpid(sub_pid, os.WNOHANG) == (0, 0):
                    notice('%s still running (%s)' % (sub_pid, remaining))
                    time.sleep(step)
                    remaining = remaining - step
                    if remaining <= 0:
                        res = {'msg': 'Timeout exceeded', 'failed': True, 'child_pid': sub_pid}
                        jwrite(res)
                        notice('Timeout reached, now killing %s' % sub_pid)
                        os.killpg(sub_pid, signal.SIGKILL)
                        notice('Sent kill to group %s ' % sub_pid)
                        time.sleep(1)
                        if not preserve_tmp:
                            shutil.rmtree(os.path.dirname(wrapped_module), True)
                        end(res)
                notice('Done in kid B.')
                if not preserve_tmp:
                    shutil.rmtree(os.path.dirname(wrapped_module), True)
                end()
            else:
                notice('Start module (%s)' % os.getpid())
                _run_module(cmd, jid)
                notice('Module complete (%s)' % os.getpid())
    except Exception as e:
        notice('error: %s' % e)
        end({'failed': True, 'msg': 'FATAL ERROR: %s' % e}, 'async_wrapper exited prematurely')
if __name__ == '__main__':
    main()