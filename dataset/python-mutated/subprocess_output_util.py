import os
import re
import subprocess
import sys
import tempfile
import time
from ray.autoscaler._private.cli_logger import cf, cli_logger
CONN_REFUSED_PATIENCE = 30
_redirect_output = False
_allow_interactive = True

def is_output_redirected():
    if False:
        for i in range(10):
            print('nop')
    return _redirect_output

def set_output_redirected(val: bool):
    if False:
        return 10
    'Choose between logging to a temporary file and to `sys.stdout`.\n\n    The default is to log to a file.\n\n    Args:\n        val: If true, subprocess output will be redirected to\n                    a temporary file.\n    '
    global _redirect_output
    _redirect_output = val

def does_allow_interactive():
    if False:
        return 10
    return _allow_interactive

def set_allow_interactive(val: bool):
    if False:
        i = 10
        return i + 15
    'Choose whether to pass on stdin to running commands.\n\n    The default is to pipe stdin and close it immediately.\n\n    Args:\n        val: If true, stdin will be passed to commands.\n    '
    global _allow_interactive
    _allow_interactive = val

class ProcessRunnerError(Exception):

    def __init__(self, msg, msg_type, code=None, command=None, special_case=None):
        if False:
            while True:
                i = 10
        super(ProcessRunnerError, self).__init__('{} (discovered={}): type={}, code={}, command={}'.format(msg, special_case, msg_type, code, command))
        self.msg_type = msg_type
        self.code = code
        self.command = command
        self.special_case = special_case
_ssh_output_regexes = {'known_host_update': re.compile("\\s*Warning: Permanently added '.+' \\(.+\\) to the list of known hosts.\\s*"), 'connection_closed': re.compile('\\s*Shared connection to .+ closed.\\s*'), 'timeout': re.compile('\\s*ssh: connect to host .+ port .+: Operation timed out\\s*'), 'conn_refused': re.compile('\\s*ssh: connect to host .+ port .+: Connection refused\\s*')}

def _read_subprocess_stream(f, output_file, is_stdout=False):
    if False:
        for i in range(10):
            print('nop')
    'Read and process a subprocess output stream.\n\n    The goal is to find error messages and respond to them in a clever way.\n    Currently just used for SSH messages (CONN_REFUSED, TIMEOUT, etc.), so\n    the user does not get confused by these.\n\n    Ran in a thread each for both `stdout` and `stderr` to\n    allow for cross-platform asynchronous IO.\n\n    Note: `select`-based IO is another option, but Windows has\n    no support for `select`ing pipes, and Linux support varies somewhat.\n    Spefically, Older *nix systems might also have quirks in how they\n    handle `select` on pipes.\n\n    Args:\n        f: File object for the stream.\n        output_file: File object to which filtered output is written.\n        is_stdout (bool):\n            When `is_stdout` is `False`, the stream is assumed to\n            be `stderr`. Different error message detectors are used,\n            and the output is displayed to the user unless it matches\n            a special case (e.g. SSH timeout), in which case this is\n            left up to the caller.\n    '
    detected_special_case = None
    while True:
        line = f.readline()
        if line is None or line == '':
            break
        if line[-1] == '\n':
            line = line[:-1]
        if not is_stdout:
            if _ssh_output_regexes['connection_closed'].fullmatch(line) is not None:
                continue
            if _ssh_output_regexes['timeout'].fullmatch(line) is not None:
                if detected_special_case is not None:
                    raise ValueError('Bug: ssh_timeout conflicts with another special codition: ' + detected_special_case)
                detected_special_case = 'ssh_timeout'
                continue
            if _ssh_output_regexes['conn_refused'].fullmatch(line) is not None:
                if detected_special_case is not None:
                    raise ValueError('Bug: ssh_conn_refused conflicts with another special codition: ' + detected_special_case)
                detected_special_case = 'ssh_conn_refused'
                continue
            if _ssh_output_regexes['known_host_update'].fullmatch(line) is not None:
                continue
            cli_logger.error(line)
        if output_file is not None and output_file != subprocess.DEVNULL:
            output_file.write(line + '\n')
    return detected_special_case

def _run_and_process_output(cmd, stdout_file, process_runner=subprocess, stderr_file=None, use_login_shells=False):
    if False:
        print('Hello World!')
    'Run a command and process its output for special cases.\n\n    Calls a standard \'check_call\' if process_runner is not subprocess.\n\n    Specifically, run all command output through regex to detect\n    error conditions and filter out non-error messages that went to stderr\n    anyway (SSH writes ALL of its "system" messages to stderr even if they\n    are not actually errors).\n\n    Args:\n        cmd (List[str]): Command to run.\n        process_runner: Used for command execution. Assumed to have\n            \'check_call\' and \'check_output\' inplemented.\n        stdout_file: File to redirect stdout to.\n        stderr_file: File to redirect stderr to.\n\n    Implementation notes:\n    1. `use_login_shells` disables special processing\n    If we run interactive apps, output processing will likely get\n    overwhelmed with the interactive output elements.\n    Thus, we disable output processing for login shells. This makes\n    the logging experience considerably worse, but it only degrades\n    to old-style logging.\n\n    For example, `pip install` outputs HUNDREDS of progress-bar lines\n    when downloading a package, and we have to\n    read + regex + write all of them.\n\n    After all, even just printing output to console can often slow\n    down a fast-printing app, and we do more than just print, and\n    all that from Python, which is much slower than C regarding\n    stream processing.\n\n    2. `stdin=PIPE` for subprocesses\n    Do not inherit stdin as it messes with bash signals\n    (ctrl-C for SIGINT) and these commands aren\'t supposed to\n    take input anyway.\n\n    3. `ThreadPoolExecutor` without the `Pool`\n    We use `ThreadPoolExecutor` to create futures from threads.\n    Threads are never reused.\n\n    This approach allows us to have no custom synchronization by\n    off-loading the return value and exception passing to the\n    standard library (`ThreadPoolExecutor` internals).\n\n    This instance will be `shutdown()` ASAP so it\'s fine to\n    create one in such a weird place.\n\n    The code is thus 100% thread-safe as long as the stream readers\n    are read-only except for return values and possible exceptions.\n    '
    stdin_overwrite = subprocess.PIPE
    assert not (does_allow_interactive() and is_output_redirected()), 'Cannot redirect output while in interactive mode.'
    if process_runner != subprocess or (does_allow_interactive() and (not is_output_redirected())):
        stdin_overwrite = None
    if use_login_shells or process_runner != subprocess:
        return process_runner.check_call(cmd, stdin=stdin_overwrite, stdout=stdout_file, stderr=stderr_file)
    with subprocess.Popen(cmd, stdin=stdin_overwrite, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True) as p:
        from concurrent.futures import ThreadPoolExecutor
        p.stdin.close()
        with ThreadPoolExecutor(max_workers=2) as executor:
            stdout_future = executor.submit(_read_subprocess_stream, p.stdout, stdout_file, is_stdout=True)
            stderr_future = executor.submit(_read_subprocess_stream, p.stderr, stderr_file, is_stdout=False)
            executor.shutdown()
            p.poll()
            detected_special_case = stdout_future.result()
            if stderr_future.result() is not None:
                if detected_special_case is not None:
                    raise ValueError('Bug: found a special case in both stdout and stderr. This is not valid behavior at the time of writing this code.')
                detected_special_case = stderr_future.result()
            if p.returncode > 0:
                raise ProcessRunnerError('Command failed', 'ssh_command_failed', code=p.returncode, command=cmd, special_case=detected_special_case)
            elif p.returncode < 0:
                raise ProcessRunnerError('Command failed', 'ssh_command_failed', code=p.returncode, command=cmd, special_case='died_to_signal')
            return p.returncode

def run_cmd_redirected(cmd, process_runner=subprocess, silent=False, use_login_shells=False):
    if False:
        i = 10
        return i + 15
    'Run a command and optionally redirect output to a file.\n\n    Args:\n        cmd (List[str]): Command to run.\n        process_runner: Process runner used for executing commands.\n        silent: If true, the command output will be silenced completely\n                       (redirected to /dev/null), unless verbose logging\n                       is enabled. Use this for running utility commands like\n                       rsync.\n    '
    if silent and cli_logger.verbosity < 1:
        return _run_and_process_output(cmd, process_runner=process_runner, stdout_file=process_runner.DEVNULL, stderr_file=process_runner.DEVNULL, use_login_shells=use_login_shells)
    if not is_output_redirected():
        return _run_and_process_output(cmd, process_runner=process_runner, stdout_file=sys.stdout, stderr_file=sys.stderr, use_login_shells=use_login_shells)
    else:
        tmpfile_path = os.path.join(tempfile.gettempdir(), 'ray-up-{}-{}.txt'.format(cmd[0], time.time()))
        with open(tmpfile_path, mode='w', buffering=1) as tmp:
            cli_logger.verbose('Command stdout is redirected to {}', cf.bold(tmp.name))
            return _run_and_process_output(cmd, process_runner=process_runner, stdout_file=tmp, stderr_file=tmp, use_login_shells=use_login_shells)

def handle_ssh_fails(e, first_conn_refused_time, retry_interval):
    if False:
        i = 10
        return i + 15
    'Handle SSH system failures coming from a subprocess.\n\n    Args:\n        e: The `ProcessRunnerException` to handle.\n        first_conn_refused_time:\n            The time (as reported by this function) or None,\n            indicating the last time a CONN_REFUSED error was caught.\n\n            After exceeding a patience value, the program will be aborted\n            since SSH will likely never recover.\n        retry_interval: The interval after which the command will be retried,\n                        used here just to inform the user.\n    '
    if e.msg_type != 'ssh_command_failed':
        return
    if e.special_case == 'ssh_conn_refused':
        if first_conn_refused_time is not None and time.time() - first_conn_refused_time > CONN_REFUSED_PATIENCE:
            cli_logger.error('SSH connection was being refused for {} seconds. Head node assumed unreachable.', cf.bold(str(CONN_REFUSED_PATIENCE)))
            cli_logger.abort("Check the node's firewall settings and the cloud network configuration.")
        cli_logger.warning('SSH connection was refused.')
        cli_logger.warning('This might mean that the SSH daemon is still setting up, or that the host is inaccessable (e.g. due to a firewall).')
        return time.time()
    if e.special_case in ['ssh_timeout', 'ssh_conn_refused']:
        cli_logger.print('SSH still not available, retrying in {} seconds.', cf.bold(str(retry_interval)))
    else:
        raise e
    return first_conn_refused_time