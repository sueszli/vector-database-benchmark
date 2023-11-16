"""Secures functions for Glances"""
from subprocess import Popen, PIPE
import re
from glances.globals import nativestr

def secure_popen(cmd):
    if False:
        return 10
    'A more or less secure way to execute system commands\n\n    Multiple command should be separated with a &&\n\n    :return: the result of the commands\n    '
    ret = ''
    for c in cmd.split('&&'):
        ret += __secure_popen(c)
    return ret

def __secure_popen(cmd):
    if False:
        i = 10
        return i + 15
    'A more or less secure way to execute system command\n\n    Manage redirection (>) and pipes (|)\n    '
    cmd_split_redirect = cmd.split('>')
    if len(cmd_split_redirect) > 2:
        return 'Glances error: Only one file redirection allowed ({})'.format(cmd)
    elif len(cmd_split_redirect) == 2:
        stdout_redirect = cmd_split_redirect[1].strip()
        cmd = cmd_split_redirect[0]
    else:
        stdout_redirect = None
    sub_cmd_stdin = None
    p_last = None
    for sub_cmd in cmd.split('|'):
        tmp_split = [_ for _ in list(filter(None, re.split('(\\s+)|(".*?"+?)|(\\\'.*?\\\'+?)', sub_cmd))) if _ != ' ']
        sub_cmd_split = [_[1:-1] if _[0] == _[-1] == '"' or _[0] == _[-1] == "'" else _ for _ in tmp_split]
        p = Popen(sub_cmd_split, shell=False, stdin=sub_cmd_stdin, stdout=PIPE, stderr=PIPE)
        if p_last is not None:
            p_last.stdout.close()
            p_last.kill()
            p_last.wait()
        p_last = p
        sub_cmd_stdin = p.stdout
    p_ret = p_last.communicate()
    if nativestr(p_ret[1]) == '':
        ret = nativestr(p_ret[0])
        if stdout_redirect is not None:
            with open(stdout_redirect, 'w') as stdout_redirect_file:
                stdout_redirect_file.write(ret)
    else:
        ret = nativestr(p_ret[1])
    return ret