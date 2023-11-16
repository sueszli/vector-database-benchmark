import logging
import os
import subprocess
import utils

def _find_prog(prog):
    if False:
        print('Hello World!')
    try:
        return subprocess.check_output(['which', prog], stderr=utils.DEV_NULL).strip()
    except subprocess.CalledProcessError:
        return None

def _exec_prog(prog):
    if False:
        for i in range(10):
            print('nop')
    logging.info('executing "%s"' % prog)
    return os.system(prog) == 0

def shut_down():
    if False:
        print('Hello World!')
    logging.info('shutting down')
    prog = _find_prog('poweroff')
    if prog:
        return _exec_prog(prog)
    prog = _find_prog('shutdown')
    if prog:
        return _exec_prog(prog + ' -h now')
    prog = _find_prog('systemctl')
    if prog:
        return _exec_prog(prog + ' poweroff')
    prog = _find_prog('init')
    if prog:
        return _exec_prog(prog + ' 0')
    return False

def reboot():
    if False:
        return 10
    logging.info('rebooting')
    prog = _find_prog('reboot')
    if prog:
        return _exec_prog(prog)
    prog = _find_prog('shutdown')
    if prog:
        return _exec_prog(prog + ' -r now')
    prog = _find_prog('systemctl')
    if prog:
        return _exec_prog(prog + ' reboot')
    prog = _find_prog('init')
    if prog:
        return _exec_prog(prog + ' 6')
    return False