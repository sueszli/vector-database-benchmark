import logging
import os
import pathlib
import time
import salt.utils.decorators
log = logging.getLogger(__name__)
REPO_ROOT_DIR = pathlib.Path(os.environ['REPO_ROOT_DIR']).resolve()
STATE_BASE_DIR = REPO_ROOT_DIR / 'tests' / 'integration' / 'files' / 'file' / 'base'
EXIT_CODE_SH = STATE_BASE_DIR / 'exit_code.sh'
EXIT_CODE_CMD = STATE_BASE_DIR / 'exit_code.cmd'

def _exit_code(code):
    if False:
        for i in range(10):
            print('nop')
    if os.name == 'nt':
        cmd = 'cmd /c {} {}'.format(EXIT_CODE_CMD, code)
    else:
        cmd = '/usr/bin/env sh {} {}'.format(EXIT_CODE_SH, code)
    return cmd

def _fallbackfunc():
    if False:
        return 10
    return (False, 'fallback')

def working_function():
    if False:
        return 10
    return True

@salt.utils.decorators.depends(True)
def booldependsTrue():
    if False:
        i = 10
        return i + 15
    return True

@salt.utils.decorators.depends(False)
def booldependsFalse():
    if False:
        for i in range(10):
            print('nop')
    return True

@salt.utils.decorators.depends('time')
def depends():
    if False:
        for i in range(10):
            print('nop')
    ret = {'ret': True, 'time': time.time()}
    return ret

@salt.utils.decorators.depends('time123')
def missing_depends():
    if False:
        i = 10
        return i + 15
    return True

@salt.utils.decorators.depends('time', fallback_function=_fallbackfunc)
def depends_will_not_fallback():
    if False:
        i = 10
        return i + 15
    ret = {'ret': True, 'time': time.time()}
    return ret

@salt.utils.decorators.depends('time123', fallback_function=_fallbackfunc)
def missing_depends_will_fallback():
    if False:
        i = 10
        return i + 15
    ret = {'ret': True, 'time': time.time()}
    return ret

@salt.utils.decorators.depends(_exit_code(42), retcode=42)
def command_success_retcode():
    if False:
        return 10
    return True

@salt.utils.decorators.depends(_exit_code(42), retcode=0)
def command_failure_retcode():
    if False:
        for i in range(10):
            print('nop')
    return True

@salt.utils.decorators.depends(_exit_code(42), nonzero_retcode=True)
def command_success_nonzero_retcode_true():
    if False:
        return 10
    return True

@salt.utils.decorators.depends(_exit_code(0), nonzero_retcode=True)
def command_failure_nonzero_retcode_true():
    if False:
        i = 10
        return i + 15
    return True

@salt.utils.decorators.depends(_exit_code(0), nonzero_retcode=False)
def command_success_nonzero_retcode_false():
    if False:
        while True:
            i = 10
    return True

@salt.utils.decorators.depends(_exit_code(42), nonzero_retcode=False)
def command_failure_nonzero_retcode_false():
    if False:
        for i in range(10):
            print('nop')
    return True

@salt.utils.decorators.depends('depends_versioned', version='1.0')
def version_depends_false():
    if False:
        i = 10
        return i + 15
    return True

@salt.utils.decorators.depends('depends_versioned', version='2.0')
def version_depends_true():
    if False:
        print('Hello World!')
    return True

@salt.utils.decorators.depends('depends_versionless', version='0.2')
def version_depends_versionless_true():
    if False:
        for i in range(10):
            print('nop')
    return True