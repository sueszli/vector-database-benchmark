import ast
import re
import bandit
from bandit.core import issue
from bandit.core import test_properties as test
full_path_match = re.compile('^(?:[A-Za-z](?=\\:)|[\\\\\\/\\.])')

def _evaluate_shell_call(context):
    if False:
        for i in range(10):
            print('nop')
    no_formatting = isinstance(context.node.args[0], ast.Str)
    if no_formatting:
        return bandit.LOW
    else:
        return bandit.HIGH

def gen_config(name):
    if False:
        for i in range(10):
            print('nop')
    if name == 'shell_injection':
        return {'subprocess': ['subprocess.Popen', 'subprocess.call', 'subprocess.check_call', 'subprocess.check_output', 'subprocess.run'], 'shell': ['os.system', 'os.popen', 'os.popen2', 'os.popen3', 'os.popen4', 'popen2.popen2', 'popen2.popen3', 'popen2.popen4', 'popen2.Popen3', 'popen2.Popen4', 'commands.getoutput', 'commands.getstatusoutput'], 'no_shell': ['os.execl', 'os.execle', 'os.execlp', 'os.execlpe', 'os.execv', 'os.execve', 'os.execvp', 'os.execvpe', 'os.spawnl', 'os.spawnle', 'os.spawnlp', 'os.spawnlpe', 'os.spawnv', 'os.spawnve', 'os.spawnvp', 'os.spawnvpe', 'os.startfile']}

def has_shell(context):
    if False:
        while True:
            i = 10
    keywords = context.node.keywords
    result = False
    if 'shell' in context.call_keywords:
        for key in keywords:
            if key.arg == 'shell':
                val = key.value
                if isinstance(val, ast.Num):
                    result = bool(val.n)
                elif isinstance(val, ast.List):
                    result = bool(val.elts)
                elif isinstance(val, ast.Dict):
                    result = bool(val.keys)
                elif isinstance(val, ast.Name) and val.id in ['False', 'None']:
                    result = False
                elif isinstance(val, ast.NameConstant):
                    result = val.value
                else:
                    result = True
    return result

@test.takes_config('shell_injection')
@test.checks('Call')
@test.test_id('B602')
def subprocess_popen_with_shell_equals_true(context, config):
    if False:
        while True:
            i = 10
    "**B602: Test for use of popen with shell equals true**\n\n    Python possesses many mechanisms to invoke an external executable. However,\n    doing so may present a security issue if appropriate care is not taken to\n    sanitize any user provided or variable input.\n\n    This plugin test is part of a family of tests built to check for process\n    spawning and warn appropriately. Specifically, this test looks for the\n    spawning of a subprocess using a command shell. This type of subprocess\n    invocation is dangerous as it is vulnerable to various shell injection\n    attacks. Great care should be taken to sanitize all input in order to\n    mitigate this risk. Calls of this type are identified by a parameter of\n    'shell=True' being given.\n\n    Additionally, this plugin scans the command string given and adjusts its\n    reported severity based on how it is presented. If the command string is a\n    simple static string containing no special shell characters, then the\n    resulting issue has low severity. If the string is static, but contains\n    shell formatting characters or wildcards, then the reported issue is\n    medium. Finally, if the string is computed using Python's string\n    manipulation or formatting operations, then the reported issue has high\n    severity. These severity levels reflect the likelihood that the code is\n    vulnerable to injection.\n\n    See also:\n\n    - :doc:`../plugins/linux_commands_wildcard_injection`\n    - :doc:`../plugins/subprocess_without_shell_equals_true`\n    - :doc:`../plugins/start_process_with_no_shell`\n    - :doc:`../plugins/start_process_with_a_shell`\n    - :doc:`../plugins/start_process_with_partial_path`\n\n    **Config Options:**\n\n    This plugin test shares a configuration with others in the same family,\n    namely `shell_injection`. This configuration is divided up into three\n    sections, `subprocess`, `shell` and `no_shell`. They each list Python calls\n    that spawn subprocesses, invoke commands within a shell, or invoke commands\n    without a shell (by replacing the calling process) respectively.\n\n    This plugin specifically scans for methods listed in `subprocess` section\n    that have shell=True specified.\n\n    .. code-block:: yaml\n\n        shell_injection:\n\n            # Start a process using the subprocess module, or one of its\n            wrappers.\n            subprocess:\n                - subprocess.Popen\n                - subprocess.call\n\n\n    :Example:\n\n    .. code-block:: none\n\n        >> Issue: subprocess call with shell=True seems safe, but may be\n        changed in the future, consider rewriting without shell\n           Severity: Low   Confidence: High\n           CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)\n           Location: ./examples/subprocess_shell.py:21\n        20  subprocess.check_call(['/bin/ls', '-l'], shell=False)\n        21  subprocess.check_call('/bin/ls -l', shell=True)\n        22\n\n        >> Issue: call with shell=True contains special shell characters,\n        consider moving extra logic into Python code\n           Severity: Medium   Confidence: High\n           CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)\n           Location: ./examples/subprocess_shell.py:26\n        25\n        26  subprocess.Popen('/bin/ls *', shell=True)\n        27  subprocess.Popen('/bin/ls %s' % ('something',), shell=True)\n\n        >> Issue: subprocess call with shell=True identified, security issue.\n           Severity: High   Confidence: High\n           CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)\n           Location: ./examples/subprocess_shell.py:27\n        26  subprocess.Popen('/bin/ls *', shell=True)\n        27  subprocess.Popen('/bin/ls %s' % ('something',), shell=True)\n        28  subprocess.Popen('/bin/ls {}'.format('something'), shell=True)\n\n    .. seealso::\n\n     - https://security.openstack.org\n     - https://docs.python.org/3/library/subprocess.html#frequently-used-arguments\n     - https://security.openstack.org/guidelines/dg_use-subprocess-securely.html\n     - https://security.openstack.org/guidelines/dg_avoid-shell-true.html\n     - https://cwe.mitre.org/data/definitions/78.html\n\n    .. versionadded:: 0.9.0\n\n    .. versionchanged:: 1.7.3\n        CWE information added\n\n    "
    if config and context.call_function_name_qual in config['subprocess']:
        if has_shell(context):
            if len(context.call_args) > 0:
                sev = _evaluate_shell_call(context)
                if sev == bandit.LOW:
                    return bandit.Issue(severity=bandit.LOW, confidence=bandit.HIGH, cwe=issue.Cwe.OS_COMMAND_INJECTION, text='subprocess call with shell=True seems safe, but may be changed in the future, consider rewriting without shell', lineno=context.get_lineno_for_call_arg('shell'))
                else:
                    return bandit.Issue(severity=bandit.HIGH, confidence=bandit.HIGH, cwe=issue.Cwe.OS_COMMAND_INJECTION, text='subprocess call with shell=True identified, security issue.', lineno=context.get_lineno_for_call_arg('shell'))

@test.takes_config('shell_injection')
@test.checks('Call')
@test.test_id('B603')
def subprocess_without_shell_equals_true(context, config):
    if False:
        print('Hello World!')
    "**B603: Test for use of subprocess without shell equals true**\n\n    Python possesses many mechanisms to invoke an external executable. However,\n    doing so may present a security issue if appropriate care is not taken to\n    sanitize any user provided or variable input.\n\n    This plugin test is part of a family of tests built to check for process\n    spawning and warn appropriately. Specifically, this test looks for the\n    spawning of a subprocess without the use of a command shell. This type of\n    subprocess invocation is not vulnerable to shell injection attacks, but\n    care should still be taken to ensure validity of input.\n\n    Because this is a lesser issue than that described in\n    `subprocess_popen_with_shell_equals_true` a LOW severity warning is\n    reported.\n\n    See also:\n\n    - :doc:`../plugins/linux_commands_wildcard_injection`\n    - :doc:`../plugins/subprocess_popen_with_shell_equals_true`\n    - :doc:`../plugins/start_process_with_no_shell`\n    - :doc:`../plugins/start_process_with_a_shell`\n    - :doc:`../plugins/start_process_with_partial_path`\n\n    **Config Options:**\n\n    This plugin test shares a configuration with others in the same family,\n    namely `shell_injection`. This configuration is divided up into three\n    sections, `subprocess`, `shell` and `no_shell`. They each list Python calls\n    that spawn subprocesses, invoke commands within a shell, or invoke commands\n    without a shell (by replacing the calling process) respectively.\n\n    This plugin specifically scans for methods listed in `subprocess` section\n    that have shell=False specified.\n\n    .. code-block:: yaml\n\n        shell_injection:\n            # Start a process using the subprocess module, or one of its\n            wrappers.\n            subprocess:\n                - subprocess.Popen\n                - subprocess.call\n\n    :Example:\n\n    .. code-block:: none\n\n        >> Issue: subprocess call - check for execution of untrusted input.\n           Severity: Low   Confidence: High\n           CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)\n           Location: ./examples/subprocess_shell.py:23\n        22\n        23    subprocess.check_output(['/bin/ls', '-l'])\n        24\n\n    .. seealso::\n\n     - https://security.openstack.org\n     - https://docs.python.org/3/library/subprocess.html#frequently-used-arguments\n     - https://security.openstack.org/guidelines/dg_avoid-shell-true.html\n     - https://security.openstack.org/guidelines/dg_use-subprocess-securely.html\n     - https://cwe.mitre.org/data/definitions/78.html\n\n    .. versionadded:: 0.9.0\n\n    .. versionchanged:: 1.7.3\n        CWE information added\n\n    "
    if config and context.call_function_name_qual in config['subprocess']:
        if not has_shell(context):
            return bandit.Issue(severity=bandit.LOW, confidence=bandit.HIGH, cwe=issue.Cwe.OS_COMMAND_INJECTION, text='subprocess call - check for execution of untrusted input.', lineno=context.get_lineno_for_call_arg('shell'))

@test.takes_config('shell_injection')
@test.checks('Call')
@test.test_id('B604')
def any_other_function_with_shell_equals_true(context, config):
    if False:
        for i in range(10):
            print('nop')
    "**B604: Test for any function with shell equals true**\n\n    Python possesses many mechanisms to invoke an external executable. However,\n    doing so may present a security issue if appropriate care is not taken to\n    sanitize any user provided or variable input.\n\n    This plugin test is part of a family of tests built to check for process\n    spawning and warn appropriately. Specifically, this plugin test\n    interrogates method calls for the presence of a keyword parameter `shell`\n    equalling true. It is related to detection of shell injection issues and is\n    intended to catch custom wrappers to vulnerable methods that may have been\n    created.\n\n    See also:\n\n    - :doc:`../plugins/linux_commands_wildcard_injection`\n    - :doc:`../plugins/subprocess_popen_with_shell_equals_true`\n    - :doc:`../plugins/subprocess_without_shell_equals_true`\n    - :doc:`../plugins/start_process_with_no_shell`\n    - :doc:`../plugins/start_process_with_a_shell`\n    - :doc:`../plugins/start_process_with_partial_path`\n\n    **Config Options:**\n\n    This plugin test shares a configuration with others in the same family,\n    namely `shell_injection`. This configuration is divided up into three\n    sections, `subprocess`, `shell` and `no_shell`. They each list Python calls\n    that spawn subprocesses, invoke commands within a shell, or invoke commands\n    without a shell (by replacing the calling process) respectively.\n\n    Specifically, this plugin excludes those functions listed under the\n    subprocess section, these methods are tested in a separate specific test\n    plugin and this exclusion prevents duplicate issue reporting.\n\n    .. code-block:: yaml\n\n        shell_injection:\n            # Start a process using the subprocess module, or one of its\n            wrappers.\n            subprocess: [subprocess.Popen, subprocess.call,\n                         subprocess.check_call, subprocess.check_output\n                         execute_with_timeout]\n\n\n    :Example:\n\n    .. code-block:: none\n\n        >> Issue: Function call with shell=True parameter identified, possible\n        security issue.\n           Severity: Medium   Confidence: High\n           CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)\n           Location: ./examples/subprocess_shell.py:9\n        8 pop('/bin/gcc --version', shell=True)\n        9 Popen('/bin/gcc --version', shell=True)\n        10\n\n    .. seealso::\n\n     - https://security.openstack.org/guidelines/dg_avoid-shell-true.html\n     - https://security.openstack.org/guidelines/dg_use-subprocess-securely.html\n     - https://cwe.mitre.org/data/definitions/78.html\n\n    .. versionadded:: 0.9.0\n\n    .. versionchanged:: 1.7.3\n        CWE information added\n\n    "
    if config and context.call_function_name_qual not in config['subprocess']:
        if has_shell(context):
            return bandit.Issue(severity=bandit.MEDIUM, confidence=bandit.LOW, cwe=issue.Cwe.OS_COMMAND_INJECTION, text='Function call with shell=True parameter identified, possible security issue.', lineno=context.get_lineno_for_call_arg('shell'))

@test.takes_config('shell_injection')
@test.checks('Call')
@test.test_id('B605')
def start_process_with_a_shell(context, config):
    if False:
        while True:
            i = 10
    "**B605: Test for starting a process with a shell**\n\n    Python possesses many mechanisms to invoke an external executable. However,\n    doing so may present a security issue if appropriate care is not taken to\n    sanitize any user provided or variable input.\n\n    This plugin test is part of a family of tests built to check for process\n    spawning and warn appropriately. Specifically, this test looks for the\n    spawning of a subprocess using a command shell. This type of subprocess\n    invocation is dangerous as it is vulnerable to various shell injection\n    attacks. Great care should be taken to sanitize all input in order to\n    mitigate this risk. Calls of this type are identified by the use of certain\n    commands which are known to use shells. Bandit will report a LOW\n    severity warning.\n\n    See also:\n\n    - :doc:`../plugins/linux_commands_wildcard_injection`\n    - :doc:`../plugins/subprocess_without_shell_equals_true`\n    - :doc:`../plugins/start_process_with_no_shell`\n    - :doc:`../plugins/start_process_with_partial_path`\n    - :doc:`../plugins/subprocess_popen_with_shell_equals_true`\n\n    **Config Options:**\n\n    This plugin test shares a configuration with others in the same family,\n    namely `shell_injection`. This configuration is divided up into three\n    sections, `subprocess`, `shell` and `no_shell`. They each list Python calls\n    that spawn subprocesses, invoke commands within a shell, or invoke commands\n    without a shell (by replacing the calling process) respectively.\n\n    This plugin specifically scans for methods listed in `shell` section.\n\n    .. code-block:: yaml\n\n        shell_injection:\n            shell:\n                - os.system\n                - os.popen\n                - os.popen2\n                - os.popen3\n                - os.popen4\n                - popen2.popen2\n                - popen2.popen3\n                - popen2.popen4\n                - popen2.Popen3\n                - popen2.Popen4\n                - commands.getoutput\n                - commands.getstatusoutput\n\n    :Example:\n\n    .. code-block:: none\n\n        >> Issue: Starting a process with a shell: check for injection.\n           Severity: Low   Confidence: Medium\n           CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)\n           Location: examples/os_system.py:3\n        2\n        3   os.system('/bin/echo hi')\n\n    .. seealso::\n\n     - https://security.openstack.org\n     - https://docs.python.org/3/library/os.html#os.system\n     - https://docs.python.org/3/library/subprocess.html#frequently-used-arguments\n     - https://security.openstack.org/guidelines/dg_use-subprocess-securely.html\n     - https://cwe.mitre.org/data/definitions/78.html\n\n    .. versionadded:: 0.10.0\n\n    .. versionchanged:: 1.7.3\n        CWE information added\n\n    "
    if config and context.call_function_name_qual in config['shell']:
        if len(context.call_args) > 0:
            sev = _evaluate_shell_call(context)
            if sev == bandit.LOW:
                return bandit.Issue(severity=bandit.LOW, confidence=bandit.HIGH, cwe=issue.Cwe.OS_COMMAND_INJECTION, text='Starting a process with a shell: Seems safe, but may be changed in the future, consider rewriting without shell')
            else:
                return bandit.Issue(severity=bandit.HIGH, confidence=bandit.HIGH, cwe=issue.Cwe.OS_COMMAND_INJECTION, text='Starting a process with a shell, possible injection detected, security issue.')

@test.takes_config('shell_injection')
@test.checks('Call')
@test.test_id('B606')
def start_process_with_no_shell(context, config):
    if False:
        print('Hello World!')
    "**B606: Test for starting a process with no shell**\n\n    Python possesses many mechanisms to invoke an external executable. However,\n    doing so may present a security issue if appropriate care is not taken to\n    sanitize any user provided or variable input.\n\n    This plugin test is part of a family of tests built to check for process\n    spawning and warn appropriately. Specifically, this test looks for the\n    spawning of a subprocess in a way that doesn't use a shell. Although this\n    is generally safe, it maybe useful for penetration testing workflows to\n    track where external system calls are used.  As such a LOW severity message\n    is generated.\n\n    See also:\n\n    - :doc:`../plugins/linux_commands_wildcard_injection`\n    - :doc:`../plugins/subprocess_without_shell_equals_true`\n    - :doc:`../plugins/start_process_with_a_shell`\n    - :doc:`../plugins/start_process_with_partial_path`\n    - :doc:`../plugins/subprocess_popen_with_shell_equals_true`\n\n    **Config Options:**\n\n    This plugin test shares a configuration with others in the same family,\n    namely `shell_injection`. This configuration is divided up into three\n    sections, `subprocess`, `shell` and `no_shell`. They each list Python calls\n    that spawn subprocesses, invoke commands within a shell, or invoke commands\n    without a shell (by replacing the calling process) respectively.\n\n    This plugin specifically scans for methods listed in `no_shell` section.\n\n    .. code-block:: yaml\n\n        shell_injection:\n            no_shell:\n                - os.execl\n                - os.execle\n                - os.execlp\n                - os.execlpe\n                - os.execv\n                - os.execve\n                - os.execvp\n                - os.execvpe\n                - os.spawnl\n                - os.spawnle\n                - os.spawnlp\n                - os.spawnlpe\n                - os.spawnv\n                - os.spawnve\n                - os.spawnvp\n                - os.spawnvpe\n                - os.startfile\n\n    :Example:\n\n    .. code-block:: none\n\n        >> Issue: [start_process_with_no_shell] Starting a process without a\n           shell.\n           Severity: Low   Confidence: Medium\n           CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)\n           Location: examples/os-spawn.py:8\n        7   os.spawnv(mode, path, args)\n        8   os.spawnve(mode, path, args, env)\n        9   os.spawnvp(mode, file, args)\n\n    .. seealso::\n\n     - https://security.openstack.org\n     - https://docs.python.org/3/library/os.html#os.system\n     - https://docs.python.org/3/library/subprocess.html#frequently-used-arguments\n     - https://security.openstack.org/guidelines/dg_use-subprocess-securely.html\n     - https://cwe.mitre.org/data/definitions/78.html\n\n    .. versionadded:: 0.10.0\n\n    .. versionchanged:: 1.7.3\n        CWE information added\n\n    "
    if config and context.call_function_name_qual in config['no_shell']:
        return bandit.Issue(severity=bandit.LOW, confidence=bandit.MEDIUM, cwe=issue.Cwe.OS_COMMAND_INJECTION, text='Starting a process without a shell.')

@test.takes_config('shell_injection')
@test.checks('Call')
@test.test_id('B607')
def start_process_with_partial_path(context, config):
    if False:
        i = 10
        return i + 15
    "**B607: Test for starting a process with a partial path**\n\n    Python possesses many mechanisms to invoke an external executable. If the\n    desired executable path is not fully qualified relative to the filesystem\n    root then this may present a potential security risk.\n\n    In POSIX environments, the `PATH` environment variable is used to specify a\n    set of standard locations that will be searched for the first matching\n    named executable. While convenient, this behavior may allow a malicious\n    actor to exert control over a system. If they are able to adjust the\n    contents of the `PATH` variable, or manipulate the file system, then a\n    bogus executable may be discovered in place of the desired one. This\n    executable will be invoked with the user privileges of the Python process\n    that spawned it, potentially a highly privileged user.\n\n    This test will scan the parameters of all configured Python methods,\n    looking for paths that do not start at the filesystem root, that is, do not\n    have a leading '/' character.\n\n    **Config Options:**\n\n    This plugin test shares a configuration with others in the same family,\n    namely `shell_injection`. This configuration is divided up into three\n    sections, `subprocess`, `shell` and `no_shell`. They each list Python calls\n    that spawn subprocesses, invoke commands within a shell, or invoke commands\n    without a shell (by replacing the calling process) respectively.\n\n    This test will scan parameters of all methods in all sections. Note that\n    methods are fully qualified and de-aliased prior to checking.\n\n    .. code-block:: yaml\n\n        shell_injection:\n            # Start a process using the subprocess module, or one of its\n            wrappers.\n            subprocess:\n                - subprocess.Popen\n                - subprocess.call\n\n            # Start a process with a function vulnerable to shell injection.\n            shell:\n                - os.system\n                - os.popen\n                - popen2.Popen3\n                - popen2.Popen4\n                - commands.getoutput\n                - commands.getstatusoutput\n            # Start a process with a function that is not vulnerable to shell\n            injection.\n            no_shell:\n                - os.execl\n                - os.execle\n\n\n    :Example:\n\n    .. code-block:: none\n\n        >> Issue: Starting a process with a partial executable path\n        Severity: Low   Confidence: High\n        CWE: CWE-78 (https://cwe.mitre.org/data/definitions/78.html)\n        Location: ./examples/partial_path_process.py:3\n        2    from subprocess import Popen as pop\n        3    pop('gcc --version', shell=False)\n\n    .. seealso::\n\n     - https://security.openstack.org\n     - https://docs.python.org/3/library/os.html#process-management\n     - https://cwe.mitre.org/data/definitions/78.html\n\n    .. versionadded:: 0.13.0\n\n    .. versionchanged:: 1.7.3\n        CWE information added\n\n    "
    if config and len(context.call_args):
        if context.call_function_name_qual in config['subprocess'] or context.call_function_name_qual in config['shell'] or context.call_function_name_qual in config['no_shell']:
            node = context.node.args[0]
            if isinstance(node, ast.List):
                node = node.elts[0]
            if isinstance(node, ast.Str) and (not full_path_match.match(node.s)):
                return bandit.Issue(severity=bandit.LOW, confidence=bandit.HIGH, cwe=issue.Cwe.OS_COMMAND_INJECTION, text='Starting a process with a partial executable path')