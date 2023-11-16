from __future__ import annotations
DOCUMENTATION = '\n---\nmodule: expect\nversion_added: \'2.0\'\nshort_description: Executes a command and responds to prompts\ndescription:\n     - The M(ansible.builtin.expect) module executes a command and responds to prompts.\n     - The given command will be executed on all selected nodes. It will not be\n       processed through the shell, so variables like C($HOME) and operations\n       like C("<"), C(">"), C("|"), and C("&") will not work.\noptions:\n  command:\n    description:\n      - The command module takes command to run.\n    required: true\n    type: str\n  creates:\n    type: path\n    description:\n      - A filename, when it already exists, this step will B(not) be run.\n  removes:\n    type: path\n    description:\n      - A filename, when it does not exist, this step will B(not) be run.\n  chdir:\n    type: path\n    description:\n      - Change into this directory before running the command.\n  responses:\n    type: dict\n    description:\n      - Mapping of expected string/regex and string to respond with. If the\n        response is a list, successive matches return successive\n        responses. List functionality is new in 2.1.\n    required: true\n  timeout:\n    type: int\n    description:\n      - Amount of time in seconds to wait for the expected strings. Use\n        V(null) to disable timeout.\n    default: 30\n  echo:\n    description:\n      - Whether or not to echo out your response strings.\n    default: false\n    type: bool\nrequirements:\n  - python >= 2.6\n  - pexpect >= 3.3\nextends_documentation_fragment: action_common_attributes\nattributes:\n    check_mode:\n        support: none\n    diff_mode:\n        support: none\n    platform:\n        support: full\n        platforms: posix\nnotes:\n  - If you want to run a command through the shell (say you are using C(<),\n    C(>), C(|), and so on), you must specify a shell in the command such as\n    C(/bin/bash -c "/path/to/something | grep else").\n  - The question, or key, under O(responses) is a python regex match. Case\n    insensitive searches are indicated with a prefix of C(?i).\n  - The C(pexpect) library used by this module operates with a search window\n    of 2000 bytes, and does not use a multiline regex match. To perform a\n    start of line bound match, use a pattern like ``(?m)^pattern``\n  - By default, if a question is encountered multiple times, its string\n    response will be repeated. If you need different responses for successive\n    question matches, instead of a string response, use a list of strings as\n    the response. The list functionality is new in 2.1.\n  - The M(ansible.builtin.expect) module is designed for simple scenarios.\n    For more complex needs, consider the use of expect code with the M(ansible.builtin.shell)\n    or M(ansible.builtin.script) modules. (An example is part of the M(ansible.builtin.shell) module documentation).\n  - If the command returns non UTF-8 data, it must be encoded to avoid issues. One option is to pipe\n    the output through C(base64).\nseealso:\n- module: ansible.builtin.script\n- module: ansible.builtin.shell\nauthor: "Matt Martz (@sivel)"\n'
EXAMPLES = '\n- name: Case insensitive password string match\n  ansible.builtin.expect:\n    command: passwd username\n    responses:\n      (?i)password: "MySekretPa$$word"\n  # you don\'t want to show passwords in your logs\n  no_log: true\n\n- name: Generic question with multiple different responses\n  ansible.builtin.expect:\n    command: /path/to/custom/command\n    responses:\n      Question:\n        - response1\n        - response2\n        - response3\n\n- name: Multiple questions with responses\n  ansible.builtin.expect:\n    command: /path/to/custom/command\n    responses:\n        "Please provide your name":\n            - "Anna"\n        "Database user":\n            - "{{ db_username }}"\n        "Database password":\n            - "{{ db_password }}"\n'
import datetime
import os
import traceback
PEXPECT_IMP_ERR = None
try:
    import pexpect
    HAS_PEXPECT = True
except ImportError:
    PEXPECT_IMP_ERR = traceback.format_exc()
    HAS_PEXPECT = False
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native

def response_closure(module, question, responses):
    if False:
        for i in range(10):
            print('nop')
    resp_gen = (b'%s\n' % to_bytes(r).rstrip(b'\n') for r in responses)

    def wrapped(info):
        if False:
            while True:
                i = 10
        try:
            return next(resp_gen)
        except StopIteration:
            module.fail_json(msg="No remaining responses for '%s', output was '%s'" % (question, info['child_result_list'][-1]))
    return wrapped

def main():
    if False:
        while True:
            i = 10
    module = AnsibleModule(argument_spec=dict(command=dict(required=True), chdir=dict(type='path'), creates=dict(type='path'), removes=dict(type='path'), responses=dict(type='dict', required=True), timeout=dict(type='int', default=30), echo=dict(type='bool', default=False)))
    if not HAS_PEXPECT:
        module.fail_json(msg=missing_required_lib('pexpect'), exception=PEXPECT_IMP_ERR)
    chdir = module.params['chdir']
    args = module.params['command']
    creates = module.params['creates']
    removes = module.params['removes']
    responses = module.params['responses']
    timeout = module.params['timeout']
    echo = module.params['echo']
    events = dict()
    for (key, value) in responses.items():
        if isinstance(value, list):
            response = response_closure(module, key, value)
        else:
            response = b'%s\n' % to_bytes(value).rstrip(b'\n')
        events[to_bytes(key)] = response
    if args.strip() == '':
        module.fail_json(rc=256, msg='no command given')
    if chdir:
        chdir = os.path.abspath(chdir)
        os.chdir(chdir)
    if creates:
        if os.path.exists(creates):
            module.exit_json(cmd=args, stdout='skipped, since %s exists' % creates, changed=False, rc=0)
    if removes:
        if not os.path.exists(removes):
            module.exit_json(cmd=args, stdout='skipped, since %s does not exist' % removes, changed=False, rc=0)
    startd = datetime.datetime.now()
    try:
        try:
            (b_out, rc) = pexpect.run(args, timeout=timeout, withexitstatus=True, events=events, cwd=chdir, echo=echo, encoding=None)
        except TypeError:
            (b_out, rc) = pexpect._run(args, timeout=timeout, withexitstatus=True, events=events, extra_args=None, logfile=None, cwd=chdir, env=None, _spawn=pexpect.spawn, echo=echo)
    except (TypeError, AttributeError) as e:
        module.fail_json(msg='Insufficient version of pexpect installed (%s), this module requires pexpect>=3.3. Error was %s' % (pexpect.__version__, to_native(e)))
    except pexpect.ExceptionPexpect as e:
        module.fail_json(msg='%s' % to_native(e), exception=traceback.format_exc())
    endd = datetime.datetime.now()
    delta = endd - startd
    if b_out is None:
        b_out = b''
    result = dict(cmd=args, stdout=to_native(b_out).rstrip('\r\n'), rc=rc, start=str(startd), end=str(endd), delta=str(delta), changed=True)
    if rc is None:
        module.fail_json(msg='command exceeded timeout', **result)
    elif rc != 0:
        module.fail_json(msg='non-zero return code', **result)
    module.exit_json(**result)
if __name__ == '__main__':
    main()