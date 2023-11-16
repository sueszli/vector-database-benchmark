from __future__ import annotations
import os
import pexpect
import sys
from ansible.module_utils.six import PY2
if PY2:
    log_buffer = sys.stdout
else:
    log_buffer = sys.stdout.buffer
env_vars = {'ANSIBLE_ROLES_PATH': './roles', 'ANSIBLE_NOCOLOR': 'True', 'ANSIBLE_RETRY_FILES_ENABLED': 'False'}

def run_test(playbook, test_spec, args=None, timeout=10, env=None):
    if False:
        print('Hello World!')
    if not env:
        env = os.environ.copy()
    env.update(env_vars)
    if not args:
        args = sys.argv[1:]
    vars_prompt_test = pexpect.spawn('ansible-playbook', args=[playbook] + args, timeout=timeout, env=env)
    vars_prompt_test.logfile = log_buffer
    for item in test_spec[0]:
        vars_prompt_test.expect(item[0])
        if item[1]:
            vars_prompt_test.send(item[1])
    vars_prompt_test.expect(test_spec[1])
    vars_prompt_test.expect(pexpect.EOF)
    vars_prompt_test.close()
tests = [{'playbook': 'vars_prompt-1.yml', 'test_spec': [[('input:', 'some input\r')], '"input": "some input"']}, {'playbook': 'vars_prompt-2.yml', 'test_spec': [[('Enter some input:', 'some more input\r')], '"input": "some more input"']}, {'playbook': 'vars_prompt-3.yml', 'test_spec': [[('input:', 'confirm me\r'), ('confirm input:', 'confirm me\r')], '"input": "confirm me"']}, {'playbook': 'vars_prompt-3.yml', 'test_spec': [[('input:', 'confirm me\r'), ('confirm input:', 'incorrect\r'), ('\\*\\*\\*\\*\\* VALUES ENTERED DO NOT MATCH \\*\\*\\*\\*', ''), ('input:', 'confirm me\r'), ('confirm input:', 'confirm me\r')], '"input": "confirm me"']}, {'playbook': 'vars_prompt-4.yml', 'test_spec': [[('not_secret', 'this is displayed\r'), ('this is displayed', '')], '"not_secret": "this is displayed"']}, {'playbook': 'vars_prompt-5.yml', 'test_spec': [[('password', 'Scenic-Improving-Payphone\r'), ('confirm password', 'Scenic-Improving-Payphone\r')], '"password": "\\$6\\$']}, {'playbook': 'vars_prompt-6.yml', 'test_spec': [[('prompt from variable:', 'input\r')], '']}, {'playbook': 'vars_prompt-7.yml', 'test_spec': [[('prompting for host:', 'testhost\r')], 'testhost.*ok=1']}, {'playbook': 'unsafe.yml', 'test_spec': [[('prompting for variable:', '{{whole}}\r')], 'testhost.*ok=2']}, {'playbook': 'unsupported.yml', 'test_spec': [[], "Invalid vars_prompt data structure, found unsupported key 'when'"]}]
for t in tests:
    run_test(playbook=t['playbook'], test_spec=t['test_spec'])