from __future__ import annotations
import json
import sys
import pytest
import subprocess
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils import basic

def test_run_non_existent_command(monkeypatch):
    if False:
        while True:
            i = 10
    ' Test that `command` returns std{out,err} even if the executable is not found '

    def fail_json(msg, **kwargs):
        if False:
            print('Hello World!')
        assert kwargs['stderr'] == b''
        assert kwargs['stdout'] == b''
        sys.exit(1)

    def popen(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        raise OSError()
    monkeypatch.setattr(basic, '_ANSIBLE_ARGS', to_bytes(json.dumps({'ANSIBLE_MODULE_ARGS': {}})))
    monkeypatch.setattr(subprocess, 'Popen', popen)
    am = basic.AnsibleModule(argument_spec={})
    monkeypatch.setattr(am, 'fail_json', fail_json)
    with pytest.raises(SystemExit):
        am.run_command('lecho', 'whatever')