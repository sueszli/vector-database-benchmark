from __future__ import annotations
from ansible.module_utils import basic
from ansible.module_utils.basic import AnsibleModule
heuristic_log_sanitize = basic.heuristic_log_sanitize

def heuristic_log_sanitize_spy(*args, **kwargs):
    if False:
        print('Hello World!')
    heuristic_log_sanitize_spy.return_value = heuristic_log_sanitize(*args, **kwargs)
    return heuristic_log_sanitize_spy.return_value
basic.heuristic_log_sanitize = heuristic_log_sanitize_spy

def main():
    if False:
        i = 10
        return i + 15
    module = AnsibleModule(argument_spec={'data': {'type': 'str', 'required': True}})
    data = module.params['data']
    left = data.rindex(':') + 1
    right = data.rindex('@')
    expected = data[:left] + '********' + data[right:]
    sanitized = heuristic_log_sanitize_spy.return_value
    if sanitized != expected:
        module.fail_json(msg='Invalid match', expected=expected, sanitized=sanitized)
    module.exit_json(match=True)
if __name__ == '__main__':
    main()