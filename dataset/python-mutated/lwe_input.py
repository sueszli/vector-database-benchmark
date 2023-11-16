from __future__ import absolute_import, division, print_function
__metaclass__ = type
import datetime
import time
from ansible.errors import AnsibleError, AnsiblePromptInterrupt, AnsiblePromptNoninteractive
from ansible.module_utils.common.text.converters import to_text
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
from lwe.core.editor import pipe_editor
DOCUMENTATION = "\n  module: lwe_input\n  short_description: Pauses execution until input is received\n  description:\n    - This module pauses the execution of a playbook until the user provides input.\n    - The user can provide input through the command line or by opening an editor.\n  options:\n    echo:\n      description:\n        - If set to True, the user's input will be displayed on the screen.\n        - If set to False, the user's input will be hidden.\n      type: bool\n      default: True\n    prompt:\n      description:\n        - The custom prompt message to display before waiting for user input.\n      type: str\n  author:\n    - Chad Phillips (@thehunmonkgroup)\n"
EXAMPLES = '\n  - name: Pause execution and wait for user input\n    lwe_input:\n\n  - name: Pause execution and wait for user input with custom prompt\n    lwe_input:\n      prompt: "Please enter your name"\n\n  - name: Pause execution and wait for user input with hidden output\n    lwe_input:\n      echo: False\n'
RETURN = '\n  stdout:\n    description: Standard output of the task, showing the duration of the pause.\n    type: str\n    returned: always\n  stop:\n    description: The end time of the pause.\n    type: str\n    returned: always\n  delta:\n    description: The duration of the pause in seconds.\n    type: int\n    returned: always\n  user_input:\n    description: The input provided by the user.\n    type: str\n    returned: always\n'
display = Display()

class ActionModule(ActionBase):
    """pauses execution until input is received"""
    BYPASS_HOST_LOOP = True

    def run(self, tmp=None, task_vars=None):
        if False:
            i = 10
            return i + 15
        'run the lwe_input action module'
        if task_vars is None:
            task_vars = dict()
        result = super(ActionModule, self).run(tmp, task_vars)
        del tmp
        (validation_result, new_module_args) = self.validate_argument_spec(argument_spec={'echo': {'type': 'bool', 'default': True}, 'prompt': {'type': 'str'}})
        prompt = None
        echo = new_module_args['echo']
        echo_prompt = ''
        result.update(dict(changed=False, rc=0, stderr='', stdout='', start=None, stop=None, delta=None, echo=echo))
        editor_blurb = "Enter 'e' to open an editor"
        if not echo:
            echo_prompt = ' (output is hidden)'
        if new_module_args['prompt']:
            prompt = '\n[%s]\n%s\n\n%s%s:' % (self._task.get_name().strip(), editor_blurb, new_module_args['prompt'], echo_prompt)
        else:
            prompt = '\n[%s]\n%s\n\n%s%s:' % (self._task.get_name().strip(), editor_blurb, 'Press enter to continue, Ctrl+C to interrupt', echo_prompt)
        start = time.time()
        result['start'] = to_text(datetime.datetime.now())
        result['user_input'] = b''
        default_input_complete = None
        user_input = b''
        try:
            _user_input = display.prompt_until(prompt, private=not echo, complete_input=default_input_complete)
        except AnsiblePromptInterrupt:
            user_input = None
        except AnsiblePromptNoninteractive:
            display.warning('Not waiting for response to prompt as stdin is not interactive')
        else:
            user_input = _user_input
        if user_input is None:
            prompt = "Press 'C' to continue the play or 'A' to abort \r"
            try:
                user_input = display.prompt_until(prompt, private=not echo, interrupt_input=(b'a',), complete_input=(b'c',))
            except AnsiblePromptInterrupt as err:
                raise AnsibleError('user requested abort!') from err
        elif user_input.strip() == b'e':
            display.display('Editor requested')
            user_input = pipe_editor('', suffix='md')
        duration = time.time() - start
        result['stop'] = to_text(datetime.datetime.now())
        result['delta'] = int(duration)
        duration = round(duration, 2)
        result['stdout'] = 'Paused for %s seconds' % duration
        result['user_input'] = to_text(user_input, errors='surrogate_or_strict')
        return result