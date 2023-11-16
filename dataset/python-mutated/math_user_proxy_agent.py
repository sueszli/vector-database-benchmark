import re
import os
from pydantic import BaseModel, Extra, root_validator
from typing import Any, Callable, Dict, List, Optional, Union
from time import sleep
from flaml.autogen.agentchat import Agent, UserProxyAgent
from flaml.autogen.code_utils import UNKNOWN, extract_code, execute_code, infer_lang
from flaml.autogen.math_utils import get_answer
PROMPTS = {'default': "Let's use Python to solve a math problem.\n\nQuery requirements:\nYou should always use the 'print' function for the output and use fractions/radical forms instead of decimals.\nYou can use packages like sympy to help you.\nYou must follow the formats below to write your code:\n```python\n# your code\n```\n\nFirst state the key idea to solve the problem. You may choose from three ways to solve the problem:\nCase 1: If the problem can be solved with Python code directly, please write a program to solve it. You can enumerate all possible arrangements if needed.\nCase 2: If the problem is mostly reasoning, you can solve it by yourself directly.\nCase 3: If the problem cannot be handled in the above two ways, please follow this process:\n1. Solve the problem step by step (do not over-divide the steps).\n2. Take out any queries that can be asked through Python (for example, any calculations or equations that can be calculated).\n3. Wait for me to give the results.\n4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.\n\nAfter all the queries are run and you get the answer, put the answer in \\boxed{}.\n\nProblem:\n", 'two_tools': "Let's use two tools (Python and Wolfram alpha) to solve a math problem.\n\nQuery requirements:\nYou must follow the formats below to write your query:\nFor Wolfram Alpha:\n```wolfram\n# one wolfram query\n```\nFor Python:\n```python\n# your code\n```\nWhen using Python, you should always use the 'print' function for the output and use fractions/radical forms instead of decimals. You can use packages like sympy to help you.\nWhen using wolfram, give one query in each code block.\n\nPlease follow this process:\n1. Solve the problem step by step (do not over-divide the steps).\n2. Take out any queries that can be asked through Python or Wolfram Alpha, select the most suitable tool to be used (for example, any calculations or equations that can be calculated).\n3. Wait for me to give the results.\n4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.\n\nAfter all the queries are run and you get the answer, put the final answer in \\boxed{}.\n\nProblem: ", 'python': "Let's use Python to solve a math problem.\n\nQuery requirements:\nYou should always use the 'print' function for the output and use fractions/radical forms instead of decimals.\nYou can use packages like sympy to help you.\nYou must follow the formats below to write your code:\n```python\n# your code\n```\n\nPlease follow this process:\n1. Solve the problem step by step (do not over-divide the steps).\n2. Take out any queries that can be asked through Python (for example, any calculations or equations that can be calculated).\n3. Wait for me to give the results.\n4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.\n\nAfter all the queries are run and you get the answer, put the answer in \\boxed{}.\n\nProblem: "}

def _is_termination_msg_mathchat(message):
    if False:
        return 10
    'Check if a message is a termination message.'
    if isinstance(message, dict):
        message = message.get('content')
        if message is None:
            return False
    cb = extract_code(message)
    contain_code = False
    for c in cb:
        if c[0] == 'python' or c[0] == 'wolfram':
            contain_code = True
            break
    return not contain_code and get_answer(message) is not None and (get_answer(message) != '')

def _add_print_to_last_line(code):
    if False:
        i = 10
        return i + 15
    'Add print() to the last line of a string.'
    if 'print(' in code:
        return code
    lines = code.splitlines()
    last_line = lines[-1]
    if '\t' in last_line or '=' in last_line:
        return code
    if '=' in last_line:
        last_line = 'print(' + last_line.split(' = ')[0] + ')'
        lines.append(last_line)
    else:
        lines[-1] = 'print(' + last_line + ')'
    return '\n'.join(lines)

def _remove_print(code):
    if False:
        for i in range(10):
            print('nop')
    'remove all print statements from a string.'
    lines = code.splitlines()
    lines = [line for line in lines if not line.startswith('print(')]
    return '\n'.join(lines)

class MathUserProxyAgent(UserProxyAgent):
    """(Experimental) A MathChat agent that can handle math problems."""
    MAX_CONSECUTIVE_AUTO_REPLY = 15
    DEFAULT_REPLY = 'Continue. Please keep solving the problem until you need to query. (If you get to the answer, put it in \\boxed{}.)'

    def __init__(self, name: Optional[str]='MathChatAgent', is_termination_msg: Optional[Callable[[Dict], bool]]=_is_termination_msg_mathchat, human_input_mode: Optional[str]='NEVER', default_auto_reply: Optional[Union[str, Dict, None]]=DEFAULT_REPLY, max_invalid_q_per_step=3, **kwargs):
        if False:
            print('Hello World!')
        '\n        Args:\n            name (str): name of the agent\n            is_termination_msg (function): a function that takes a message in the form of a dictionary and returns a boolean value indicating if this received message is a termination message.\n                The dict can contain the following keys: "content", "role", "name", "function_call".\n            human_input_mode (str): whether to ask for human inputs every time a message is received.\n                Possible values are "ALWAYS", "TERMINATE", "NEVER".\n                (1) When "ALWAYS", the agent prompts for human input every time a message is received.\n                    Under this mode, the conversation stops when the human input is "exit",\n                    or when is_termination_msg is True and there is no human input.\n                (2) When "TERMINATE", the agent only prompts for human input only when a termination message is received or\n                    the number of auto reply reaches the max_consecutive_auto_reply.\n                (3) (Default) When "NEVER", the agent will never prompt for human input. Under this mode, the conversation stops\n                    when the number of auto reply reaches the max_consecutive_auto_reply or when is_termination_msg is True.\n            default_auto_reply (str or dict or None): the default auto reply message when no code execution or llm based reply is generated.\n            max_invalid_q_per_step (int): (ADDED) the maximum number of invalid queries per step.\n            **kwargs (dict): other kwargs in [UserProxyAgent](user_proxy_agent#__init__).\n        '
        super().__init__(name=name, is_termination_msg=is_termination_msg, human_input_mode=human_input_mode, default_auto_reply=default_auto_reply, **kwargs)
        self.register_reply([Agent, None], MathUserProxyAgent._generate_math_reply, 1)
        self._max_invalid_q_per_step = max_invalid_q_per_step
        self._valid_q_count = 0
        self._total_q_count = 0
        self._accum_invalid_q_per_step = 0
        self._previous_code = ''
        self.last_reply = None

    def generate_init_message(self, problem, prompt_type='default', customized_prompt=None):
        if False:
            while True:
                i = 10
        'Generate a prompt for the assitant agent with the given problem and prompt.\n\n        Args:\n            problem (str): the problem to be solved.\n            prompt_type (str): the type of the prompt. Possible values are "default", "python", "wolfram".\n                (1) "default": the prompt that allows the agent to choose between 3 ways to solve a problem:\n                    1. write a python program to solve it directly.\n                    2. solve it directly without python.\n                    3. solve it step by step with python.\n                (2) "python":\n                    a simplified prompt from the third way of the "default" prompt, that asks the assistant\n                    to solve the problem step by step with python.\n                (3) "two_tools":\n                    a simplified prompt similar to the "python" prompt, but allows the model to choose between\n                    Python and Wolfram Alpha to solve the problem.\n            customized_prompt (str): a customized prompt to be used. If it is not None, the prompt_type will be ignored.\n\n        Returns:\n            str: the generated prompt ready to be sent to the assistant agent.\n        '
        self._reset()
        if customized_prompt is not None:
            return customized_prompt + problem
        return PROMPTS[prompt_type] + problem

    def _reset(self):
        if False:
            while True:
                i = 10
        self._valid_q_count = 0
        self._total_q_count = 0
        self._accum_invalid_q_per_step = 0
        self._previous_code = ''
        self.last_reply = None

    def execute_one_python_code(self, pycode):
        if False:
            return 10
        'Execute python code blocks.\n\n        Previous python code will be saved and executed together with the new code.\n        the "print" function will also be added to the last line of the code if needed\n        '
        pycode = pycode.replace('; ', '\n').replace(';', '\n')
        pycode = self._previous_code + _add_print_to_last_line(pycode)
        (return_code, output, _) = execute_code(pycode, **self._code_execution_config, timeout=5)
        is_success = return_code == 0
        if not is_success:
            pattern = 'File "/[^"]+\\.py", line \\d+, in .+\\n'
            if isinstance(output, str):
                output = re.sub(pattern, '', output)
            output = 'Error: ' + output
        elif output == '':
            if 'print' not in pycode:
                output = 'No output found. Make sure you print the results.'
                is_success = False
            else:
                output = 'No output found.'
                is_success = True
        if len(output) > 2000:
            output = 'Your requested query response is too long. You might have made a mistake. Please revise your reasoning and query.'
            is_success = False
        if is_success:
            tmp = self._previous_code + '\n' + _remove_print(pycode) + '\n'
            (rcode, _, _) = execute_code(tmp, **self._code_execution_config)
        else:
            tmp = self._previous_code + '\n'
            for line in pycode.split('\n'):
                if 'import' in line:
                    tmp += line + '\n'
            (rcode, _, _) = execute_code(tmp, **self._code_execution_config)
        if rcode == 0:
            self._previous_code = tmp
        return (output, is_success)

    def execute_one_wolfram_query(self, query: str):
        if False:
            return 10
        'Run one wolfram query and return the output.\n\n        Args:\n            query: string of the query.\n\n        Returns:\n            output: string with the output of the query.\n            is_success: boolean indicating whether the query was successful.\n        '
        wolfram = WolframAlphaAPIWrapper()
        (output, is_success) = wolfram.run(query)
        if output == '':
            output = 'Error: The wolfram query is invalid.'
            is_success = False
        return (output, is_success)

    def _generate_math_reply(self, messages: Optional[List[Dict]]=None, sender: Optional[Agent]=None, config: Optional[Any]=None):
        if False:
            return 10
        'Generate an auto reply.'
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        message = message.get('content', '')
        code_blocks = extract_code(message)
        if len(code_blocks) == 1 and code_blocks[0][0] == UNKNOWN:
            return (True, self._default_auto_reply)
        (is_success, all_success) = (True, True)
        reply = ''
        for code_block in code_blocks:
            (lang, code) = code_block
            if not lang:
                lang = infer_lang(code)
            if lang == 'python':
                (output, is_success) = self.execute_one_python_code(code)
            elif lang == 'wolfram':
                (output, is_success) = self.execute_one_wolfram_query(code)
            else:
                output = 'Error: Unknown language.'
                is_success = False
            reply += output + '\n'
            if not is_success:
                all_success = False
                self._valid_q_count -= 1
        reply = reply.strip()
        if self.last_reply == reply:
            return (True, reply + '\nYour query or result is same from the last, please try a new approach.')
        self.last_reply = reply
        if not all_success:
            self._accum_invalid_q_per_step += 1
            if self._accum_invalid_q_per_step > self._max_invalid_q_per_step:
                self._accum_invalid_q_per_step = 0
                reply = 'Please revisit the problem statement and your reasoning. If you think this step is correct, solve it yourself and continue the next step. Otherwise, correct this step.'
        return (True, reply)

def get_from_dict_or_env(data: Dict[str, Any], key: str, env_key: str, default: Optional[str]=None) -> str:
    if False:
        while True:
            i = 10
    'Get a value from a dictionary or an environment variable.'
    if key in data and data[key]:
        return data[key]
    elif env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    elif default is not None:
        return default
    else:
        raise ValueError(f'Did not find {key}, please add an environment variable `{env_key}` which contains it, or pass  `{key}` as a named parameter.')

class WolframAlphaAPIWrapper(BaseModel):
    """Wrapper for Wolfram Alpha.

    Docs for using:

    1. Go to wolfram alpha and sign up for a developer account
    2. Create an app and get your APP ID
    3. Save your APP ID into WOLFRAM_ALPHA_APPID env variable
    4. pip install wolframalpha

    """
    wolfram_client: Any
    wolfram_alpha_appid: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    @root_validator(skip_on_failure=True)
    def validate_environment(cls, values: Dict) -> Dict:
        if False:
            while True:
                i = 10
        'Validate that api key and python package exists in environment.'
        wolfram_alpha_appid = get_from_dict_or_env(values, 'wolfram_alpha_appid', 'WOLFRAM_ALPHA_APPID')
        values['wolfram_alpha_appid'] = wolfram_alpha_appid
        try:
            import wolframalpha
        except ImportError:
            raise ImportError('wolframalpha is not installed. Please install it with `pip install wolframalpha`')
        client = wolframalpha.Client(wolfram_alpha_appid)
        values['wolfram_client'] = client
        return values

    def run(self, query: str) -> str:
        if False:
            i = 10
            return i + 15
        'Run query through WolframAlpha and parse result.'
        from urllib.error import HTTPError
        is_success = False
        res = None
        for _ in range(20):
            try:
                res = self.wolfram_client.query(query)
                break
            except HTTPError:
                sleep(1)
            except Exception:
                return ("Wolfram Alpha wasn't able to answer it. Please try a new query for wolfram or use python.", is_success)
        if res is None:
            return ("Wolfram Alpha wasn't able to answer it (may due to web error), you can try again or use python.", is_success)
        try:
            if not res['@success']:
                return ('Your Wolfram query is invalid. Please try a new query for wolfram or use python.', is_success)
            assumption = next(res.pods).text
            answer = ''
            for result in res['pod']:
                if result['@title'] == 'Solution':
                    answer = result['subpod']['plaintext']
                if result['@title'] == 'Results' or result['@title'] == 'Solutions':
                    for (i, sub) in enumerate(result['subpod']):
                        answer += f'ans {i}: ' + sub['plaintext'] + '\n'
                    break
            if answer == '':
                answer = next(res.results).text
        except Exception:
            return ("Wolfram Alpha wasn't able to answer it. Please try a new query for wolfram or use python.", is_success)
        if answer is None or answer == '':
            return ('No good Wolfram Alpha Result was found', is_success)
        is_success = True
        return (f'Assumption: {assumption} \nAnswer: {answer}', is_success)