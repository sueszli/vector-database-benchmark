import logging
import json
import os
import os.path
import sys
FALSISH = ('No', 'NO', 'no', 'N', 'n', '0')
TRUISH = ('Yes', 'YES', 'yes', 'Y', 'y', '1')
DEFAULTISH = ('Default', 'DEFAULT', 'default', 'D', 'd', '')

def yes(msg=None, false_msg=None, true_msg=None, default_msg=None, retry_msg=None, invalid_msg=None, env_msg='{} (from {})', falsish=FALSISH, truish=TRUISH, defaultish=DEFAULTISH, default=False, retry=True, env_var_override=None, ofile=None, input=input, prompt=True, msgid=None):
    if False:
        for i in range(10):
            print('nop')
    "Output <msg> (usually a question) and let user input an answer.\n    Qualifies the answer according to falsish, truish and defaultish as True, False or <default>.\n    If it didn't qualify and retry is False (no retries wanted), return the default [which\n    defaults to False]. If retry is True let user retry answering until answer is qualified.\n\n    If env_var_override is given and this var is present in the environment, do not ask\n    the user, but just use the env var contents as answer as if it was typed in.\n    Otherwise read input from stdin and proceed as normal.\n    If EOF is received instead an input or an invalid input without retry possibility,\n    return default.\n\n    :param msg: introducing message to output on ofile, no \n is added [None]\n    :param retry_msg: retry message to output on ofile, no \n is added [None]\n    :param false_msg: message to output before returning False [None]\n    :param true_msg: message to output before returning True [None]\n    :param default_msg: message to output before returning a <default> [None]\n    :param invalid_msg: message to output after a invalid answer was given [None]\n    :param env_msg: message to output when using input from env_var_override ['{} (from {})'],\n           needs to have 2 placeholders for answer and env var name\n    :param falsish: sequence of answers qualifying as False\n    :param truish: sequence of answers qualifying as True\n    :param defaultish: sequence of answers qualifying as <default>\n    :param default: default return value (defaultish answer was given or no-answer condition) [False]\n    :param retry: if True and input is incorrect, retry. Otherwise return default. [True]\n    :param env_var_override: environment variable name [None]\n    :param ofile: output stream [sys.stderr]\n    :param input: input function [input from builtins]\n    :return: boolean answer value, True or False\n    "

    def output(msg, msg_type, is_prompt=False, **kwargs):
        if False:
            while True:
                i = 10
        json_output = getattr(logging.getLogger('borg'), 'json', False)
        if json_output:
            kwargs.update(dict(type='question_%s' % msg_type, msgid=msgid, message=msg))
            print(json.dumps(kwargs), file=sys.stderr)
        elif is_prompt:
            print(msg, file=ofile, end='', flush=True)
        else:
            print(msg, file=ofile)
    msgid = msgid or env_var_override
    if ofile is None:
        ofile = sys.stderr
    if default not in (True, False):
        raise ValueError('invalid default value, must be True or False')
    if msg:
        output(msg, 'prompt', is_prompt=True)
    while True:
        answer = None
        if env_var_override:
            answer = os.environ.get(env_var_override)
            if answer is not None and env_msg:
                output(env_msg.format(answer, env_var_override), 'env_answer', env_var=env_var_override)
        if answer is None:
            if not prompt:
                return default
            try:
                answer = input()
            except EOFError:
                answer = truish[0] if default else falsish[0]
        if answer in defaultish:
            if default_msg:
                output(default_msg, 'accepted_default')
            return default
        if answer in truish:
            if true_msg:
                output(true_msg, 'accepted_true')
            return True
        if answer in falsish:
            if false_msg:
                output(false_msg, 'accepted_false')
            return False
        if invalid_msg:
            output(invalid_msg, 'invalid_answer')
        if not retry:
            return default
        if retry_msg:
            output(retry_msg, 'prompt_retry', is_prompt=True)
        env_var_override = None