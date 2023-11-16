import os
import traceback
import litellm
import openai
import tokentrim as tt
from ..utils.display_markdown_message import display_markdown_message

def setup_text_llm(interpreter):
    if False:
        print('Hello World!')
    '\n    Takes an Interpreter (which includes a ton of LLM settings),\n    returns a text LLM (an OpenAI-compatible chat LLM with baked-in settings. Only takes `messages`).\n    '

    def base_llm(messages):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a generator\n        '
        system_message = messages[0]['content']
        messages = messages[1:]
        try:
            if interpreter.context_window and interpreter.max_tokens:
                trim_to_be_this_many_tokens = interpreter.context_window - interpreter.max_tokens - 25
                messages = tt.trim(messages, system_message=system_message, max_tokens=trim_to_be_this_many_tokens)
            elif interpreter.context_window and (not interpreter.max_tokens):
                messages = tt.trim(messages, system_message=system_message, max_tokens=interpreter.context_window)
            else:
                try:
                    messages = tt.trim(messages, system_message=system_message, model=interpreter.model)
                except:
                    if len(messages) == 1:
                        display_markdown_message('\n                        **We were unable to determine the context window of this model.** Defaulting to 3000.\n                        If your model can handle more, run `interpreter --context_window {token limit}` or `interpreter.context_window = {token limit}`.\n                        Also, please set max_tokens: `interpreter --max_tokens {max tokens per response}` or `interpreter.max_tokens = {max tokens per response}`\n                        ')
                    messages = tt.trim(messages, system_message=system_message, max_tokens=3000)
        except TypeError as e:
            if interpreter.vision and str(e) == 'expected string or buffer':
                if interpreter.debug_mode:
                    print("Couldn't token trim image messages. Error:", e)
                messages = [{'role': 'system', 'content': system_message}] + messages
            else:
                raise
        if interpreter.debug_mode:
            print('Passing messages into LLM:', messages)
        params = {'model': interpreter.model, 'messages': messages, 'stream': True}
        if interpreter.api_base:
            params['api_base'] = interpreter.api_base
        if interpreter.api_key:
            params['api_key'] = interpreter.api_key
        if interpreter.max_tokens:
            params['max_tokens'] = interpreter.max_tokens
        if interpreter.temperature is not None:
            params['temperature'] = interpreter.temperature
        else:
            params['temperature'] = 0.0
        if interpreter.model == 'gpt-4-vision-preview':
            return openai.ChatCompletion.create(**params)
        if interpreter.max_budget:
            litellm.max_budget = interpreter.max_budget
        if interpreter.debug_mode:
            litellm.set_verbose = True
        if interpreter.debug_mode:
            print('Sending this to LiteLLM:', params)
        return litellm.completion(**params)
    return base_llm