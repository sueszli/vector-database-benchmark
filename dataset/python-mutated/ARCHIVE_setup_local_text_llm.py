import copy
import html
import inquirer
import ooba
from ..utils.display_markdown_message import display_markdown_message

def setup_local_text_llm(interpreter):
    if False:
        while True:
            i = 10
    '\n    Takes an Interpreter (which includes a ton of LLM settings),\n    returns a text LLM (an OpenAI-compatible chat LLM with baked-in settings. Only takes `messages`).\n    '
    repo_id = interpreter.model.replace('huggingface/', '')
    display_markdown_message(f'> **Warning**: Local LLM usage is an experimental, unstable feature.')
    if repo_id != 'TheBloke/Mistral-7B-Instruct-v0.1-GGUF':
        display_markdown_message(f'**Open Interpreter** will use `{repo_id}` for local execution.')
    if 'gguf' in repo_id.lower() and interpreter.gguf_quality == None:
        gguf_quality_choices = {'Extra Small': 0.0, 'Small': 0.25, 'Medium': 0.5, 'Large': 0.75, 'Extra Large': 1.0}
        questions = [inquirer.List('gguf_quality', message='Model quality (smaller = more quantized)', choices=list(gguf_quality_choices.keys()))]
        answers = inquirer.prompt(questions)
        interpreter.gguf_quality = gguf_quality_choices[answers['gguf_quality']]
    path = ooba.download(f'https://huggingface.co/{repo_id}')
    ooba_llm = ooba.llm(path, verbose=interpreter.debug_mode)
    print('\nReady.\n')

    def local_text_llm(messages):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a generator. Makes ooba fully openai compatible\n        '
        '\n        system_message = messages[0]["content"]\n        messages = messages[1:]\n\n        if interpreter.context_window:\n            context_window = interpreter.context_window\n        else:\n            context_window = DEFAULT_CONTEXT_WINDOW\n\n        if interpreter.max_tokens:\n            max_tokens = interpreter.max_tokens\n        else:\n            max_tokens = DEFAULT_MAX_TOKENS\n        \n        messages = tt.trim(\n            messages,\n            max_tokens=(context_window-max_tokens-25),\n            system_message=system_message\n        )\n\n        prompt = messages_to_prompt(messages, interpreter.model)\n        '
        if 'mistral' in repo_id.lower():
            messages[0]['content'] = 'You are Open Interpreter. You almost always run code to complete user requests. Outside code, use markdown.'
            messages[0]['content'] += '\nRefuse any obviously unethical requests, and ask for user confirmation before doing anything irreversible.'
        messages = copy.deepcopy(messages)
        messages[0]['content'] += "\nTo execute code on the user's machine, write a markdown code block *with the language*, i.e:\n\n```python\nprint('Hi!')\n```\nYou will recieve the output ('Hi!'). Use any language."
        if interpreter.debug_mode:
            print('Messages going to ooba:', messages)
        buffer = ''
        for token in ooba_llm.chat(messages):
            buffer += token
            while '&' in buffer and ';' in buffer or (buffer.count('&') == 1 and ';' not in buffer):
                start_idx = buffer.find('&')
                end_idx = buffer.find(';', start_idx)
                if start_idx == -1 or end_idx == -1:
                    break
                for char in buffer[:start_idx]:
                    yield make_chunk(char)
                entity = buffer[start_idx:end_idx + 1]
                yield make_chunk(html.unescape(entity))
                buffer = buffer[end_idx + 1:]
            if '&' not in buffer:
                for char in buffer:
                    yield make_chunk(char)
                buffer = ''
        for char in buffer:
            yield make_chunk(char)
    return local_text_llm

def make_chunk(token):
    if False:
        for i in range(10):
            print('nop')
    return {'choices': [{'delta': {'content': token}}]}