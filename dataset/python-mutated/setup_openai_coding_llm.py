import litellm
import tokentrim as tt
from ..utils.convert_to_openai_messages import convert_to_openai_messages
from ..utils.display_markdown_message import display_markdown_message
from ..utils.merge_deltas import merge_deltas
from ..utils.parse_partial_json import parse_partial_json
function_schema = {'name': 'execute', 'description': "Executes code on the user's machine, **in the users local environment**, and returns the output", 'parameters': {'type': 'object', 'properties': {'language': {'type': 'string', 'description': 'The programming language (required parameter to the `execute` function)', 'enum': ['python', 'R', 'shell', 'applescript', 'javascript', 'html', 'powershell']}, 'code': {'type': 'string', 'description': 'The code to execute (required)'}}, 'required': ['language', 'code']}}

def setup_openai_coding_llm(interpreter):
    if False:
        i = 10
        return i + 15
    '\n    Takes an Interpreter (which includes a ton of LLM settings),\n    returns a OI Coding LLM (a generator that takes OI messages and streams deltas with `message`, `language`, and `code`).\n    '

    def coding_llm(messages):
        if False:
            for i in range(10):
                print('nop')
        messages = convert_to_openai_messages(messages, function_calling=True)
        messages[0]['content'] += '\n\nOnly use the function you have been provided with.'
        system_message = messages[0]['content']
        messages = messages[1:]
        try:
            messages = tt.trim(messages=messages, system_message=system_message, model=interpreter.model)
        except:
            if interpreter.context_window:
                messages = tt.trim(messages=messages, system_message=system_message, max_tokens=interpreter.context_window)
            else:
                if len(messages) == 1:
                    display_markdown_message('\n                    **We were unable to determine the context window of this model.** Defaulting to 3000.\n                    If your model can handle more, run `interpreter --context_window {token limit}` or `interpreter.context_window = {token limit}`.\n                    ')
                messages = tt.trim(messages=messages, system_message=system_message, max_tokens=3000)
        if interpreter.debug_mode:
            print('Sending this to the OpenAI LLM:', messages)
        params = {'model': interpreter.model, 'messages': messages, 'stream': True, 'functions': [function_schema]}
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
        if interpreter.max_budget:
            litellm.max_budget = interpreter.max_budget
        if interpreter.debug_mode:
            litellm.set_verbose = True
        if interpreter.debug_mode:
            print('Sending this to LiteLLM:', params)
        response = litellm.completion(**params)
        accumulated_deltas = {}
        language = None
        code = ''
        for chunk in response:
            if interpreter.debug_mode:
                print('Chunk from LLM', chunk)
            if 'choices' not in chunk or len(chunk['choices']) == 0:
                continue
            delta = chunk['choices'][0]['delta']
            accumulated_deltas = merge_deltas(accumulated_deltas, delta)
            if interpreter.debug_mode:
                print('Accumulated deltas', accumulated_deltas)
            if 'content' in delta and delta['content']:
                yield {'message': delta['content']}
            if 'function_call' in accumulated_deltas and 'arguments' in accumulated_deltas['function_call']:
                if 'name' in accumulated_deltas['function_call'] and accumulated_deltas['function_call']['name'] == 'execute':
                    arguments = accumulated_deltas['function_call']['arguments']
                    arguments = parse_partial_json(arguments)
                    if arguments:
                        if language is None and 'language' in arguments and ('code' in arguments) and arguments['language']:
                            language = arguments['language']
                            yield {'language': language}
                        if language is not None and 'code' in arguments:
                            code_delta = arguments['code'][len(code):]
                            code = arguments['code']
                            if code_delta:
                                yield {'code': code_delta}
                    elif interpreter.debug_mode:
                        print('Arguments not a dict.')
                elif 'name' in accumulated_deltas['function_call'] and accumulated_deltas['function_call']['name'] == 'python':
                    if interpreter.debug_mode:
                        print('Got direct python call')
                    if language is None:
                        language = 'python'
                        yield {'language': language}
                    if language is not None:
                        code_delta = accumulated_deltas['function_call']['arguments'][len(code):]
                        code = accumulated_deltas['function_call']['arguments']
                        if code_delta:
                            yield {'code': code_delta}
                elif 'name' in accumulated_deltas['function_call']:
                    print('Encountered an unexpected function call: ', accumulated_deltas['function_call'], '\nPlease open an issue and provide the above info at: https://github.com/KillianLucas/open-interpreter')
    return coding_llm