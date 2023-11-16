import time
import traceback
import litellm
from ..code_interpreters.create_code_interpreter import create_code_interpreter
from ..code_interpreters.language_map import language_map
from ..utils.display_markdown_message import display_markdown_message
from ..utils.html_to_base64 import html_to_base64
from ..utils.merge_deltas import merge_deltas
from ..utils.truncate_output import truncate_output

def respond(interpreter):
    if False:
        i = 10
        return i + 15
    '\n    Yields tokens, but also adds them to interpreter.messages. TBH probably would be good to seperate those two responsibilities someday soon\n    Responds until it decides not to run any more code or say anything else.\n    '
    last_unsupported_code = ''
    while True:
        system_message = interpreter.generate_system_message()
        system_message = {'role': 'system', 'message': system_message}
        messages_for_llm = interpreter.messages.copy()
        messages_for_llm = [system_message] + messages_for_llm
        for message in messages_for_llm:
            if 'output' in message and message['output'] == '':
                message['output'] = 'No output'
        interpreter.messages.append({'role': 'assistant'})
        try:
            chunk_type = None
            for chunk in interpreter._llm(messages_for_llm):
                interpreter.messages[-1] = merge_deltas(interpreter.messages[-1], chunk)
                for new_chunk_type in ['message', 'language', 'code']:
                    if new_chunk_type in chunk and chunk_type != new_chunk_type:
                        if chunk_type != None:
                            yield {f'end_of_{chunk_type}': True}
                        if new_chunk_type == 'language':
                            new_chunk_type = 'code'
                        chunk_type = new_chunk_type
                        yield {f'start_of_{chunk_type}': True}
                yield chunk
            if chunk_type:
                yield {f'end_of_{chunk_type}': True}
        except litellm.exceptions.BudgetExceededError:
            display_markdown_message(f'> Max budget exceeded\n\n                **Session spend:** ${litellm._current_cost}\n                **Max budget:** ${interpreter.max_budget}\n\n                Press CTRL-C then run `interpreter --max_budget [higher USD amount]` to proceed.\n            ')
            break
        except Exception as e:
            if interpreter.local == False and 'auth' in str(e).lower() or 'api key' in str(e).lower():
                output = traceback.format_exc()
                raise Exception(f"{output}\n\nThere might be an issue with your API key(s).\n\nTo reset your API key (we'll use OPENAI_API_KEY for this example, but you may need to reset your ANTHROPIC_API_KEY, HUGGINGFACE_API_KEY, etc):\n        Mac/Linux: 'export OPENAI_API_KEY=your-key-here',\n        Windows: 'setx OPENAI_API_KEY your-key-here' then restart terminal.\n\n")
            elif interpreter.local:
                raise Exception(str(e) + "\n\nPlease make sure LM Studio's local server is running by following the steps above.\n\nIf LM Studio's local server is running, please try a language model with a different architecture.\n\n                    ")
            else:
                raise
        if 'code' in interpreter.messages[-1]:
            if interpreter.debug_mode:
                print('Running code:', interpreter.messages[-1])
            try:
                code = interpreter.messages[-1]['code']
                if interpreter.messages[-1]['language'] == 'python' and code.startswith('!'):
                    code = code[1:]
                    interpreter.messages[-1]['code'] = code
                    interpreter.messages[-1]['language'] = 'shell'
                language = interpreter.messages[-1]['language'].lower().strip()
                if language in language_map:
                    if language not in interpreter._code_interpreters:
                        config = {'language': language, 'vision': interpreter.vision}
                        interpreter._code_interpreters[language] = create_code_interpreter(config)
                    code_interpreter = interpreter._code_interpreters[language]
                else:
                    output = f'Open Interpreter does not currently support `{language}`.'
                    yield {'output': output}
                    interpreter.messages[-1]['output'] = output
                    if code != last_unsupported_code:
                        last_unsupported_code = code
                        continue
                    else:
                        break
                try:
                    yield {'executing': {'code': code, 'language': language}}
                except GeneratorExit:
                    break
                interpreter.messages[-1]['output'] = ''
                for line in code_interpreter.run(code):
                    yield line
                    if 'output' in line:
                        output = interpreter.messages[-1]['output']
                        output += '\n' + line['output']
                        output = truncate_output(output, interpreter.max_output)
                        interpreter.messages[-1]['output'] = output.strip()
                    if interpreter.vision:
                        base64_image = None
                        if 'image' in line:
                            base64_image = line['image']
                        if 'html' in line:
                            base64_image = html_to_base64(line['html'])
                        if base64_image:
                            yield {'output': 'Sending image output to GPT-4V...'}
                            interpreter.messages[-1]['image'] = f'data:image/jpeg;base64,{base64_image}'
            except:
                output = traceback.format_exc()
                yield {'output': output.strip()}
                interpreter.messages[-1]['output'] = output.strip()
            yield {'active_line': None}
            yield {'end_of_execution': True}
        else:
            break
    return