import os
import litellm
from .convert_to_coding_llm import convert_to_coding_llm
from .setup_openai_coding_llm import setup_openai_coding_llm
from .setup_text_llm import setup_text_llm

def setup_llm(interpreter):
    if False:
        print('Hello World!')
    '\n    Takes an Interpreter (which includes a ton of LLM settings),\n    returns a Coding LLM (a generator that streams deltas with `message` and `code`).\n    '
    if interpreter.function_calling_llm == None:
        if not interpreter.local and (interpreter.model != 'gpt-4-vision-preview' and interpreter.model in litellm.open_ai_chat_completion_models or interpreter.model.startswith('azure/')):
            interpreter.function_calling_llm = True
        else:
            interpreter.function_calling_llm = False
    if interpreter.function_calling_llm:
        coding_llm = setup_openai_coding_llm(interpreter)
    else:
        if interpreter.disable_procedures == None:
            if interpreter.model != 'gpt-4-vision-preview':
                interpreter.disable_procedures = True
        text_llm = setup_text_llm(interpreter)
        coding_llm = convert_to_coding_llm(text_llm, debug_mode=interpreter.debug_mode)
    return coding_llm