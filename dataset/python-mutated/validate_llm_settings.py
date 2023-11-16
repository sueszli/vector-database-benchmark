import getpass
import os
import time
import litellm
from ..utils.display_markdown_message import display_markdown_message

def validate_llm_settings(interpreter):
    if False:
        for i in range(10):
            print('nop')
    '\n    Interactivley prompt the user for required LLM settings\n    '
    while True:
        if interpreter.local:
            break
        else:
            if interpreter.model in litellm.open_ai_chat_completion_models:
                if not os.environ.get('OPENAI_API_KEY') and (not interpreter.api_key):
                    display_welcome_message_once()
                    display_markdown_message('---\n                    > OpenAI API key not found\n\n                    To use `GPT-4` (highly recommended) please provide an OpenAI API key.\n\n                    To use another language model, consult the documentation at [docs.openinterpreter.com](https://docs.openinterpreter.com/language-model-setup/).\n                    \n                    ---\n                    ')
                    response = getpass.getpass('OpenAI API key: ')
                    print(f'OpenAI API key: {response[:4]}...{response[-4:]}')
                    display_markdown_message('\n\n                    **Tip:** To save this key for later, run `export OPENAI_API_KEY=your_api_key` on Mac/Linux or `setx OPENAI_API_KEY your_api_key` on Windows.\n                    \n                    ---')
                    interpreter.api_key = response
                    time.sleep(2)
                    break
            break
    if not interpreter.auto_run and (not interpreter.local):
        display_markdown_message(f'> Model set to `{interpreter.model}`')
    return

def display_welcome_message_once():
    if False:
        print('Hello World!')
    '\n    Displays a welcome message only on its first call.\n\n    (Uses an internal attribute `_displayed` to track its state.)\n    '
    if not hasattr(display_welcome_message_once, '_displayed'):
        display_markdown_message('\n        ●\n\n        Welcome to **Open Interpreter**.\n        ')
        time.sleep(1.5)
        display_welcome_message_once._displayed = True