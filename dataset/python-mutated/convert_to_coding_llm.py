from ..utils.convert_to_openai_messages import convert_to_openai_messages
from .setup_text_llm import setup_text_llm

def convert_to_coding_llm(text_llm, debug_mode=False):
    if False:
        print('Hello World!')
    "\n    Takes a text_llm\n    returns an OI Coding LLM (a generator that takes OI messages and streams deltas with `message`, 'language', and `code`).\n    "

    def coding_llm(messages):
        if False:
            for i in range(10):
                print('nop')
        assert messages[0]['role'] == 'system'
        messages[0]['message'] += "\nTo execute code on the user's machine, write a markdown code block. Specify the language after the ```. You will receive the output. Use any programming language."
        '\n        gaslight = None\n        if messages[-1]["role"] == "user":\n            # Last message came from the user.\n            if messages[-1]["message"].lower() not in [\n                "hello",\n                "hi",\n                "hey",\n                "helo",\n                "hii",\n                "hi!",\n            ]:  # :)\n                gaslight = """Let\'s explore this. I can run code on your machine by writing the code in a markdown code block. This works if I put a newline after ```shell, ```python, ```applescript, etc. then write code. I\'m going to try to do this for your task **after I make a plan**. I\'ll put the *correct* language after the "```"."""\n        else:\n            # Last message came from the assistant.\n\n            # (The below should actually always be True in OI if last message came from the assistant)\n            # I think we don\'t need this actually.\n            """\n            if "output" in messages[-1]:\n                if messages[-1]["output"] != "No output":\n                    gaslight = "(Thought: I see that the code I just ran produced an output. The next message I send will go to the user.)"\n                elif messages[-1]["output"] == "No output":\n                    gaslight = "(Thought: I see that the code I just ran produced no output. The next message I send will go to the user.)"\n            """\n\n        if gaslight:\n            messages.append({"role": "assistant", "message": gaslight})\n        '
        if 'code' in messages[-1]:
            if any([line.startswith('!') for line in messages[-1]['code'].split('\n')]):
                if 'syntax' in messages[-1]['output'].lower():
                    messages[-1]['output'] += "\nRemember you are not in a Jupyter notebook. Run shell by writing a markdown shell codeblock, not '!'."
        messages = convert_to_openai_messages(messages, function_calling=False)
        inside_code_block = False
        accumulated_block = ''
        language = None
        for chunk in text_llm(messages):
            if debug_mode:
                print('Chunk in coding_llm', chunk)
            if 'choices' not in chunk or len(chunk['choices']) == 0:
                continue
            content = chunk['choices'][0]['delta'].get('content', '')
            accumulated_block += content
            if accumulated_block.endswith('`'):
                continue
            if '```' in accumulated_block and (not inside_code_block):
                inside_code_block = True
                accumulated_block = accumulated_block.split('```')[1]
            if inside_code_block and '```' in accumulated_block:
                return
            if inside_code_block:
                if language is None and '\n' in accumulated_block:
                    language = accumulated_block.split('\n')[0]
                    if language == '':
                        language = 'python'
                    else:
                        language = ''.join((char for char in language if char.isalpha()))
                    output = {'language': language}
                    if content.split('\n')[1]:
                        output['code'] = content.split('\n')[1]
                    yield output
                elif language:
                    yield {'code': content}
            if not inside_code_block:
                yield {'message': content}
    return coding_llm