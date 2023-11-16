from ..utils.display_markdown_message import display_markdown_message
from .components.code_block import CodeBlock
from .components.message_block import MessageBlock
from .magic_commands import handle_magic_command

def render_past_conversation(messages):
    if False:
        for i in range(10):
            print('nop')
    active_block = None
    render_cursor = False
    ran_code_block = False
    for chunk in messages:
        if chunk['role'] == 'user':
            if active_block:
                active_block.end()
                active_block = None
            print('>', chunk['message'])
            continue
        if 'message' in chunk:
            if active_block is None:
                active_block = MessageBlock()
            if active_block.type != 'message':
                active_block.end()
                active_block = MessageBlock()
            active_block.message += chunk['message']
        if 'code' in chunk or 'language' in chunk:
            if active_block is None:
                active_block = CodeBlock()
            if active_block.type != 'code' or ran_code_block:
                active_block.end()
                active_block = CodeBlock()
            ran_code_block = False
            render_cursor = True
        if 'language' in chunk:
            active_block.language = chunk['language']
        if 'code' in chunk:
            active_block.code += chunk['code']
        if 'active_line' in chunk:
            active_block.active_line = chunk['active_line']
        if 'output' in chunk:
            ran_code_block = True
            render_cursor = False
            active_block.output += '\n' + chunk['output']
            active_block.output = active_block.output.strip()
        if active_block:
            active_block.refresh(cursor=render_cursor)
    if active_block:
        active_block.end()
        active_block = None