"""
This file handles conversations.
"""
import json
import os
import platform
import subprocess
import inquirer
from ..utils.display_markdown_message import display_markdown_message
from ..utils.local_storage_path import get_storage_path
from .render_past_conversation import render_past_conversation

def conversation_navigator(interpreter):
    if False:
        for i in range(10):
            print('nop')
    conversations_dir = get_storage_path('conversations')
    display_markdown_message(f'> Conversations are stored in "`{conversations_dir}`".\n    \n    Select a conversation to resume.\n    ')
    if not os.path.exists(conversations_dir):
        print(f'No conversations found in {conversations_dir}')
        return None
    json_files = sorted([f for f in os.listdir(conversations_dir) if f.endswith('.json')], key=lambda x: os.path.getmtime(os.path.join(conversations_dir, x)), reverse=True)
    readable_names_and_filenames = {}
    for filename in json_files:
        name = filename.replace('.json', '').replace('.JSON', '').replace('__', '... (').replace('_', ' ') + ')'
        readable_names_and_filenames[name] = filename
    readable_names_and_filenames['> Open folder'] = None
    questions = [inquirer.List('name', message='', choices=readable_names_and_filenames.keys())]
    answers = inquirer.prompt(questions)
    if answers['name'] == '> Open folder':
        open_folder(conversations_dir)
        return
    selected_filename = readable_names_and_filenames[answers['name']]
    with open(os.path.join(conversations_dir, selected_filename), 'r') as f:
        messages = json.load(f)
    render_past_conversation(messages)
    interpreter.messages = messages
    interpreter.conversation_filename = selected_filename
    interpreter.chat()

def open_folder(path):
    if False:
        while True:
            i = 10
    if platform.system() == 'Windows':
        os.startfile(path)
    elif platform.system() == 'Darwin':
        subprocess.run(['open', path])
    else:
        subprocess.run(['xdg-open', path])