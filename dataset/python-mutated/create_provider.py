import sys, re
from pathlib import Path
from os import path
sys.path.append(str(Path(__file__).parent.parent.parent))
import g4f
g4f.debug.logging = True

def read_code(text):
    if False:
        i = 10
        return i + 15
    if (match := re.search('```(python|py|)\\n(?P<code>[\\S\\s]+?)\\n```', text)):
        return match.group('code')

def input_command():
    if False:
        print('Hello World!')
    print('Enter/Paste the cURL command. Ctrl-D or Ctrl-Z ( windows ) to save it.')
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        contents.append(line)
    return '\n'.join(contents)
name = input('Name: ')
provider_path = f'g4f/Provider/{name}.py'
example = '\nfrom __future__ import annotations\n\nfrom aiohttp import ClientSession\n\nfrom ..typing import AsyncResult, Messages\nfrom .base_provider import AsyncGeneratorProvider\nfrom .helper import format_prompt\n\n\nclass ChatGpt(AsyncGeneratorProvider):\n    url                   = "https://chat-gpt.com"\n    supports_gpt_35_turbo = True\n    working               = True\n\n    @classmethod\n    async def create_async_generator(\n        cls,\n        model: str,\n        messages: Messages,\n        proxy: str = None,\n        **kwargs\n    ) -> AsyncResult:\n        headers = {\n            "authority": "chat-gpt.com",\n            "accept": "application/json",\n            "origin": cls.url,\n            "referer": f"{cls.url}/chat",\n        }\n        async with ClientSession(headers=headers) as session:\n            prompt = format_prompt(messages)\n            data = {\n                "prompt": prompt,\n                "purpose": "",\n            }\n            async with session.post(f"{cls.url}/api/chat", json=data, proxy=proxy) as response:\n                response.raise_for_status()\n                async for chunk in response.content:\n                    if chunk:\n                        yield chunk.decode()\n'
if not path.isfile(provider_path):
    command = input_command()
    prompt = f'\nCreate a provider from a cURL command. The command is:\n```bash\n{command}\n```\nA example for a provider:\n```py\n{example}\n```\nThe name for the provider class:\n{name}\nReplace "hello" with `format_prompt(messages)`.\nAnd replace "gpt-3.5-turbo" with `model`.\n'
    print('Create code...')
    response = []
    for chunk in g4f.ChatCompletion.create(model=g4f.models.gpt_35_long, messages=[{'role': 'user', 'content': prompt}], timeout=300, stream=True):
        print(chunk, end='', flush=True)
        response.append(chunk)
    print()
    response = ''.join(response)
    if (code := read_code(response)):
        with open(provider_path, 'w') as file:
            file.write(code)
        print('Saved at:', provider_path)
        with open('g4f/Provider/__init__.py', 'a') as file:
            file.write(f'\nfrom .{name} import {name}')
else:
    with open(provider_path, 'r') as file:
        code = file.read()