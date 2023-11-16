from __future__ import annotations

import time, json, re
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider
from .helper import format_prompt

class ChatgptDemo(AsyncGeneratorProvider):
    url = "https://chat.chatgptdemo.net"
    supports_gpt_35_turbo = True
    working = False

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "authority": "chat.chatgptdemo.net",
            "accept-language": "de-DE,de;q=0.9,en-DE;q=0.8,en;q=0.7,en-US",
            "origin": "https://chat.chatgptdemo.net",
            "referer": "https://chat.chatgptdemo.net/",
            "sec-ch-ua": '"Google Chrome";v="117", "Not;A=Brand";v="8", "Chromium";v="117"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
        }
        async with ClientSession(headers=headers) as session:
            async with session.get(f"{cls.url}/", proxy=proxy) as response:
                response.raise_for_status()
                response = await response.text()
                if result := re.search(
                    r'<div id="USERID" style="display: none">(.*?)<\/div>',
                    response,
                ):
                    user_id = result.group(1)
                else:
                    raise RuntimeError("No user id found")
            async with session.post(f"{cls.url}/new_chat", json={"user_id": user_id}, proxy=proxy) as response:
                response.raise_for_status()
                chat_id = (await response.json())["id_"]
            if not chat_id:
                raise RuntimeError("Could not create new chat")
            data = {
                "question": format_prompt(messages),
                "chat_id": chat_id,
                "timestamp": int(time.time()*1000),
            }
            async with session.post(f"{cls.url}/chat_api_stream", json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for line in response.content:
                    if line.startswith(b"data: "):
                        line = json.loads(line[6:-1])
                        if chunk := line["choices"][0]["delta"].get("content"):
                            yield chunk