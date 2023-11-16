import asyncio
import logging
import logging.handlers
import os
from typing import List, Optional
import asyncpg
import discord
from discord.ext import commands
from aiohttp import ClientSession

class CustomBot(commands.Bot):

    def __init__(self, *args, initial_extensions: List[str], db_pool: asyncpg.Pool, web_client: ClientSession, testing_guild_id: Optional[int]=None, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self.db_pool = db_pool
        self.web_client = web_client
        self.testing_guild_id = testing_guild_id
        self.initial_extensions = initial_extensions

    async def setup_hook(self) -> None:
        for extension in self.initial_extensions:
            await self.load_extension(extension)
        if self.testing_guild_id:
            guild = discord.Object(self.testing_guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)

async def main():
    logger = logging.getLogger('discord')
    logger.setLevel(logging.INFO)
    handler = logging.handlers.RotatingFileHandler(filename='discord.log', encoding='utf-8', maxBytes=32 * 1024 * 1024, backupCount=5)
    dt_fmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter('[{asctime}] [{levelname:<8}] {name}: {message}', dt_fmt, style='{')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    async with ClientSession() as our_client, asyncpg.create_pool(user='postgres', command_timeout=30) as pool:
        exts = ['general', 'mod', 'dice']
        intents = discord.Intents.default()
        intents.message_content = True
        async with CustomBot(commands.when_mentioned, db_pool=pool, web_client=our_client, initial_extensions=exts, intents=intents) as bot:
            await bot.start('token')
asyncio.run(main())