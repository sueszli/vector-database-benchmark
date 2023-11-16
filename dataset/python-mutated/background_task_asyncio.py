import discord
import asyncio

class MyClient(discord.Client):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)

    async def setup_hook(self) -> None:
        self.bg_task = self.loop.create_task(self.my_background_task())

    async def on_ready(self):
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('------')

    async def my_background_task(self):
        await self.wait_until_ready()
        counter = 0
        channel = self.get_channel(1234567)
        while not self.is_closed():
            counter += 1
            await channel.send(counter)
            await asyncio.sleep(60)
client = MyClient(intents=discord.Intents.default())
client.run('token')