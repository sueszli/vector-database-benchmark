from discord.ext import tasks
import discord

class MyClient(discord.Client):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self.counter = 0

    async def setup_hook(self) -> None:
        self.my_background_task.start()

    async def on_ready(self):
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('------')

    @tasks.loop(seconds=60)
    async def my_background_task(self):
        channel = self.get_channel(1234567)
        self.counter += 1
        await channel.send(self.counter)

    @my_background_task.before_loop
    async def before_my_task(self):
        await self.wait_until_ready()
client = MyClient(intents=discord.Intents.default())
client.run('token')