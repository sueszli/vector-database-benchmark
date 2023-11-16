from __future__ import annotations
from discord.ext import commands
import discord
import re

class DynamicCounter(discord.ui.DynamicItem[discord.ui.Button], template='counter:(?P<count>[0-9]+):user:(?P<id>[0-9]+)'):

    def __init__(self, user_id: int, count: int=0) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.user_id: int = user_id
        self.count: int = count
        super().__init__(discord.ui.Button(label=f'Total: {count}', style=self.style, custom_id=f'counter:{count}:user:{user_id}', emoji='👍'))

    @property
    def style(self) -> discord.ButtonStyle:
        if False:
            print('Hello World!')
        if self.count < 10:
            return discord.ButtonStyle.grey
        if self.count < 15:
            return discord.ButtonStyle.red
        if self.count < 20:
            return discord.ButtonStyle.blurple
        return discord.ButtonStyle.green

    @classmethod
    async def from_custom_id(cls, interaction: discord.Interaction, item: discord.ui.Button, match: re.Match[str], /):
        count = int(match['count'])
        user_id = int(match['id'])
        return cls(user_id, count=count)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return interaction.user.id == self.user_id

    async def callback(self, interaction: discord.Interaction) -> None:
        self.count += 1
        self.item.label = f'Total: {self.count}'
        self.custom_id = f'counter:{self.count}:user:{self.user_id}'
        self.item.style = self.style
        await interaction.response.edit_message(view=self.view)

class DynamicCounterBot(commands.Bot):

    def __init__(self):
        if False:
            while True:
                i = 10
        intents = discord.Intents.default()
        super().__init__(command_prefix=commands.when_mentioned, intents=intents)

    async def setup_hook(self) -> None:
        self.add_dynamic_items(DynamicCounter)

    async def on_ready(self):
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('------')
bot = DynamicCounterBot()

@bot.command()
async def counter(ctx: commands.Context):
    """Starts a dynamic counter."""
    view = discord.ui.View(timeout=None)
    view.add_item(DynamicCounter(ctx.author.id))
    await ctx.send('Here is your very own button!', view=view)
bot.run('token')