from discord.ext import commands
import discord

class EphemeralCounterBot(commands.Bot):

    def __init__(self):
        if False:
            while True:
                i = 10
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=commands.when_mentioned_or('$'), intents=intents)

    async def on_ready(self):
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('------')

class Counter(discord.ui.View):

    @discord.ui.button(label='0', style=discord.ButtonStyle.red)
    async def count(self, interaction: discord.Interaction, button: discord.ui.Button):
        number = int(button.label) if button.label else 0
        if number + 1 >= 5:
            button.style = discord.ButtonStyle.green
            button.disabled = True
        button.label = str(number + 1)
        await interaction.response.edit_message(view=self)

class EphemeralCounter(discord.ui.View):

    @discord.ui.button(label='Click', style=discord.ButtonStyle.blurple)
    async def receive(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message('Enjoy!', view=Counter(), ephemeral=True)
bot = EphemeralCounterBot()

@bot.command()
async def counter(ctx: commands.Context):
    """Starts a counter for pressing."""
    await ctx.send('Press!', view=EphemeralCounter())
bot.run('token')