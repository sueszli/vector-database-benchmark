import discord

class MyClient(discord.Client):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self.role_message_id = 0
        self.emoji_to_role = {discord.PartialEmoji(name='🔴'): 0, discord.PartialEmoji(name='🟡'): 0, discord.PartialEmoji(name='green', id=0): 0}

    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        """Gives a role based on a reaction emoji."""
        if payload.message_id != self.role_message_id:
            return
        guild = self.get_guild(payload.guild_id)
        if guild is None:
            return
        try:
            role_id = self.emoji_to_role[payload.emoji]
        except KeyError:
            return
        role = guild.get_role(role_id)
        if role is None:
            return
        try:
            await payload.member.add_roles(role)
        except discord.HTTPException:
            pass

    async def on_raw_reaction_remove(self, payload: discord.RawReactionActionEvent):
        """Removes a role based on a reaction emoji."""
        if payload.message_id != self.role_message_id:
            return
        guild = self.get_guild(payload.guild_id)
        if guild is None:
            return
        try:
            role_id = self.emoji_to_role[payload.emoji]
        except KeyError:
            return
        role = guild.get_role(role_id)
        if role is None:
            return
        member = guild.get_member(payload.user_id)
        if member is None:
            return
        try:
            await member.remove_roles(role)
        except discord.HTTPException:
            pass
intents = discord.Intents.default()
intents.members = True
client = MyClient(intents=intents)
client.run('token')