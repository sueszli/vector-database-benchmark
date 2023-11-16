import discord
import random
__all__ = ('randomize_colour', 'randomize_color')

def randomize_colour(embed: discord.Embed) -> discord.Embed:
    if False:
        for i in range(10):
            print('nop')
    '\n    Gives the provided embed a random color.\n    There is an alias for this called randomize_color\n\n    Parameters\n    ----------\n    embed : discord.Embed\n        The embed to add a color to\n\n    Returns\n    -------\n    discord.Embed\n        The embed with the color set to a random color\n\n    '
    embed.colour = discord.Color(value=random.randint(0, 16777215))
    return embed
randomize_color = randomize_colour