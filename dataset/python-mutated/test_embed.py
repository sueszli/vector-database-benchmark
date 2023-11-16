from sentry.integrations.discord.message_builder import LEVEL_TO_COLOR
from sentry.integrations.discord.message_builder.base.embed.base import DiscordMessageEmbed
from sentry.integrations.discord.message_builder.base.embed.field import DiscordMessageEmbedField
from sentry.integrations.discord.message_builder.base.embed.footer import DiscordMessageEmbedFooter
from sentry.testutils.cases import TestCase

class TestDiscordMessageEmbed(TestCase):

    def test_empty(self):
        if False:
            i = 10
            return i + 15
        embed = DiscordMessageEmbed()
        result = embed.build()
        assert result == {}

    def test_some(self):
        if False:
            while True:
                i = 10
        embed = DiscordMessageEmbed(title='Title', url='https://sentry.io', color=LEVEL_TO_COLOR['warning'])
        result = embed.build()
        assert result == {'title': 'Title', 'url': 'https://sentry.io', 'color': 16761383}

    def test_footer(self):
        if False:
            print('Hello World!')
        footer = DiscordMessageEmbedFooter(text='footer text', icon_url='https://sentry.io')
        embed = DiscordMessageEmbed(footer=footer)
        result = embed.build()
        assert result == {'footer': {'text': 'footer text', 'icon_url': 'https://sentry.io'}}

    def test_fields(self):
        if False:
            i = 10
            return i + 15
        field = DiscordMessageEmbedField('field name', 'field value', True)
        other_field = DiscordMessageEmbedField('other field name', 'other field value')
        embed = DiscordMessageEmbed(fields=[field, other_field])
        result = embed.build()
        assert result == {'fields': [{'name': 'field name', 'value': 'field value', 'inline': True}, {'name': 'other field name', 'value': 'other field value', 'inline': False}]}

    def test_all(self):
        if False:
            return 10
        footer = DiscordMessageEmbedFooter(text='footer text', icon_url='https://sentry.io')
        field = DiscordMessageEmbedField('field name', 'field value', True)
        embed = DiscordMessageEmbed(title='Title', description='description', url='https://sentry.io', color=LEVEL_TO_COLOR['warning'], footer=footer, fields=[field])
        result = embed.build()
        assert result == {'title': 'Title', 'description': 'description', 'url': 'https://sentry.io', 'color': 16761383, 'footer': {'text': 'footer text', 'icon_url': 'https://sentry.io'}, 'fields': [{'name': 'field name', 'value': 'field value', 'inline': True}]}