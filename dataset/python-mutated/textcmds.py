from errbot import BotPlugin, botcmd
(INROOM, USER, MULTILINE) = ('inroom', 'user', 'multiline')

class TextModeCmds(BotPlugin):
    """
    Internal to TextBackend.
    """
    __errdoc__ = 'Added commands for testing purposes'

    def activate(self):
        if False:
            for i in range(10):
                print('nop')
        if self.mode != 'text':
            return
        super().activate()
        if INROOM not in self:
            self[INROOM] = False
        if USER not in self:
            self[USER] = self.build_identifier(self.bot_config.BOT_ADMINS[0])
        if MULTILINE not in self:
            self[MULTILINE] = False
        self._bot._inroom = self[INROOM]
        self._bot.user = self[USER]
        self._bot._multiline = self[MULTILINE]

    def deactivate(self):
        if False:
            print('Hello World!')
        self[INROOM] = self._bot._inroom
        self[USER] = self._bot.user
        self[MULTILINE] = self._bot._multiline
        super().deactivate()

    @botcmd
    def inroom(self, msg, args):
        if False:
            while True:
                i = 10
        '\n        This puts you in a room with the bot.\n        '
        self._bot._inroom = True
        if args:
            room = args
        else:
            room = '#testroom'
        self._bot.query_room(room).join()
        return f'Joined Room {room}.'

    @botcmd
    def inperson(self, msg, _):
        if False:
            return 10
        '\n        This puts you in a 1-1 chat with the bot.\n        '
        self._bot._inroom = False
        return 'Now in one-on-one with the bot.'

    @botcmd
    def asuser(self, msg, args):
        if False:
            print('Hello World!')
        "\n        This puts you in a room with the bot. You can specify a name otherwise it will default to 'luser'.\n        "
        if args:
            usr = args
            if usr[0] != '@':
                usr = '@' + usr
            self._bot.user = self.build_identifier(usr)
        else:
            self._bot.user = self.build_identifier('@luser')
        return f'You are now: {self._bot.user}.'

    @botcmd
    def asadmin(self, msg, _):
        if False:
            for i in range(10):
                print('nop')
        '\n        This puts you in a 1-1 chat with the bot.\n        '
        self._bot.user = self.build_identifier(self.bot_config.BOT_ADMINS[0])
        return f'You are now an admin: {self._bot.user}.'

    @botcmd
    def ml(self, msg, _):
        if False:
            i = 10
            return i + 15
        '\n        Switch back and forth between normal mode and multiline mode. Use this if you want to test\n        commands spanning multiple lines. Note: in multiline, press enter twice to end and send the message.\n        '
        self._bot._multiline = not self._bot._multiline
        return 'Multiline mode, press enter twice to end messages' if self._bot._multiline else 'Normal one line mode.'