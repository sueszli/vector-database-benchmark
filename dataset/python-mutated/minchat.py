"""
A very simple twisted.words.im-based logbot.

To run the script:
$ python minchat.py
"""
from twisted.words.im import baseaccount, basechat, ircsupport
accounts = [ircsupport.IRCAccount('IRC', 1, 'Tooty', '', 'irc.freenode.net', 6667, '#twisted')]

class AccountManager(baseaccount.AccountManager):
    """
    This class is a minimal implementation of the Account Manager.

    Most implementations will show some screen that lets the user add and
    remove accounts, but we're not quite that sophisticated.
    """

    def __init__(self):
        if False:
            return 10
        self.chatui = MinChat()
        if len(accounts) == 0:
            print('You have defined no accounts.')
        for acct in accounts:
            acct.logOn(self.chatui)

class MinConversation(basechat.Conversation):
    """
    This class is a minimal implementation of the abstract Conversation class.

    This is all you need to override to receive one-on-one messages.
    """

    def show(self):
        if False:
            print('Hello World!')
        "\n        If you don't have a GUI, this is a no-op.\n        "
        pass

    def hide(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        If you don't have a GUI, this is a no-op.\n        "
        pass

    def showMessage(self, text, metadata=None):
        if False:
            i = 10
            return i + 15
        print(f'<{self.person.name}> {text}')

    def contactChangedNick(self, person, newnick):
        if False:
            return 10
        basechat.Conversation.contactChangedNick(self, person, newnick)
        print(f'-!- {person.name} is now known as {newnick}')

class MinGroupConversation(basechat.GroupConversation):
    """
    This class is a minimal implementation of the abstract GroupConversation class.

    This is all you need to override to listen in on a group conversation.
    """

    def show(self):
        if False:
            return 10
        "\n        If you don't have a GUI, this is a no-op.\n        "
        pass

    def hide(self):
        if False:
            while True:
                i = 10
        "\n        If you don't have a GUI, this is a no-op.\n        "
        pass

    def showGroupMessage(self, sender, text, metadata=None):
        if False:
            while True:
                i = 10
        print(f'<{sender}/{self.group.name}> {text}')

    def setTopic(self, topic, author):
        if False:
            while True:
                i = 10
        print(f'-!- {author} set the topic of {self.group.name} to: {topic}')

    def memberJoined(self, member):
        if False:
            print('Hello World!')
        basechat.GroupConversation.memberJoined(self, member)
        print(f'-!- {member} joined {self.group.name}')

    def memberChangedNick(self, oldnick, newnick):
        if False:
            return 10
        basechat.GroupConversation.memberChangedNick(self, oldnick, newnick)
        print(f'-!- {oldnick} is now known as {newnick} in {self.group.name}')

    def memberLeft(self, member):
        if False:
            return 10
        basechat.GroupConversation.memberLeft(self, member)
        print(f'-!- {member} left {self.group.name}')

class MinChat(basechat.ChatUI):
    """
    This class is a minimal implementation of the abstract ChatUI class.

    There are only two methods that need overriding - and of those two,
    the only change that needs to be made is the default value of the Class
    parameter.
    """

    def getGroupConversation(self, group, Class=MinGroupConversation, stayHidden=0):
        if False:
            return 10
        return basechat.ChatUI.getGroupConversation(self, group, Class, stayHidden)

    def getConversation(self, person, Class=MinConversation, stayHidden=0):
        if False:
            while True:
                i = 10
        return basechat.ChatUI.getConversation(self, person, Class, stayHidden)
if __name__ == '__main__':
    from twisted.internet import reactor
    AccountManager()
    reactor.run()