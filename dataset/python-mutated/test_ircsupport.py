"""
Tests for L{twisted.words.im.ircsupport}.
"""
from twisted.internet.testing import StringTransport
from twisted.words.im.basechat import ChatUI, Conversation, GroupConversation
from twisted.words.im.ircsupport import IRCAccount, IRCProto
from twisted.words.im.locals import OfflineError
from twisted.words.test.test_irc import IRCTestCase

class StubConversation(Conversation):

    def show(self):
        if False:
            i = 10
            return i + 15
        pass

    def showMessage(self, message, metadata):
        if False:
            while True:
                i = 10
        self.message = message
        self.metadata = metadata

class StubGroupConversation(GroupConversation):

    def setTopic(self, topic, nickname):
        if False:
            return 10
        self.topic = topic
        self.topicSetBy = nickname

    def show(self):
        if False:
            while True:
                i = 10
        pass

    def showGroupMessage(self, sender, text, metadata=None):
        if False:
            return 10
        self.metadata = metadata
        self.text = text
        self.metadata = metadata

class StubChatUI(ChatUI):

    def getConversation(self, group, Class=StubConversation, stayHidden=0):
        if False:
            return 10
        return ChatUI.getGroupConversation(self, group, Class, stayHidden)

    def getGroupConversation(self, group, Class=StubGroupConversation, stayHidden=0):
        if False:
            return 10
        return ChatUI.getGroupConversation(self, group, Class, stayHidden)

class IRCProtoTests(IRCTestCase):
    """
    Tests for L{IRCProto}.
    """

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.account = IRCAccount('Some account', False, 'alice', None, 'example.com', 6667)
        self.proto = IRCProto(self.account, StubChatUI(), None)
        self.transport = StringTransport()

    def test_login(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        When L{IRCProto} is connected to a transport, it sends I{NICK} and\n        I{USER} commands with the username from the account object.\n        '
        self.proto.makeConnection(self.transport)
        self.assertEqualBufferValue(self.transport.value(), 'NICK alice\r\nUSER alice foo bar :Twisted-IM user\r\n')

    def test_authenticate(self) -> None:
        if False:
            return 10
        '\n        If created with an account with a password, L{IRCProto} sends a\n        I{PASS} command before the I{NICK} and I{USER} commands.\n        '
        self.account.password = 'secret'
        self.proto.makeConnection(self.transport)
        self.assertEqualBufferValue(self.transport.value(), 'PASS secret\r\nNICK alice\r\nUSER alice foo bar :Twisted-IM user\r\n')

    def test_channels(self) -> None:
        if False:
            return 10
        '\n        If created with an account with a list of channels, L{IRCProto}\n        joins each of those channels after registering.\n        '
        self.account.channels = ['#foo', '#bar']
        self.proto.makeConnection(self.transport)
        self.assertEqualBufferValue(self.transport.value(), 'NICK alice\r\nUSER alice foo bar :Twisted-IM user\r\nJOIN #foo\r\nJOIN #bar\r\n')

    def test_isupport(self) -> None:
        if False:
            return 10
        '\n        L{IRCProto} can interpret I{ISUPPORT} (I{005}) messages from the server\n        and reflect their information in its C{supported} attribute.\n        '
        self.proto.makeConnection(self.transport)
        self.proto.dataReceived(':irc.example.com 005 alice MODES=4 CHANLIMIT=#:20\r\n')
        self.assertEqual(4, self.proto.supported.getFeature('MODES'))

    def test_nick(self) -> None:
        if False:
            while True:
                i = 10
        '\n        IRC NICK command changes the nickname of a user.\n        '
        self.proto.makeConnection(self.transport)
        self.proto.dataReceived(':alice JOIN #group1\r\n')
        self.proto.dataReceived(':alice1 JOIN #group1\r\n')
        self.proto.dataReceived(':alice1 NICK newnick\r\n')
        self.proto.dataReceived(':alice3 NICK newnick3\r\n')
        self.assertIn('newnick', self.proto._ingroups)
        self.assertNotIn('alice1', self.proto._ingroups)

    def test_part(self) -> None:
        if False:
            while True:
                i = 10
        '\n        IRC PART command removes a user from an IRC channel.\n        '
        self.proto.makeConnection(self.transport)
        self.proto.dataReceived(':alice1 JOIN #group1\r\n')
        self.assertIn('group1', self.proto._ingroups['alice1'])
        self.assertNotIn('group2', self.proto._ingroups['alice1'])
        self.proto.dataReceived(':alice PART #group1\r\n')
        self.proto.dataReceived(':alice1 PART #group1\r\n')
        self.proto.dataReceived(':alice1 PART #group2\r\n')
        self.assertNotIn('group1', self.proto._ingroups['alice1'])
        self.assertNotIn('group2', self.proto._ingroups['alice1'])

    def test_quit(self) -> None:
        if False:
            print('Hello World!')
        '\n        IRC QUIT command removes a user from all IRC channels.\n        '
        self.proto.makeConnection(self.transport)
        self.proto.dataReceived(':alice1 JOIN #group1\r\n')
        self.assertIn('group1', self.proto._ingroups['alice1'])
        self.assertNotIn('group2', self.proto._ingroups['alice1'])
        self.proto.dataReceived(':alice1 JOIN #group3\r\n')
        self.assertIn('group3', self.proto._ingroups['alice1'])
        self.proto.dataReceived(':alice1 QUIT\r\n')
        self.assertTrue(len(self.proto._ingroups['alice1']) == 0)
        self.proto.dataReceived(':alice3 QUIT\r\n')

    def test_topic(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        IRC TOPIC command changes the topic of an IRC channel.\n        '
        self.proto.makeConnection(self.transport)
        self.proto.dataReceived(':alice1 JOIN #group1\r\n')
        self.proto.dataReceived(':alice1 TOPIC #group1 newtopic\r\n')
        groupConversation = self.proto.getGroupConversation('group1')
        self.assertEqual(groupConversation.topic, 'newtopic')
        self.assertEqual(groupConversation.topicSetBy, 'alice1')

    def test_privmsg(self) -> None:
        if False:
            while True:
                i = 10
        '\n        PRIVMSG sends a private message to a user or channel.\n        '
        self.proto.makeConnection(self.transport)
        self.proto.dataReceived(':alice1 PRIVMSG t2 test_message_1\r\n')
        conversation = self.proto.chat.getConversation(self.proto.getPerson('alice1'))
        self.assertEqual(conversation.message, 'test_message_1')
        self.proto.dataReceived(':alice1 PRIVMSG #group1 test_message_2\r\n')
        groupConversation = self.proto.getGroupConversation('group1')
        self.assertEqual(groupConversation.text, 'test_message_2')
        self.proto.setNick('alice')
        self.proto.dataReceived(':alice PRIVMSG #foo test_message_3\r\n')
        groupConversation = self.proto.getGroupConversation('foo')
        self.assertFalse(hasattr(groupConversation, 'text'))
        conversation = self.proto.chat.getConversation(self.proto.getPerson('alice'))
        self.assertFalse(hasattr(conversation, 'message'))

    def test_action(self) -> None:
        if False:
            while True:
                i = 10
        '\n        CTCP ACTION to a user or channel.\n        '
        self.proto.makeConnection(self.transport)
        self.proto.dataReceived(':alice1 PRIVMSG alice1 :\x01ACTION smiles\x01\r\n')
        conversation = self.proto.chat.getConversation(self.proto.getPerson('alice1'))
        self.assertEqual(conversation.message, 'smiles')
        self.proto.dataReceived(':alice1 PRIVMSG #group1 :\x01ACTION laughs\x01\r\n')
        groupConversation = self.proto.getGroupConversation('group1')
        self.assertEqual(groupConversation.text, 'laughs')
        self.proto.setNick('alice')
        self.proto.dataReceived(':alice PRIVMSG #group1 :\x01ACTION cries\x01\r\n')
        groupConversation = self.proto.getGroupConversation('foo')
        self.assertFalse(hasattr(groupConversation, 'text'))
        conversation = self.proto.chat.getConversation(self.proto.getPerson('alice'))
        self.assertFalse(hasattr(conversation, 'message'))

    def test_rplNamreply(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        RPL_NAMREPLY server response (353) lists all the users in a channel.\n        RPL_ENDOFNAMES server response (363) is sent at the end of RPL_NAMREPLY\n        to indicate that there are no more names.\n        '
        self.proto.makeConnection(self.transport)
        self.proto.dataReceived(':example.com 353 z3p = #bnl :pSwede Dan- SkOyg @MrOp +MrPlus\r\n')
        expectedInGroups = {'Dan-': ['bnl'], 'pSwede': ['bnl'], 'SkOyg': ['bnl'], 'MrOp': ['bnl'], 'MrPlus': ['bnl']}
        expectedNamReplies = {'bnl': ['pSwede', 'Dan-', 'SkOyg', 'MrOp', 'MrPlus']}
        self.assertEqual(expectedInGroups, self.proto._ingroups)
        self.assertEqual(expectedNamReplies, self.proto._namreplies)
        self.proto.dataReceived(':example.com 366 alice #bnl :End of /NAMES list\r\n')
        self.assertEqual({}, self.proto._namreplies)
        groupConversation = self.proto.getGroupConversation('bnl')
        self.assertEqual(expectedNamReplies['bnl'], groupConversation.members)

    def test_rplTopic(self) -> None:
        if False:
            return 10
        "\n        RPL_TOPIC server response (332) is sent when a channel's topic is changed\n        "
        self.proto.makeConnection(self.transport)
        self.proto.dataReceived(':example.com 332 alice, #foo :Some random topic\r\n')
        self.assertEqual('Some random topic', self.proto._topics['foo'])

    def test_sendMessage(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{IRCPerson.sendMessage}\n        '
        self.proto.makeConnection(self.transport)
        person = self.proto.getPerson('alice')
        self.assertRaises(OfflineError, person.sendMessage, 'Some message')
        person.account.client = self.proto
        self.transport.clear()
        person.sendMessage('Some message 2')
        self.assertEqual(self.transport.io.getvalue(), b'PRIVMSG alice :Some message 2\r\n')
        self.transport.clear()
        person.sendMessage('smiles', {'style': 'emote'})
        self.assertEqual(self.transport.io.getvalue(), b'PRIVMSG alice :\x01ACTION smiles\x01\r\n')

    def test_sendGroupMessage(self) -> None:
        if False:
            while True:
                i = 10
        '\n        L{IRCGroup.sendGroupMessage}\n        '
        self.proto.makeConnection(self.transport)
        group = self.proto.chat.getGroup('#foo', self.proto)
        self.assertRaises(OfflineError, group.sendGroupMessage, 'Some message')
        group.account.client = self.proto
        self.transport.clear()
        group.sendGroupMessage('Some message 2')
        self.assertEqual(self.transport.io.getvalue(), b'PRIVMSG #foo :Some message 2\r\n')
        self.transport.clear()
        group.sendGroupMessage('smiles', {'style': 'emote'})
        self.assertEqual(self.transport.io.getvalue(), b'PRIVMSG #foo :\x01ACTION smiles\x01\r\n')