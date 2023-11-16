"""
Base classes for Instance Messenger clients.
"""
from twisted.words.im.locals import AWAY, OFFLINE, ONLINE

class ContactsList:
    """
    A GUI object that displays a contacts list.

    @ivar chatui: The GUI chat client associated with this contacts list.
    @type chatui: L{ChatUI}

    @ivar contacts: The contacts.
    @type contacts: C{dict} mapping C{str} to a L{IPerson<interfaces.IPerson>}
        provider

    @ivar onlineContacts: The contacts who are currently online (have a status
        that is not C{OFFLINE}).
    @type onlineContacts: C{dict} mapping C{str} to a
        L{IPerson<interfaces.IPerson>} provider

    @ivar clients: The signed-on clients.
    @type clients: C{list} of L{IClient<interfaces.IClient>} providers
    """

    def __init__(self, chatui):
        if False:
            for i in range(10):
                print('nop')
        '\n        @param chatui: The GUI chat client associated with this contacts list.\n        @type chatui: L{ChatUI}\n        '
        self.chatui = chatui
        self.contacts = {}
        self.onlineContacts = {}
        self.clients = []

    def setContactStatus(self, person):
        if False:
            for i in range(10):
                print('nop')
        "\n        Inform the user that a person's status has changed.\n\n        @param person: The person whose status has changed.\n        @type person: L{IPerson<interfaces.IPerson>} provider\n        "
        if person.name not in self.contacts:
            self.contacts[person.name] = person
        if person.name not in self.onlineContacts and (person.status == ONLINE or person.status == AWAY):
            self.onlineContacts[person.name] = person
        if person.name in self.onlineContacts and person.status == OFFLINE:
            del self.onlineContacts[person.name]

    def registerAccountClient(self, client):
        if False:
            for i in range(10):
                print('nop')
        '\n        Notify the user that an account client has been signed on to.\n\n        @param client: The client being added to your list of account clients.\n        @type client: L{IClient<interfaces.IClient>} provider\n        '
        if client not in self.clients:
            self.clients.append(client)

    def unregisterAccountClient(self, client):
        if False:
            print('Hello World!')
        '\n        Notify the user that an account client has been signed off or\n        disconnected from.\n\n        @param client: The client being removed from the list of account\n            clients.\n        @type client: L{IClient<interfaces.IClient>} provider\n        '
        if client in self.clients:
            self.clients.remove(client)

    def contactChangedNick(self, person, newnick):
        if False:
            return 10
        "\n        Update your contact information to reflect a change to a contact's\n        nickname.\n\n        @param person: The person in your contacts list whose nickname is\n            changing.\n        @type person: L{IPerson<interfaces.IPerson>} provider\n\n        @param newnick: The new nickname for this person.\n        @type newnick: C{str}\n        "
        oldname = person.name
        if oldname in self.contacts:
            del self.contacts[oldname]
            person.name = newnick
            self.contacts[newnick] = person
            if oldname in self.onlineContacts:
                del self.onlineContacts[oldname]
                self.onlineContacts[newnick] = person

class Conversation:
    """
    A GUI window of a conversation with a specific person.

    @ivar person: The person who you're having this conversation with.
    @type person: L{IPerson<interfaces.IPerson>} provider

    @ivar chatui: The GUI chat client associated with this conversation.
    @type chatui: L{ChatUI}
    """

    def __init__(self, person, chatui):
        if False:
            i = 10
            return i + 15
        "\n        @param person: The person who you're having this conversation with.\n        @type person: L{IPerson<interfaces.IPerson>} provider\n\n        @param chatui: The GUI chat client associated with this conversation.\n        @type chatui: L{ChatUI}\n        "
        self.chatui = chatui
        self.person = person

    def show(self):
        if False:
            while True:
                i = 10
        '\n        Display the ConversationWindow.\n        '
        raise NotImplementedError('Subclasses must implement this method')

    def hide(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Hide the ConversationWindow.\n        '
        raise NotImplementedError('Subclasses must implement this method')

    def sendText(self, text):
        if False:
            print('Hello World!')
        '\n        Send text to the person with whom the user is conversing.\n\n        @param text: The text to be sent.\n        @type text: C{str}\n        '
        self.person.sendMessage(text, None)

    def showMessage(self, text, metadata=None):
        if False:
            return 10
        '\n        Display a message sent from the person with whom the user is conversing.\n\n        @param text: The sent message.\n        @type text: C{str}\n\n        @param metadata: Metadata associated with this message.\n        @type metadata: C{dict}\n        '
        raise NotImplementedError('Subclasses must implement this method')

    def contactChangedNick(self, person, newnick):
        if False:
            while True:
                i = 10
        "\n        Change a person's name.\n\n        @param person: The person whose nickname is changing.\n        @type person: L{IPerson<interfaces.IPerson>} provider\n\n        @param newnick: The new nickname for this person.\n        @type newnick: C{str}\n        "
        self.person.name = newnick

class GroupConversation:
    """
    A GUI window of a conversation with a group of people.

    @ivar chatui: The GUI chat client associated with this conversation.
    @type chatui: L{ChatUI}

    @ivar group: The group of people that are having this conversation.
    @type group: L{IGroup<interfaces.IGroup>} provider

    @ivar members: The names of the people in this conversation.
    @type members: C{list} of C{str}
    """

    def __init__(self, group, chatui):
        if False:
            i = 10
            return i + 15
        '\n        @param chatui: The GUI chat client associated with this conversation.\n        @type chatui: L{ChatUI}\n\n        @param group: The group of people that are having this conversation.\n        @type group: L{IGroup<interfaces.IGroup>} provider\n        '
        self.chatui = chatui
        self.group = group
        self.members = []

    def show(self):
        if False:
            i = 10
            return i + 15
        '\n        Display the GroupConversationWindow.\n        '
        raise NotImplementedError('Subclasses must implement this method')

    def hide(self):
        if False:
            while True:
                i = 10
        '\n        Hide the GroupConversationWindow.\n        '
        raise NotImplementedError('Subclasses must implement this method')

    def sendText(self, text):
        if False:
            return 10
        '\n        Send text to the group.\n\n        @param text: The text to be sent.\n        @type text: C{str}\n        '
        self.group.sendGroupMessage(text, None)

    def showGroupMessage(self, sender, text, metadata=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Display to the user a message sent to this group from the given sender.\n\n        @param sender: The person sending the message.\n        @type sender: C{str}\n\n        @param text: The sent message.\n        @type text: C{str}\n\n        @param metadata: Metadata associated with this message.\n        @type metadata: C{dict}\n        '
        raise NotImplementedError('Subclasses must implement this method')

    def setGroupMembers(self, members):
        if False:
            i = 10
            return i + 15
        '\n        Set the list of members in the group.\n\n        @param members: The names of the people that will be in this group.\n        @type members: C{list} of C{str}\n        '
        self.members = members

    def setTopic(self, topic, author):
        if False:
            for i in range(10):
                print('nop')
        "\n        Change the topic for the group conversation window and display this\n        change to the user.\n\n        @param topic: This group's topic.\n        @type topic: C{str}\n\n        @param author: The person changing the topic.\n        @type author: C{str}\n        "
        raise NotImplementedError('Subclasses must implement this method')

    def memberJoined(self, member):
        if False:
            while True:
                i = 10
        '\n        Add the given member to the list of members in the group conversation\n        and displays this to the user.\n\n        @param member: The person joining the group conversation.\n        @type member: C{str}\n        '
        if member not in self.members:
            self.members.append(member)

    def memberChangedNick(self, oldnick, newnick):
        if False:
            while True:
                i = 10
        '\n        Change the nickname for a member of the group conversation and displays\n        this change to the user.\n\n        @param oldnick: The old nickname.\n        @type oldnick: C{str}\n\n        @param newnick: The new nickname.\n        @type newnick: C{str}\n        '
        if oldnick in self.members:
            self.members.remove(oldnick)
            self.members.append(newnick)

    def memberLeft(self, member):
        if False:
            i = 10
            return i + 15
        '\n        Delete the given member from the list of members in the group\n        conversation and displays the change to the user.\n\n        @param member: The person leaving the group conversation.\n        @type member: C{str}\n        '
        if member in self.members:
            self.members.remove(member)

class ChatUI:
    """
    A GUI chat client.

    @type conversations: C{dict} of L{Conversation}
    @ivar conversations: A cache of all the direct windows.

    @type groupConversations: C{dict} of L{GroupConversation}
    @ivar groupConversations: A cache of all the group windows.

    @type persons: C{dict} with keys that are a C{tuple} of (C{str},
       L{IAccount<interfaces.IAccount>} provider) and values that are
       L{IPerson<interfaces.IPerson>} provider
    @ivar persons: A cache of all the users associated with this client.

    @type groups: C{dict} with keys that are a C{tuple} of (C{str},
        L{IAccount<interfaces.IAccount>} provider) and values that are
        L{IGroup<interfaces.IGroup>} provider
    @ivar groups: A cache of all the groups associated with this client.

    @type onlineClients: C{list} of L{IClient<interfaces.IClient>} providers
    @ivar onlineClients: A list of message sources currently online.

    @type contactsList: L{ContactsList}
    @ivar contactsList: A contacts list.
    """

    def __init__(self):
        if False:
            return 10
        self.conversations = {}
        self.groupConversations = {}
        self.persons = {}
        self.groups = {}
        self.onlineClients = []
        self.contactsList = ContactsList(self)

    def registerAccountClient(self, client):
        if False:
            for i in range(10):
                print('nop')
        '\n        Notify the user that an account has been signed on to.\n\n        @type client: L{IClient<interfaces.IClient>} provider\n        @param client: The client account for the person who has just signed on.\n\n        @rtype: L{IClient<interfaces.IClient>} provider\n        @return: The client, so that it may be used in a callback chain.\n        '
        self.onlineClients.append(client)
        self.contactsList.registerAccountClient(client)
        return client

    def unregisterAccountClient(self, client):
        if False:
            for i in range(10):
                print('nop')
        '\n        Notify the user that an account has been signed off or disconnected.\n\n        @type client: L{IClient<interfaces.IClient>} provider\n        @param client: The client account for the person who has just signed\n            off.\n        '
        self.onlineClients.remove(client)
        self.contactsList.unregisterAccountClient(client)

    def getContactsList(self):
        if False:
            return 10
        '\n        Get the contacts list associated with this chat window.\n\n        @rtype: L{ContactsList}\n        @return: The contacts list associated with this chat window.\n        '
        return self.contactsList

    def getConversation(self, person, Class=Conversation, stayHidden=False):
        if False:
            i = 10
            return i + 15
        "\n        For the given person object, return the conversation window or create\n        and return a new conversation window if one does not exist.\n\n        @type person: L{IPerson<interfaces.IPerson>} provider\n        @param person: The person whose conversation window we want to get.\n\n        @type Class: L{IConversation<interfaces.IConversation>} implementor\n        @param Class: The kind of conversation window we want. If the conversation\n            window for this person didn't already exist, create one of this type.\n\n        @type stayHidden: C{bool}\n        @param stayHidden: Whether or not the conversation window should stay\n            hidden.\n\n        @rtype: L{IConversation<interfaces.IConversation>} provider\n        @return: The conversation window.\n        "
        conv = self.conversations.get(person)
        if not conv:
            conv = Class(person, self)
            self.conversations[person] = conv
        if stayHidden:
            conv.hide()
        else:
            conv.show()
        return conv

    def getGroupConversation(self, group, Class=GroupConversation, stayHidden=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        For the given group object, return the group conversation window or\n        create and return a new group conversation window if it doesn't exist.\n\n        @type group: L{IGroup<interfaces.IGroup>} provider\n        @param group: The group whose conversation window we want to get.\n\n        @type Class: L{IConversation<interfaces.IConversation>} implementor\n        @param Class: The kind of conversation window we want. If the conversation\n            window for this person didn't already exist, create one of this type.\n\n        @type stayHidden: C{bool}\n        @param stayHidden: Whether or not the conversation window should stay\n            hidden.\n\n        @rtype: L{IGroupConversation<interfaces.IGroupConversation>} provider\n        @return: The group conversation window.\n        "
        conv = self.groupConversations.get(group)
        if not conv:
            conv = Class(group, self)
            self.groupConversations[group] = conv
        if stayHidden:
            conv.hide()
        else:
            conv.show()
        return conv

    def getPerson(self, name, client):
        if False:
            print('Hello World!')
        '\n        For the given name and account client, return an instance of a\n        L{IGroup<interfaces.IPerson>} provider or create and return a new\n        instance of a L{IGroup<interfaces.IPerson>} provider.\n\n        @type name: C{str}\n        @param name: The name of the person of interest.\n\n        @type client: L{IClient<interfaces.IClient>} provider\n        @param client: The client account of interest.\n\n        @rtype: L{IPerson<interfaces.IPerson>} provider\n        @return: The person with that C{name}.\n        '
        account = client.account
        p = self.persons.get((name, account))
        if not p:
            p = account.getPerson(name)
            self.persons[name, account] = p
        return p

    def getGroup(self, name, client):
        if False:
            return 10
        '\n        For the given name and account client, return an instance of a\n        L{IGroup<interfaces.IGroup>} provider or create and return a new instance\n        of a L{IGroup<interfaces.IGroup>} provider.\n\n        @type name: C{str}\n        @param name: The name of the group of interest.\n\n        @type client: L{IClient<interfaces.IClient>} provider\n        @param client: The client account of interest.\n\n        @rtype: L{IGroup<interfaces.IGroup>} provider\n        @return: The group with that C{name}.\n        '
        account = client.account
        g = self.groups.get((name, account))
        if not g:
            g = account.getGroup(name)
            self.groups[name, account] = g
        return g

    def contactChangedNick(self, person, newnick):
        if False:
            for i in range(10):
                print('nop')
        "\n        For the given C{person}, change the C{person}'s C{name} to C{newnick}\n        and tell the contact list and any conversation windows with that\n        C{person} to change as well.\n\n        @type person: L{IPerson<interfaces.IPerson>} provider\n        @param person: The person whose nickname will get changed.\n\n        @type newnick: C{str}\n        @param newnick: The new C{name} C{person} will take.\n        "
        oldnick = person.name
        if (oldnick, person.account) in self.persons:
            conv = self.conversations.get(person)
            if conv:
                conv.contactChangedNick(person, newnick)
            self.contactsList.contactChangedNick(person, newnick)
            del self.persons[oldnick, person.account]
            person.name = newnick
            self.persons[person.name, person.account] = person