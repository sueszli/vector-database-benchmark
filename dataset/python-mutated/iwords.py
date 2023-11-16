from zope.interface import Attribute, Interface

class IProtocolPlugin(Interface):
    """Interface for plugins providing an interface to a Words service"""
    name = Attribute('A single word describing what kind of interface this is (eg, irc or web)')

    def getFactory(realm, portal):
        if False:
            while True:
                i = 10
        'Retrieve a C{twisted.internet.interfaces.IServerFactory} provider\n\n        @param realm: An object providing C{twisted.cred.portal.IRealm} and\n        L{IChatService}, with which service information should be looked up.\n\n        @param portal: An object providing C{twisted.cred.portal.IPortal},\n        through which logins should be performed.\n        '

class IGroup(Interface):
    name = Attribute('A short string, unique among groups.')

    def add(user):
        if False:
            return 10
        'Include the given user in this group.\n\n        @type user: L{IUser}\n        '

    def remove(user, reason=None):
        if False:
            return 10
        'Remove the given user from this group.\n\n        @type user: L{IUser}\n        @type reason: C{unicode}\n        '

    def size():
        if False:
            return 10
        'Return the number of participants in this group.\n\n        @rtype: L{twisted.internet.defer.Deferred}\n        @return: A Deferred which fires with an C{int} representing the\n        number of participants in this group.\n        '

    def receive(sender, recipient, message):
        if False:
            return 10
        '\n        Broadcast the given message from the given sender to other\n        users in group.\n\n        The message is not re-transmitted to the sender.\n\n        @param sender: L{IUser}\n\n        @type recipient: L{IGroup}\n        @param recipient: This is probably a wart.  Maybe it will be removed\n        in the future.  For now, it should be the group object the message\n        is being delivered to.\n\n        @param message: C{dict}\n\n        @rtype: L{twisted.internet.defer.Deferred}\n        @return: A Deferred which fires with None when delivery has been\n        attempted for all users.\n        '

    def setMetadata(meta):
        if False:
            print('Hello World!')
        'Change the metadata associated with this group.\n\n        @type meta: C{dict}\n        '

    def iterusers():
        if False:
            return 10
        'Return an iterator of all users in this group.'

class IChatClient(Interface):
    """Interface through which IChatService interacts with clients."""
    name = Attribute('A short string, unique among users.  This will be set by the L{IChatService} at login time.')

    def receive(sender, recipient, message):
        if False:
            i = 10
            return i + 15
        '\n        Callback notifying this user of the given message sent by the\n        given user.\n\n        This will be invoked whenever another user sends a message to a\n        group this user is participating in, or whenever another user sends\n        a message directly to this user.  In the former case, C{recipient}\n        will be the group to which the message was sent; in the latter, it\n        will be the same object as the user who is receiving the message.\n\n        @type sender: L{IUser}\n        @type recipient: L{IUser} or L{IGroup}\n        @type message: C{dict}\n\n        @rtype: L{twisted.internet.defer.Deferred}\n        @return: A Deferred which fires when the message has been delivered,\n        or which fails in some way.  If the Deferred fails and the message\n        was directed at a group, this user will be removed from that group.\n        '

    def groupMetaUpdate(group, meta):
        if False:
            for i in range(10):
                print('nop')
        '\n        Callback notifying this user that the metadata for the given\n        group has changed.\n\n        @type group: L{IGroup}\n        @type meta: C{dict}\n\n        @rtype: L{twisted.internet.defer.Deferred}\n        '

    def userJoined(group, user):
        if False:
            while True:
                i = 10
        '\n        Callback notifying this user that the given user has joined\n        the given group.\n\n        @type group: L{IGroup}\n        @type user: L{IUser}\n\n        @rtype: L{twisted.internet.defer.Deferred}\n        '

    def userLeft(group, user, reason=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Callback notifying this user that the given user has left the\n        given group for the given reason.\n\n        @type group: L{IGroup}\n        @type user: L{IUser}\n        @type reason: C{unicode}\n\n        @rtype: L{twisted.internet.defer.Deferred}\n        '

class IUser(Interface):
    """Interface through which clients interact with IChatService."""
    realm = Attribute('A reference to the Realm to which this user belongs.  Set if and only if the user is logged in.')
    mind = Attribute('A reference to the mind which logged in to this user.  Set if and only if the user is logged in.')
    name = Attribute('A short string, unique among users.')
    lastMessage = Attribute('A POSIX timestamp indicating the time of the last message received from this user.')
    signOn = Attribute("A POSIX timestamp indicating this user's most recent sign on time.")

    def loggedIn(realm, mind):
        if False:
            return 10
        'Invoked by the associated L{IChatService} when login occurs.\n\n        @param realm: The L{IChatService} through which login is occurring.\n        @param mind: The mind object used for cred login.\n        '

    def send(recipient, message):
        if False:
            i = 10
            return i + 15
        'Send the given message to the given user or group.\n\n        @type recipient: Either L{IUser} or L{IGroup}\n        @type message: C{dict}\n        '

    def join(group):
        if False:
            print('Hello World!')
        'Attempt to join the given group.\n\n        @type group: L{IGroup}\n        @rtype: L{twisted.internet.defer.Deferred}\n        '

    def leave(group):
        if False:
            i = 10
            return i + 15
        'Discontinue participation in the given group.\n\n        @type group: L{IGroup}\n        @rtype: L{twisted.internet.defer.Deferred}\n        '

    def itergroups():
        if False:
            for i in range(10):
                print('nop')
        '\n        Return an iterator of all groups of which this user is a\n        member.\n        '

class IChatService(Interface):
    name = Attribute('A short string identifying this chat service (eg, a hostname)')
    createGroupOnRequest = Attribute('A boolean indicating whether L{getGroup} should implicitly create groups which are requested but which do not yet exist.')
    createUserOnRequest = Attribute('A boolean indicating whether L{getUser} should implicitly create users which are requested but which do not yet exist.')

    def itergroups():
        if False:
            for i in range(10):
                print('nop')
        'Return all groups available on this service.\n\n        @rtype: C{twisted.internet.defer.Deferred}\n        @return: A Deferred which fires with a list of C{IGroup} providers.\n        '

    def getGroup(name):
        if False:
            for i in range(10):
                print('nop')
        'Retrieve the group by the given name.\n\n        @type name: C{str}\n\n        @rtype: L{twisted.internet.defer.Deferred}\n        @return: A Deferred which fires with the group with the given\n        name if one exists (or if one is created due to the setting of\n        L{IChatService.createGroupOnRequest}, or which fails with\n        L{twisted.words.ewords.NoSuchGroup} if no such group exists.\n        '

    def createGroup(name):
        if False:
            while True:
                i = 10
        'Create a new group with the given name.\n\n        @type name: C{str}\n\n        @rtype: L{twisted.internet.defer.Deferred}\n        @return: A Deferred which fires with the created group, or\n        with fails with L{twisted.words.ewords.DuplicateGroup} if a\n        group by that name exists already.\n        '

    def lookupGroup(name):
        if False:
            i = 10
            return i + 15
        'Retrieve a group by name.\n\n        Unlike C{getGroup}, this will never implicitly create a group.\n\n        @type name: C{str}\n\n        @rtype: L{twisted.internet.defer.Deferred}\n        @return: A Deferred which fires with the group by the given\n        name, or which fails with L{twisted.words.ewords.NoSuchGroup}.\n        '

    def getUser(name):
        if False:
            i = 10
            return i + 15
        'Retrieve the user by the given name.\n\n        @type name: C{str}\n\n        @rtype: L{twisted.internet.defer.Deferred}\n        @return: A Deferred which fires with the user with the given\n        name if one exists (or if one is created due to the setting of\n        L{IChatService.createUserOnRequest}, or which fails with\n        L{twisted.words.ewords.NoSuchUser} if no such user exists.\n        '

    def createUser(name):
        if False:
            i = 10
            return i + 15
        'Create a new user with the given name.\n\n        @type name: C{str}\n\n        @rtype: L{twisted.internet.defer.Deferred}\n        @return: A Deferred which fires with the created user, or\n        with fails with L{twisted.words.ewords.DuplicateUser} if a\n        user by that name exists already.\n        '
__all__ = ['IGroup', 'IChatClient', 'IUser', 'IChatService']