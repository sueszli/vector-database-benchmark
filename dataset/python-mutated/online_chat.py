from abc import ABCMeta
from enum import Enum

class UserService(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.users_by_id = {}

    def add_user(self, user_id, name, pass_hash):
        if False:
            for i in range(10):
                print('nop')
        pass

    def remove_user(self, user_id):
        if False:
            return 10
        pass

    def add_friend_request(self, from_user_id, to_user_id):
        if False:
            while True:
                i = 10
        pass

    def approve_friend_request(self, from_user_id, to_user_id):
        if False:
            while True:
                i = 10
        pass

    def reject_friend_request(self, from_user_id, to_user_id):
        if False:
            while True:
                i = 10
        pass

class User(object):

    def __init__(self, user_id, name, pass_hash):
        if False:
            for i in range(10):
                print('nop')
        self.user_id = user_id
        self.name = name
        self.pass_hash = pass_hash
        self.friends_by_id = {}
        self.friend_ids_to_private_chats = {}
        self.group_chats_by_id = {}
        self.received_friend_requests_by_friend_id = {}
        self.sent_friend_requests_by_friend_id = {}

    def message_user(self, friend_id, message):
        if False:
            for i in range(10):
                print('nop')
        pass

    def message_group(self, group_id, message):
        if False:
            for i in range(10):
                print('nop')
        pass

    def send_friend_request(self, friend_id):
        if False:
            return 10
        pass

    def receive_friend_request(self, friend_id):
        if False:
            i = 10
            return i + 15
        pass

    def approve_friend_request(self, friend_id):
        if False:
            for i in range(10):
                print('nop')
        pass

    def reject_friend_request(self, friend_id):
        if False:
            print('Hello World!')
        pass

class Chat(metaclass=ABCMeta):

    def __init__(self, chat_id):
        if False:
            while True:
                i = 10
        self.chat_id = chat_id
        self.users = []
        self.messages = []

class PrivateChat(Chat):

    def __init__(self, first_user, second_user):
        if False:
            while True:
                i = 10
        super(PrivateChat, self).__init__()
        self.users.append(first_user)
        self.users.append(second_user)

class GroupChat(Chat):

    def add_user(self, user):
        if False:
            for i in range(10):
                print('nop')
        pass

    def remove_user(self, user):
        if False:
            i = 10
            return i + 15
        pass

class Message(object):

    def __init__(self, message_id, message, timestamp):
        if False:
            print('Hello World!')
        self.message_id = message_id
        self.message = message
        self.timestamp = timestamp

class AddRequest(object):

    def __init__(self, from_user_id, to_user_id, request_status, timestamp):
        if False:
            print('Hello World!')
        self.from_user_id = from_user_id
        self.to_user_id = to_user_id
        self.request_status = request_status
        self.timestamp = timestamp

class RequestStatus(Enum):
    UNREAD = 0
    READ = 1
    ACCEPTED = 2
    REJECTED = 3