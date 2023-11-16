from instabot.singleton import Singleton

class BotCache(object):
    __metaclass__ = Singleton

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.following = None
        self.followers = None
        self.user_infos = {}
        self.usernames = {}

    def __repr__(self):
        if False:
            print('Hello World!')
        return self.__dict__