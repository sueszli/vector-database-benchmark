from Config import config
from Plugin import PluginManager
allow_reload = False

@PluginManager.registerTo('UserManager')
class UserManagerPlugin(object):

    def load(self):
        if False:
            while True:
                i = 10
        if not config.multiuser_local:
            if not self.users:
                self.users = {}
            return self.users
        else:
            return super(UserManagerPlugin, self).load()

    def get(self, master_address=None):
        if False:
            return 10
        users = self.list()
        if master_address in users:
            user = users[master_address]
        else:
            user = None
        return user

@PluginManager.registerTo('User')
class UserPlugin(object):

    def save(self):
        if False:
            for i in range(10):
                print('nop')
        if not config.multiuser_local:
            return False
        else:
            return super(UserPlugin, self).save()