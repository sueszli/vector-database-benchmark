"""

This is a account reset tool.
Use this before you start boting.
You can then reset the users you follow to what you had before botting.

"""
import os
import sys
sys.path.append(os.path.join(sys.path[0], '../'))
from instabot import Bot

class Task(object):

    @staticmethod
    def start(bot):
        if False:
            while True:
                i = 10
        answer = input('\n        Please select\n        1) Create Friends List\n            Make a list of the users you follow before you follow bot.\n        2) Restore Friends List\n            Unfollow all user accept for the users in your friends list.\n        3) Exit\n        \n\n        ')
        answer = str(answer)
        if answer == '1':
            Task.one(bot)
        if answer == '2':
            Task.two(bot)
        if answer == '3':
            exit()
        else:
            print('Type 1,2 or 3.')
            Task.start(bot)

    @staticmethod
    def one(bot):
        if False:
            print('Hello World!')
        print('Creating List')
        friends = bot.following
        with open('friends_{}.txt'.format(bot.username), 'w') as file:
            for user_id in friends:
                file.write(str(user_id) + '\n')
        print('Task Done')
        Task.start(bot)

    @staticmethod
    def two(bot):
        if False:
            print('Hello World!')
        friends = bot.read_list_from_file('friends_{}.txt'.format(bot.username))
        your_following = bot.following
        unfollow = list(set(your_following) - set(friends))
        bot.unfollow_users(unfollow)
        Task.start(bot)
bot = Bot()
bot.login()
print('\n    Welcome to this bot.\n    It will now get a list of all of the users you are following.\n    You will need this if you follow bot your account and you want\n    to reset your following to just your friends.\n    ')
Task.start(bot)