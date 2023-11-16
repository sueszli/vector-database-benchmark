from __future__ import unicode_literals
import getpass
import os
import random
import sys
import time
from tqdm import tqdm
sys.path.append(os.path.join(sys.path[0], '../'))
from instabot import Bot

def initial_checker():
    if False:
        return 10
    files = [hashtag_file, users_file, whitelist, blacklist, comment, setting_file]
    try:
        for f in files:
            with open(f, 'r') as f:
                pass
    except BaseException:
        for f in files:
            with open(f, 'w') as f:
                pass
        print("\n        Welcome to instabot, it seems this is your first time.\n        Before starting, let's setup the basics.\n        So the bot functions the way you want.\n        ")
        setting_input()
        print('\n        You can add hashtag database, competitor database,\n        whitelists, blacklists and also add users in setting menu.\n        Have fun with the bot!\n        ')
        time.sleep(5)
        os.system('cls')

def read_input(f, msg, n=None):
    if False:
        print('Hello World!')
    if n is not None:
        msg += ' (enter to use default number: {})'.format(n)
    print(msg)
    entered = sys.stdin.readline().strip() or str(n)
    if isinstance(n, int):
        entered = int(entered)
    f.write(str(entered) + '\n')

def setting_input():
    if False:
        print('Hello World!')
    inputs = [('How many likes do you want to do in a day?', 500), ('How about unlike? ', 250), ('How many follows do you want to do in a day? ', 250), ('How about unfollow? ', 50), ('How many comments do you want to do in a day? ', 100), ('Maximal likes in media you will like?\nWe will skip media that have greater like than this value ', 100), ('Maximal followers of account you want to follow?\nWe will skip media that have greater followers than ' + 'this value ', 2000), ('Minimum followers a account should have before we follow?\nWe will skip media that have lesser followers than ' + 'this value ', 10), ('Maximum following of account you want to follow?\nWe will skip media that have a greater following ' + 'than this value ', 7500), ('Minimum following of account you want to follow?\nWe will skip media that have lesser following ' + 'from this value ', 10), ('Maximal followers to following_ratio ', 10), ('Maximal following to followers_ratio ', 2), ('Minimal media the account you will follow have.\nWe will skip media that have lesser media from this value ', 3), ('Delay from one like to another like you will perform ', 20), ('Delay from one unlike to another unlike you will perform ', 20), ('Delay from one follow to another follow you will perform ', 90), ('Delay from one unfollow to another unfollow you will perform ', 90), ('Delay from one comment to another comment you will perform ', 600), ('Want to use proxy? insert your proxy or leave it blank ' + 'if no. (just enter', 'None')]
    with open(setting_file, 'w') as f:
        for (msg, n) in inputs:
            read_input(f, msg, n)
        print('Done with all settings!')

def parameter_setting():
    if False:
        return 10
    settings = ['Max likes per day: ', 'Max unlikes per day: ', 'Max follows per day: ', 'Max unfollows per day: ', 'Max comments per day: ', 'Max likes to like: ', 'Max followers to follow: ', 'Min followers to follow: ', 'Max following to follow: ', 'Min following to follow: ', 'Max followers to following_ratio: ', 'Max following to followers_ratio: ', 'Min media_count to follow:', 'Like delay: ', 'Unlike delay: ', 'Follow delay: ', 'Unfollow delay: ', 'Comment delay: ', 'Proxy: ']
    with open(setting_file) as f:
        data = f.readlines()
    print('Current parameters\n')
    for (s, d) in zip(settings, data):
        print(s + d)

def username_adder():
    if False:
        i = 10
        return i + 15
    with open(SECRET_FILE, 'a') as f:
        print('We will add your instagram account.')
        print("Don't worry. It will be stored locally.")
        while True:
            print('Enter your login: ')
            f.write(str(sys.stdin.readline().strip()) + ':')
            print('Enter your password: (it will not be shown due to security reasons - just start typing and press Enter)')
            f.write(getpass.getpass() + '\n')
            if input('Do you want to add another account? (y/n)').lower() != 'y':
                break

def get_adder(name, fname):
    if False:
        return 10

    def _adder():
        if False:
            return 10
        print('Current Database:')
        print(bot.read_list_from_file(fname))
        with open(fname, 'a') as f:
            print('Add {} to database'.format(name))
            while True:
                print('Enter {}: '.format(name))
                f.write(str(sys.stdin.readline().strip()) + '\n')
                print('Do you want to add another {}? (y/n)\n'.format(name))
                if 'y' not in sys.stdin.readline():
                    print('Done adding {}s to database'.format(name))
                    break
    return _adder()

def hashtag_adder():
    if False:
        return 10
    return get_adder('hashtag', fname=hashtag_file)

def competitor_adder():
    if False:
        return 10
    return get_adder('username', fname=users_file)

def blacklist_adder():
    if False:
        for i in range(10):
            print('nop')
    return get_adder('username', fname=blacklist)

def whitelist_adder():
    if False:
        return 10
    return get_adder('username', fname=whitelist)

def comment_adder():
    if False:
        print('Hello World!')
    return get_adder('comment', fname=comment)

def userlist_maker():
    if False:
        while True:
            i = 10
    return get_adder('username', userlist)

def menu():
    if False:
        return 10
    ans = True
    while ans:
        print('\n        1.Follow\n        2.Like\n        3.Comment\n        4.Unfollow\n        5.Block\n        6.Setting\n        7.Exit\n        ')
        ans = input('What would you like to do?\n').strip()
        if ans == '1':
            menu_follow()
        elif ans == '2':
            menu_like()
        elif ans == '3':
            menu_comment()
        elif ans == '4':
            menu_unfollow()
        elif ans == '5':
            menu_block()
        elif ans == '6':
            menu_setting()
        elif ans == '7':
            bot.logout()
            sys.exit()
        else:
            print('\n Not A Valid Choice, Try again')

def menu_follow():
    if False:
        while True:
            i = 10
    ans = True
    while ans:
        print('\n        1. Follow from hashtag\n        2. Follow followers\n        3. Follow following\n        4. Follow by likes on media\n        5. Main menu\n        ')
        ans = input('How do you want to follow?\n').strip()
        if ans == '1':
            print('\n            1.Insert hashtag\n            2.Use hashtag database\n            ')
            hashtags = []
            if '1' in sys.stdin.readline():
                hashtags = input('Insert hashtags separated by spaces\nExample: cat dog\nwhat hashtags?\n').strip().split(' ')
            else:
                hashtags = bot.read_list_from_file(hashtag_file)
            for hashtag in hashtags:
                print('Begin following: ' + hashtag)
                users = bot.get_hashtag_users(hashtag)
                bot.follow_users(users)
            menu_follow()
        elif ans == '2':
            print('\n            1.Insert username\n            2.Use username database\n            ')
            if '1' in sys.stdin.readline():
                user_id = input('who?\n').strip()
            else:
                user_id = random.choice(bot.read_list_from_file(users_file))
            bot.follow_followers(user_id)
            menu_follow()
        elif ans == '3':
            print('\n            1.Insert username\n            2.Use username database\n            ')
            if '1' in sys.stdin.readline():
                user_id = input('who?\n').strip()
            else:
                user_id = random.choice(bot.read_list_from_file(users_file))
            bot.follow_following(user_id)
            menu_follow()
        elif ans == '4':
            print('\n            1.Insert username\n            2.Use username database\n            ')
            if '1' in sys.stdin.readline():
                user_id = input('who?\n').strip()
            else:
                user_id = random.choice(bot.read_list_from_file(users_file))
            medias = bot.get_user_medias(user_id, filtration=False)
            if len(medias):
                likers = bot.get_media_likers(medias[0])
                for liker in tqdm(likers):
                    bot.follow(liker)
        elif ans == '5':
            menu()
        else:
            print('This number is not in the list?')
            menu_follow()

def menu_like():
    if False:
        print('Hello World!')
    ans = True
    while ans:
        print('\n        1. Like from hashtag(s)\n        2. Like followers\n        3. Like following\n        4. Like last media likers\n        5. Like our timeline\n        6. Main menu\n        ')
        ans = input('How do you want to like?\n').strip()
        if ans == '1':
            print('\n            1.Insert hashtag(s)\n            2.Use hashtag database\n            ')
            hashtags = []
            if '1' in sys.stdin.readline():
                hashtags = input('Insert hashtags separated by spaces\nExample: cat dog\nwhat hashtags?\n').strip().split(' ')
            else:
                hashtags.append(random.choice(bot.read_list_from_file(hashtag_file)))
            for hashtag in hashtags:
                bot.like_hashtag(hashtag)
        elif ans == '2':
            print('\n            1.Insert username\n            2.Use username database\n            ')
            if '1' in sys.stdin.readline():
                user_id = input('who?\n').strip()
            else:
                user_id = random.choice(bot.read_list_from_file(users_file))
            bot.like_followers(user_id)
        elif ans == '3':
            print('\n            1.Insert username\n            2.Use username database\n            ')
            if '1' in sys.stdin.readline():
                user_id = input('who?\n').strip()
            else:
                user_id = random.choice(bot.read_list_from_file(users_file))
            bot.like_following(user_id)
        elif ans == '4':
            print('\n            1.Insert username\n            2.Use username database\n            ')
            if '1' in sys.stdin.readline():
                user_id = input('who?\n').strip()
            else:
                user_id = random.choice(bot.read_list_from_file(users_file))
            medias = bot.get_user_medias(user_id, filtration=False)
            if len(medias):
                likers = bot.get_media_likers(medias[0])
                for liker in tqdm(likers):
                    bot.like_user(liker, amount=2, filtration=False)
        elif ans == '5':
            bot.like_timeline()
        elif ans == '6':
            menu()
        else:
            print('This number is not in the list?')
            menu_like()

def menu_comment():
    if False:
        return 10
    ans = True
    while ans:
        print('\n        1. Comment from hashtag\n        2. Comment spesific user media\n        3. Comment userlist\n        4. Comment our timeline\n        5. Main menu\n        ')
        ans = input('How do you want to comment?\n').strip()
        if ans == '1':
            print('\n            1.Insert hashtag\n            2.Use hashtag database\n            ')
            if '1' in sys.stdin.readline():
                hashtag = input('what?').strip()
            else:
                hashtag = random.choice(bot.read_list_from_file(hashtag_file))
            bot.comment_hashtag(hashtag)
        elif ans == '2':
            print('\n            1.Insert username\n            2.Use username database\n            ')
            if '1' in sys.stdin.readline():
                user_id = input('who?\n').strip()
            else:
                user_id = random.choice(bot.read_list_from_file(users_file))
            bot.comment_medias(bot.get_user_medias(user_id, filtration=False))
        elif ans == '3':
            print('\n            1.Make a list\n            2.Use existing list\n            ')
            if '1' in sys.stdin.readline():
                userlist_maker()
            if '2' in sys.stdin.readline():
                print(userlist)
            users = bot.read_list_from_file(userlist)
            for user_id in users:
                bot.comment_medias(bot.get_user_medias(user_id, filtration=True))
        elif ans == '4':
            bot.comment_medias(bot.get_timeline_medias())
        elif ans == '5':
            menu()
        else:
            print('This number is not in the list?')
            menu_comment()

def menu_unfollow():
    if False:
        print('Hello World!')
    ans = True
    while ans:
        print('\n        1. Unfollow non followers\n        2. Unfollow everyone\n        3. Main menu\n        ')
        ans = input('How do you want to unfollow?\n').strip()
        if ans == '1':
            bot.unfollow_non_followers()
            menu_unfollow()
        elif ans == '2':
            bot.unfollow_everyone()
            menu_unfollow()
        elif ans == '3':
            menu()
        else:
            print('This number is not in the list?')
            menu_unfollow()

def menu_block():
    if False:
        for i in range(10):
            print('nop')
    ans = True
    while ans:
        print('\n        1. Block bot\n        2. Main menu\n        ')
        ans = input('how do you want to block?\n').strip()
        if ans == '1':
            bot.block_bots()
            menu_block()
        elif ans == '2':
            menu()
        else:
            print('This number is not in the list?')
            menu_block()

def menu_setting():
    if False:
        return 10
    ans = True
    while ans:
        print('\n        1. Setting bot parameter\n        2. Add user accounts\n        3. Add competitor database\n        4. Add hashtag database\n        5. Add Comment database\n        6. Add blacklist\n        7. Add whitelist\n        8. Clear all database\n        9. Main menu\n        ')
        ans = input('What setting do you need?\n').strip()
        if ans == '1':
            parameter_setting()
            change = input('Want to change it? y/n\n').strip()
            if change == 'y' or change == 'Y':
                setting_input()
            else:
                menu_setting()
        elif ans == '2':
            username_adder()
        elif ans == '3':
            competitor_adder()
        elif ans == '4':
            hashtag_adder()
        elif ans == '5':
            comment_adder()
        elif ans == '6':
            blacklist_adder()
        elif ans == '7':
            whitelist_adder()
        elif ans == '8':
            print('Whis will clear all database except your user accounts and paramater settings')
            time.sleep(5)
            open(hashtag_file, 'w')
            open(users_file, 'w')
            open(whitelist, 'w')
            open(blacklist, 'w')
            open(comment, 'w')
            print('Done, you can add new one!')
        elif ans == '9':
            menu()
        else:
            print('This number is not in the list?')
            menu_setting()
try:
    input = raw_input
except NameError:
    pass
hashtag_file = 'config/hashtag_database.txt'
users_file = 'config/username_database.txt'
whitelist = 'config/whitelist.txt'
blacklist = 'config/blacklist.txt'
userlist = 'config/userlist.txt'
comment = 'config/comments.txt'
setting_file = 'config/setting_multiscript.txt'
SECRET_FILE = 'config/secret.txt'
initial_checker()
if os.stat(setting_file).st_size == 0:
    print('Looks like setting are broken')
    print("Let's make new one")
    setting_input()
f = open(setting_file)
lines = f.readlines()
settings = []
for i in range(0, 19):
    settings.append(lines[i].strip())
bot = Bot(max_likes_per_day=int(settings[0]), max_unlikes_per_day=int(settings[1]), max_follows_per_day=int(settings[2]), max_unfollows_per_day=int(settings[3]), max_comments_per_day=int(settings[4]), max_likes_to_like=int(settings[5]), max_followers_to_follow=int(settings[6]), min_followers_to_follow=int(settings[7]), max_following_to_follow=int(settings[8]), min_following_to_follow=int(settings[9]), max_followers_to_following_ratio=int(settings[10]), max_following_to_followers_ratio=int(settings[11]), min_media_count_to_follow=int(settings[12]), like_delay=int(settings[13]), unlike_delay=int(settings[14]), follow_delay=int(settings[15]), unfollow_delay=int(settings[16]), comment_delay=int(settings[17]))
bot.login()
while True:
    try:
        menu()
    except Exception as e:
        bot.logger.exception(str(e))
        bot.logger.debug('error, retry')
    time.sleep(1)