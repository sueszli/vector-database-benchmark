from __future__ import unicode_literals
from wxpy import *
from requests import get
from requests import post
from platform import system
from os import chdir
from random import choice
from threading import Thread
import configparser
import time
import sys

def get_message():
    if False:
        i = 10
        return i + 15
    r = get('http://open.iciba.com/dsapi/')
    note = r.json()['note']
    content = r.json()['content']
    return (note, content)

def send_message(your_message):
    if False:
        i = 10
        return i + 15
    try:
        my_friend = bot.friends().search(my_lady_wechat_name)[0]
        my_friend.send(your_message)
    except:
        bot.file_helper.send(u'守护女友出问题了，赶紧去看看咋回事~')

def start_care():
    if False:
        return 10
    message = ''
    while True:
        print('守护中，时间:%s' % time.ctime())
        now_time = time.ctime()[-13:-8]
        if now_time == say_good_morning:
            message = choice(str_list_good_morning)
            if flag_wx_emoj:
                message = message + choice(str_list_emoj)
            send_message(message)
            print('提醒女友早上起床:%s' % time.ctime())
        elif now_time == say_good_lunch:
            message = choice(str_list_good_lunch)
            if flag_wx_emoj:
                message = message + choice(str_list_emoj)
            send_message(message)
            print('提醒女友中午吃饭:%s' % time.ctime())
        elif now_time == say_good_dinner:
            message = choice(str_list_good_dinner)
            if flag_wx_emoj:
                message = message + choice(str_list_emoj)
            send_message(message)
            print('提醒女友晚上吃饭:%s' % time.ctime())
        elif now_time == say_good_dream:
            if flag_learn_english:
                (note, content) = get_message()
                message = choice(str_list_good_dream) + '\n\n' + '顺便一起来学英语哦：\n' + '原文: ' + content + '\n\n翻译: ' + note
            else:
                message = choice(str_list_good_dream)
            if flag_wx_emoj:
                message = message + choice(str_list_emoj)
            send_message(message)
            print('提醒女友晚上睡觉:%s' % time.ctime())
        festival_month = time.strftime('%m', time.localtime())
        festival_day = time.strftime('%d', time.localtime())
        if festival_month == '02' and festival_day == '14' and (now_time == '08:00'):
            send_message(str_Valentine)
            print('发送情人节祝福:%s' % time.ctime())
        elif festival_month == '03' and festival_day == '08' and (now_time == '08:00'):
            send_message(str_Women)
            print('发送三八妇女节祝福:%s' % time.ctime())
        elif festival_month == '12' and festival_day == '24' and (now_time == '00:00'):
            send_message(str_Christmas_Eve)
            print('发送平安夜祝福:%s' % time.ctime())
        elif festival_month == '12' and festival_day == '25' and (now_time == '00:00'):
            send_message(str_Christmas)
            print('发送圣诞节祝福:%s' % time.ctime())
        if festival_month == birthday_month and festival_day == birthday_day and (now_time == '00:00'):
            send_message(str_birthday)
            print('发送生日祝福:%s' % time.ctime())
        time.sleep(60)
if __name__ == '__main__':
    if 'Windows' in system():
        bot = Bot()
    elif 'Darwin' in system():
        bot = Bot()
    elif 'Linux' in system():
        bot = Bot(console_qr=2, cache_path=True)
    else:
        print('无法识别你的操作系统类型，请自己设置')
    cf = configparser.ConfigParser()
    cf.read('./config.ini', encoding='UTF-8')
    my_lady_wechat_name = cf.get('configuration', 'my_lady_wechat_name')
    say_good_morning = cf.get('configuration', 'say_good_morning')
    say_good_lunch = cf.get('configuration', 'say_good_lunch')
    say_good_dinner = cf.get('configuration', 'say_good_dinner')
    say_good_dream = cf.get('configuration', 'say_good_dream')
    birthday_month = cf.get('configuration', 'birthday_month')
    birthday_day = cf.get('configuration', 'birthday_day')
    str_list_good_morning = ''
    with open('./remind_sentence/sentence_good_morning.txt', 'r', encoding='UTF-8') as f:
        str_list_good_morning = f.readlines()
    print(str_list_good_morning)
    str_list_good_lunch = ''
    with open('./remind_sentence/sentence_good_lunch.txt', 'r', encoding='UTF-8') as f:
        str_list_good_lunch = f.readlines()
    print(str_list_good_lunch)
    str_list_good_dinner = ''
    with open('./remind_sentence/sentence_good_dinner.txt', 'r', encoding='UTF-8') as f:
        str_list_good_dinner = f.readlines()
    print(str_list_good_dinner)
    str_list_good_dream = ''
    with open('./remind_sentence/sentence_good_dream.txt', 'r', encoding='UTF-8') as f:
        str_list_good_dream = f.readlines()
    print(str_list_good_dream)
    if cf.get('configuration', 'flag_learn_english') == '1':
        flag_learn_english = True
    else:
        flag_learn_english = False
    print(flag_learn_english)
    str_emoj = "(•‾̑⌣‾̑•)✧˖°----(๑´ڡ`๑)----(๑¯ิε ¯ิ๑)----(๑•́ ₃ •̀๑)----( ∙̆ .̯ ∙̆ )----(๑˘ ˘๑)----(●′ω`●)----(●･̆⍛･̆●)----ಥ_ಥ----_(:qゝ∠)----(´；ω；`)----( `)3')----Σ((( つ•̀ω•́)つ----╰(*´︶`*)╯----( ´´ิ∀´ิ` )----(´∩｀。)----( ื▿ ื)----(｡ŏ_ŏ)----( •ิ _ •ิ )----ヽ(*΄◞ิ౪◟ิ‵ *)----( ˘ ³˘)----(; ´_ゝ`)----(*ˉ﹃ˉ)----(◍'౪`◍)ﾉﾞ----(｡◝‿◜｡)----(ಠ .̫.̫ ಠ)----(´◞⊖◟`)----(。≖ˇェˇ≖｡)----(◕ܫ◕)----(｀◕‸◕´+)----(▼ _ ▼)----( ◉ืൠ◉ื)----ㄟ(◑‿◐ )ㄏ----(●'◡'●)ﾉ♥----(｡◕ˇ∀ˇ◕）----( ◔ ڼ ◔ )----( ´◔ ‸◔`)----(☍﹏⁰)----(♥◠‿◠)----ლ(╹◡╹ლ )----(๑꒪◞౪◟꒪๑)"
    str_list_emoj = str_emoj.split('----')
    if cf.get('configuration', 'flag_wx_emoj') == '1':
        flag_wx_emoj = True
    else:
        flag_wx_emoj = False
    print(str_list_emoj)
    str_Valentine = cf.get('configuration', 'str_Valentine')
    print(str_Valentine)
    str_Women = cf.get('configuration', 'str_Women')
    print(str_Women)
    str_Christmas_Eve = cf.get('configuration', 'str_Christmas_Eve')
    print(str_Christmas_Eve)
    str_Christmas = cf.get('configuration', 'str_Christmas')
    print(str_Christmas)
    str_birthday = cf.get('configuration', 'str_birthday')
    print(str_birthday)
    t = Thread(target=start_care, name='start_care')
    t.start()
my_girl_friend = bot.friends().search(my_lady_wechat_name)[0]

@bot.register(chats=my_girl_friend, except_self=False)
def print_others(msg):
    if False:
        while True:
            i = 10
    print(msg.text)
    postData = {'data': msg.text}
    response = post('https://bosonnlp.com/analysis/sentiment?analysisType=', data=postData)
    data = response.text
    now_mod_rank = data.split(',')[0].replace('[[', '')
    print('来自女友的消息:%s\n当前情感得分:%s\n越接近1表示心情越好，越接近0表示心情越差，情感结果仅供参考，请勿完全相信！\n\n' % (msg.text, now_mod_rank))
    mood_message = u'来自女友的消息:' + msg.text + '\n当前情感得分:' + now_mod_rank + '\n越接近1表示心情越好，越接近0表示心情越差，情感结果仅供参考，请勿完全相信！\n\n'
    bot.file_helper.send(mood_message)