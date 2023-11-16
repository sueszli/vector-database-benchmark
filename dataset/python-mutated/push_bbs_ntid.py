import sys
import io
import os
import time
import re
import json
import base64
import threading
sys.path.append(os.getcwd() + '/class/core')
import mw
import telebot
from telebot import types
from telebot.util import quick_markup
chat_id = -1001578009023

def writeLog(log_str):
    if False:
        return 10
    if __name__ == '__main__':
        print(log_str)
    now = mw.getDateFromNow()
    log_file = mw.getServerDir() + '/tgbot/task.log'
    mw.writeFileLog(now + ':' + log_str, log_file, limit_size=5 * 1024)
    return True

def get_newest_tid():
    if False:
        while True:
            i = 10
    api_new = 'https://bbs.midoks.me/plugin.php?id=external_api&f=bbs_newest'
    api_next = 'https://bbs.midoks.me/plugin.php?id=external_api&f=bbs_next_tid&tid='
    tid_push = mw.getServerDir() + '/tgbot/bbs_newest_push.json'
    if not os.path.exists(tid_push):
        data = mw.httpGet(api_new)
        data = json.loads(data)
        if data['code'] == 0:
            tid = data['data'][0]['tid']
            mw.writeFile(tid_push, tid)
            return (True, data['data'][0])
    tid = mw.readFile(tid_push)
    data = mw.httpGet(api_next + tid)
    data = json.loads(data)
    if data['code'] == 0 and len(data['data']) > 0:
        tid = data['data'][0]['tid']
        mw.writeFile(tid_push, tid)
        return (True, data['data'][0])
    return (False, None)

def send_msg(bot, tag='ad', trigger_time=300):
    if False:
        while True:
            i = 10
    lock_file = mw.getServerDir() + '/tgbot/lock.json'
    if not os.path.exists(lock_file):
        mw.writeFile(lock_file, '{}')
    lock_data = json.loads(mw.readFile(lock_file))
    if tag in lock_data:
        diff_time = time.time() - lock_data[tag]['do_time']
        if diff_time >= trigger_time:
            lock_data[tag]['do_time'] = time.time()
        else:
            return (False, 0, 0)
    else:
        lock_data[tag] = {'do_time': time.time()}
    mw.writeFile(lock_file, json.dumps(lock_data))
    (yes, info) = get_newest_tid()
    if yes:
        url = 'https://bbs.midoks.me/thread-' + info['tid'] + '-1-1.html'
        keyboard = [[types.InlineKeyboardButton(text=info['subject'], url=url)], [types.InlineKeyboardButton(text='论坛', url='https://bbs.midoks.me'), types.InlineKeyboardButton(text='搜索', url='https://bbs.midoks.me/search.php')]]
        markup = types.InlineKeyboardMarkup(keyboard)
        bot.send_message(chat_id, '由【' + info['author'] + '】发帖!', reply_markup=markup)

def run(bot):
    if False:
        i = 10
        return i + 15
    try:
        send_msg(bot, 'push_bbs_newest_tid', 300)
    except Exception as e:
        writeLog('-----push_bbs_newest_tid error start -------')
        print(mw.getTracebackInfo())
        writeLog('-----push_bbs_newest_tid error start -------')
if __name__ == '__main__':
    print(get_newest_tid())