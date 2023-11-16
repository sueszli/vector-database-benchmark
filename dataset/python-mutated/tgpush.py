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

def getPluginName():
    if False:
        i = 10
        return i + 15
    return 'tgbot'

def getPluginDir():
    if False:
        while True:
            i = 10
    return mw.getPluginDir() + '/' + getPluginName()

def getServerDir():
    if False:
        return 10
    return mw.getServerDir() + '/' + getPluginName()
sys.path.append(getServerDir() + '/extend')

def getConfigData():
    if False:
        i = 10
        return i + 15
    cfg_path = getServerDir() + '/data.cfg'
    if not os.path.exists(cfg_path):
        mw.writeFile(cfg_path, '{}')
    t = mw.readFile(cfg_path)
    return json.loads(t)

def writeConf(data):
    if False:
        print('Hello World!')
    cfg_path = getServerDir() + '/data.cfg'
    mw.writeFile(cfg_path, json.dumps(data))
    return True

def getExtCfg():
    if False:
        print('Hello World!')
    cfg_path = getServerDir() + '/extend.cfg'
    if not os.path.exists(cfg_path):
        mw.writeFile(cfg_path, '{}')
    t = mw.readFile(cfg_path)
    return json.loads(t)

def getStartExtCfgByTag(tag='push'):
    if False:
        return 10
    elist = getExtCfg()
    rlist = []
    for x in elist:
        if x['tag'] == tag and x['status'] == 'start':
            rlist.append(x)
    return rlist

def writeLog(log_str):
    if False:
        print('Hello World!')
    if __name__ == '__main__':
        print(log_str)
    now = mw.getDateFromNow()
    log_file = getServerDir() + '/push.log'
    mw.writeFileLog(now + ':' + log_str, log_file, limit_size=5 * 1024)
    return True
cfg = getConfigData()
while True:
    cfg = getConfigData()
    if 'bot' in cfg and 'app_token' in cfg['bot']:
        if cfg['bot']['app_token'] != '' and cfg['bot']['app_token'] != 'app_token':
            break
    writeLog('等待输入配置,填写app_token')
    time.sleep(3)
bot = telebot.TeleBot(cfg['bot']['app_token'])

def runBotPushTask():
    if False:
        while True:
            i = 10
    plist = getStartExtCfgByTag('push')
    for p in plist:
        try:
            script = p['name'].split('.')[0]
            __import__(script).run(bot)
        except Exception as e:
            writeLog('-----runBotPushTask error start -------')
            writeLog(mw.getTracebackInfo())
            writeLog('-----runBotPushTask error end -------')

def botPush():
    if False:
        for i in range(10):
            print('nop')
    while True:
        runBotPushTask()
        time.sleep(1)

def runBotPushOtherTask():
    if False:
        while True:
            i = 10
    plist = getStartExtCfgByTag('other')
    for p in plist:
        try:
            script = p['name'].split('.')[0]
            __import__(script).run(bot)
        except Exception as e:
            writeLog('-----runBotPushOtherTask error start -------')
            writeLog(mw.getTracebackInfo())
            writeLog('-----runBotPushOtherTask error end -------')

def botPushOther():
    if False:
        while True:
            i = 10
    while True:
        runBotPushOtherTask()
        time.sleep(1)
if __name__ == '__main__':
    botPushTask = threading.Thread(target=botPush)
    botPushTask.start()
    botPushOtherTask = threading.Thread(target=botPushOther)
    botPushOtherTask.start()