import sys
import io
import os
import time
import re
import json
import base64
import threading
import asyncio
import logging
'\ncd /www/server/mdserver-web && source bin/activate  &&  python3 /www/server/tgclient/tgclient.py\n'
from telethon import TelegramClient
sys.path.append(os.getcwd() + '/class/core')
import mw
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def getPluginName():
    if False:
        for i in range(10):
            print('nop')
    return 'tgclient'

def getPluginDir():
    if False:
        print('Hello World!')
    return mw.getPluginDir() + '/' + getPluginName()

def getServerDir():
    if False:
        return 10
    return mw.getServerDir() + '/' + getPluginName()
sys.path.append(getServerDir() + '/extend')

def getConfigData():
    if False:
        for i in range(10):
            print('nop')
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
        while True:
            i = 10
    cfg_path = getServerDir() + '/extend.cfg'
    if not os.path.exists(cfg_path):
        mw.writeFile(cfg_path, '{}')
    t = mw.readFile(cfg_path)
    return json.loads(t)

def getStartExtCfgByTag(tag='push'):
    if False:
        print('Hello World!')
    elist = getExtCfg()
    rlist = []
    for x in elist:
        if x['tag'] == tag and x['status'] == 'start':
            rlist.append(x)
    return rlist

def writeLog(log_str):
    if False:
        i = 10
        return i + 15
    if __name__ == '__main__':
        print(log_str)
    now = mw.getDateFromNow()
    log_file = getServerDir() + '/task.log'
    mw.writeFileLog(now + ':' + log_str, log_file, limit_size=5 * 1024)
    return True
cfg = getConfigData()
while True:
    cfg = getConfigData()
    if 'bot' in cfg and 'api_id' in cfg['bot']:
        if cfg['bot']['api_id'] != '' and cfg['bot']['api_id'] != 'api_id':
            break
        if cfg['bot']['api_hash'] != '' and cfg['bot']['api_hash'] != 'api_hash':
            break
    writeLog('等待输入配置,api_id')
    time.sleep(3)
client = TelegramClient('mdioks', cfg['bot']['api_id'], cfg['bot']['api_hash'])

async def plugins_run_task():
    plist = getStartExtCfgByTag('client')
    for p in plist:
        try:
            script = p['name'].split('.')[0]
            await __import__(script).run(client)
        except Exception as e:
            writeLog('----- client error start -------')
            writeLog(mw.getTracebackInfo())
            writeLog('----- client error end -------')

async def plugins_run():
    while True:
        await plugins_run_task()
        time.sleep(1)

async def main(loop):
    await client.start()
    writeLog('creating plugins_run task.')
    task = loop.create_task(plugins_run())
    await task
    writeLog('It works.')
    await client.run_until_disconnected()
    task.cancel()
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(loop))