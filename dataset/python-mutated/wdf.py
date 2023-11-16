from __future__ import print_function
import os
import requests
import re
import time
import xml.dom.minidom
import json
import sys
import math
import subprocess
import ssl
import threading
DEBUG = False
MAX_GROUP_NUM = 2
INTERFACE_CALLING_INTERVAL = 5
MAX_PROGRESS_LEN = 50
QRImagePath = os.path.join(os.getcwd(), 'qrcode.jpg')
tip = 0
uuid = ''
base_uri = ''
redirect_uri = ''
push_uri = ''
skey = ''
wxsid = ''
wxuin = ''
pass_ticket = ''
deviceId = 'e000000000000000'
BaseRequest = {}
ContactList = []
My = []
SyncKey = []
try:
    xrange
    range = xrange
except:
    pass

def responseState(func, BaseResponse):
    if False:
        while True:
            i = 10
    ErrMsg = BaseResponse['ErrMsg']
    Ret = BaseResponse['Ret']
    if DEBUG or Ret != 0:
        print('func: %s, Ret: %d, ErrMsg: %s' % (func, Ret, ErrMsg))
    if Ret != 0:
        return False
    return True

def getUUID():
    if False:
        while True:
            i = 10
    global uuid
    url = 'https://login.weixin.qq.com/jslogin'
    params = {'appid': 'wx782c26e4c19acffb', 'fun': 'new', 'lang': 'zh_CN', '_': int(time.time())}
    r = myRequests.get(url=url, params=params)
    r.encoding = 'utf-8'
    data = r.text
    regx = 'window.QRLogin.code = (\\d+); window.QRLogin.uuid = "(\\S+?)"'
    pm = re.search(regx, data)
    code = pm.group(1)
    uuid = pm.group(2)
    if code == '200':
        return True
    return False

def showQRImage():
    if False:
        while True:
            i = 10
    global tip
    url = 'https://login.weixin.qq.com/qrcode/' + uuid
    params = {'t': 'webwx', '_': int(time.time())}
    r = myRequests.get(url=url, params=params)
    tip = 1
    f = open(QRImagePath, 'wb')
    f.write(r.content)
    f.close()
    time.sleep(1)
    if sys.platform.find('darwin') >= 0:
        subprocess.call(['open', QRImagePath])
    elif sys.platform.find('linux') >= 0:
        subprocess.call(['xdg-open', QRImagePath])
    else:
        os.startfile(QRImagePath)
    print('请使用微信扫描二维码以登录')

def waitForLogin():
    if False:
        i = 10
        return i + 15
    global tip, base_uri, redirect_uri, push_uri
    url = 'https://login.weixin.qq.com/cgi-bin/mmwebwx-bin/login?tip=%s&uuid=%s&_=%s' % (tip, uuid, int(time.time()))
    r = myRequests.get(url=url)
    r.encoding = 'utf-8'
    data = r.text
    regx = 'window.code=(\\d+);'
    pm = re.search(regx, data)
    code = pm.group(1)
    if code == '201':
        print('成功扫描,请在手机上点击确认以登录')
        tip = 0
    elif code == '200':
        print('正在登录...')
        regx = 'window.redirect_uri="(\\S+?)";'
        pm = re.search(regx, data)
        redirect_uri = pm.group(1) + '&fun=new'
        base_uri = redirect_uri[:redirect_uri.rfind('/')]
        services = [('wx2.qq.com', 'webpush2.weixin.qq.com'), ('qq.com', 'webpush.weixin.qq.com'), ('web1.wechat.com', 'webpush1.wechat.com'), ('web2.wechat.com', 'webpush2.wechat.com'), ('wechat.com', 'webpush.wechat.com'), ('web1.wechatapp.com', 'webpush1.wechatapp.com')]
        push_uri = base_uri
        for (searchUrl, pushUrl) in services:
            if base_uri.find(searchUrl) >= 0:
                push_uri = 'https://%s/cgi-bin/mmwebwx-bin' % pushUrl
                break
        if sys.platform.find('darwin') >= 0:
            os.system('osascript -e \'quit app "Preview"\'')
    elif code == '408':
        pass
    return code

def login():
    if False:
        return 10
    global skey, wxsid, wxuin, pass_ticket, BaseRequest
    r = myRequests.get(url=redirect_uri)
    r.encoding = 'utf-8'
    data = r.text
    doc = xml.dom.minidom.parseString(data)
    root = doc.documentElement
    for node in root.childNodes:
        if node.nodeName == 'skey':
            skey = node.childNodes[0].data
        elif node.nodeName == 'wxsid':
            wxsid = node.childNodes[0].data
        elif node.nodeName == 'wxuin':
            wxuin = node.childNodes[0].data
        elif node.nodeName == 'pass_ticket':
            pass_ticket = node.childNodes[0].data
    if not all((skey, wxsid, wxuin, pass_ticket)):
        return False
    BaseRequest = {'Uin': int(wxuin), 'Sid': wxsid, 'Skey': skey, 'DeviceID': deviceId}
    return True

def webwxinit():
    if False:
        i = 10
        return i + 15
    url = base_uri + '/webwxinit?pass_ticket=%s&skey=%s&r=%s' % (pass_ticket, skey, int(time.time()))
    params = {'BaseRequest': BaseRequest}
    headers = {'content-type': 'application/json; charset=UTF-8'}
    r = myRequests.post(url=url, data=json.dumps(params), headers=headers)
    r.encoding = 'utf-8'
    data = r.json()
    if DEBUG:
        f = open(os.path.join(os.getcwd(), 'webwxinit.json'), 'wb')
        f.write(r.content)
        f.close()
    global ContactList, My, SyncKey
    dic = data
    ContactList = dic['ContactList']
    My = dic['User']
    SyncKey = dic['SyncKey']
    state = responseState('webwxinit', dic['BaseResponse'])
    return state

def webwxgetcontact():
    if False:
        return 10
    url = base_uri + '/webwxgetcontact?pass_ticket=%s&skey=%s&r=%s' % (pass_ticket, skey, int(time.time()))
    headers = {'content-type': 'application/json; charset=UTF-8'}
    r = myRequests.post(url=url, headers=headers)
    r.encoding = 'utf-8'
    data = r.json()
    if DEBUG:
        f = open(os.path.join(os.getcwd(), 'webwxgetcontact.json'), 'wb')
        f.write(r.content)
        f.close()
    dic = data
    MemberList = dic['MemberList']
    SpecialUsers = ['newsapp', 'fmessage', 'filehelper', 'weibo', 'qqmail', 'tmessage', 'qmessage', 'qqsync', 'floatbottle', 'lbsapp', 'shakeapp', 'medianote', 'qqfriend', 'readerapp', 'blogapp', 'facebookapp', 'masssendapp', 'meishiapp', 'feedsapp', 'voip', 'blogappweixin', 'weixin', 'brandsessionholder', 'weixinreminder', 'wxid_novlwrv3lqwv11', 'gh_22b87fa7cb3c', 'officialaccounts', 'notification_messages', 'wxitil', 'userexperience_alarm']
    for i in range(len(MemberList) - 1, -1, -1):
        Member = MemberList[i]
        if Member['VerifyFlag'] & 8 != 0:
            MemberList.remove(Member)
        elif Member['UserName'] in SpecialUsers:
            MemberList.remove(Member)
        elif Member['UserName'].find('@@') != -1:
            MemberList.remove(Member)
        elif Member['UserName'] == My['UserName']:
            MemberList.remove(Member)
    return MemberList

def createChatroom(UserNames):
    if False:
        while True:
            i = 10
    MemberList = [{'UserName': UserName} for UserName in UserNames]
    url = base_uri + '/webwxcreatechatroom?pass_ticket=%s&r=%s' % (pass_ticket, int(time.time()))
    params = {'BaseRequest': BaseRequest, 'MemberCount': len(MemberList), 'MemberList': MemberList, 'Topic': ''}
    headers = {'content-type': 'application/json; charset=UTF-8'}
    r = myRequests.post(url=url, data=json.dumps(params), headers=headers)
    r.encoding = 'utf-8'
    data = r.json()
    dic = data
    ChatRoomName = dic['ChatRoomName']
    MemberList = dic['MemberList']
    DeletedList = []
    BlockedList = []
    for Member in MemberList:
        if Member['MemberStatus'] == 4:
            DeletedList.append(Member['UserName'])
        elif Member['MemberStatus'] == 3:
            BlockedList.append(Member['UserName'])
    state = responseState('createChatroom', dic['BaseResponse'])
    return (ChatRoomName, DeletedList, BlockedList)

def deleteMember(ChatRoomName, UserNames):
    if False:
        i = 10
        return i + 15
    url = base_uri + '/webwxupdatechatroom?fun=delmember&pass_ticket=%s' % pass_ticket
    params = {'BaseRequest': BaseRequest, 'ChatRoomName': ChatRoomName, 'DelMemberList': ','.join(UserNames)}
    headers = {'content-type': 'application/json; charset=UTF-8'}
    r = myRequests.post(url=url, data=json.dumps(params), headers=headers)
    r.encoding = 'utf-8'
    data = r.json()
    dic = data
    state = responseState('deleteMember', dic['BaseResponse'])
    return state

def addMember(ChatRoomName, UserNames):
    if False:
        i = 10
        return i + 15
    url = base_uri + '/webwxupdatechatroom?fun=addmember&pass_ticket=%s' % pass_ticket
    params = {'BaseRequest': BaseRequest, 'ChatRoomName': ChatRoomName, 'AddMemberList': ','.join(UserNames)}
    headers = {'content-type': 'application/json; charset=UTF-8'}
    r = myRequests.post(url=url, data=json.dumps(params), headers=headers)
    r.encoding = 'utf-8'
    data = r.json()
    dic = data
    MemberList = dic['MemberList']
    DeletedList = []
    BlockedList = []
    for Member in MemberList:
        if Member['MemberStatus'] == 4:
            DeletedList.append(Member['UserName'])
        elif Member['MemberStatus'] == 3:
            BlockedList.append(Member['UserName'])
    state = responseState('addMember', dic['BaseResponse'])
    return (DeletedList, BlockedList)

def syncKey():
    if False:
        print('Hello World!')
    SyncKeyItems = ['%s_%s' % (item['Key'], item['Val']) for item in SyncKey['List']]
    SyncKeyStr = '|'.join(SyncKeyItems)
    return SyncKeyStr

def syncCheck():
    if False:
        return 10
    url = push_uri + '/synccheck?'
    params = {'skey': BaseRequest['Skey'], 'sid': BaseRequest['Sid'], 'uin': BaseRequest['Uin'], 'deviceId': BaseRequest['DeviceID'], 'synckey': syncKey(), 'r': int(time.time())}
    r = myRequests.get(url=url, params=params)
    r.encoding = 'utf-8'
    data = r.text
    regx = 'window.synccheck={retcode:"(\\d+)",selector:"(\\d+)"}'
    pm = re.search(regx, data)
    retcode = pm.group(1)
    selector = pm.group(2)
    return selector

def webwxsync():
    if False:
        for i in range(10):
            print('nop')
    global SyncKey
    url = base_uri + '/webwxsync?lang=zh_CN&skey=%s&sid=%s&pass_ticket=%s' % (BaseRequest['Skey'], BaseRequest['Sid'], quote_plus(pass_ticket))
    params = {'BaseRequest': BaseRequest, 'SyncKey': SyncKey, 'rr': ~int(time.time())}
    headers = {'content-type': 'application/json; charset=UTF-8'}
    r = myRequests.post(url=url, data=json.dumps(params))
    r.encoding = 'utf-8'
    data = r.json()
    dic = data
    SyncKey = dic['SyncKey']
    state = responseState('webwxsync', dic['BaseResponse'])
    return state

def heartBeatLoop():
    if False:
        for i in range(10):
            print('nop')
    while True:
        selector = syncCheck()
        if selector != '0':
            webwxsync()
        time.sleep(1)

def main():
    if False:
        while True:
            i = 10
    global myRequests
    if hasattr(ssl, '_create_unverified_context'):
        ssl._create_default_https_context = ssl._create_unverified_context
    headers = {'User-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.125 Safari/537.36'}
    myRequests = requests.Session()
    myRequests.headers.update(headers)
    if not getUUID():
        print('获取uuid失败')
        return
    print('正在获取二维码图片...')
    showQRImage()
    while waitForLogin() != '200':
        pass
    os.remove(QRImagePath)
    if not login():
        print('登录失败')
        return
    if not webwxinit():
        print('初始化失败')
        return
    MemberList = webwxgetcontact()
    print('开启心跳线程')
    threading.Thread(target=heartBeatLoop)
    MemberCount = len(MemberList)
    print('通讯录共%s位好友' % MemberCount)
    ChatRoomName = ''
    result = []
    d = {}
    for Member in MemberList:
        d[Member['UserName']] = (Member['NickName'], Member['RemarkName'])
    print('开始查找...')
    group_num = int(math.ceil(MemberCount / float(MAX_GROUP_NUM)))
    for i in range(0, group_num):
        UserNames = []
        for j in range(0, MAX_GROUP_NUM):
            if i * MAX_GROUP_NUM + j >= MemberCount:
                break
            Member = MemberList[i * MAX_GROUP_NUM + j]
            UserNames.append(Member['UserName'])
        if ChatRoomName == '':
            (ChatRoomName, DeletedList, BlockedList) = createChatroom(UserNames)
        else:
            (DeletedList, BlockedList) = addMember(ChatRoomName, UserNames)
        DeletedCount = len(DeletedList)
        if DeletedCount > 0:
            result += DeletedList
        deleteMember(ChatRoomName, UserNames)
        progress = MAX_PROGRESS_LEN * (i + 1) / group_num
        print('[', '#' * int(progress), '-' * int(MAX_PROGRESS_LEN - progress), ']', end=' ')
        print('新发现你被%d人删除' % DeletedCount)
        for i in range(DeletedCount):
            if d[DeletedList[i]][1] != '':
                print('%s(%s)' % (d[DeletedList[i]][0], d[DeletedList[i]][1]))
            else:
                print(d[DeletedList[i]][0])
        if i != group_num - 1:
            print('正在继续查找,请耐心等待...')
            time.sleep(INTERFACE_CALLING_INTERVAL)
    print('\n结果汇总完毕,20s后可重试...')
    resultNames = []
    for r in result:
        if d[r][1] != '':
            resultNames.append('%s(%s)' % (d[r][0], d[r][1]))
        else:
            resultNames.append(d[r][0])
    print('---------- 被删除的好友列表(共%d人) ----------' % len(result))
    resultNames = list(map(lambda x: re.sub('<span.+/span>', '', x), resultNames))
    if len(resultNames):
        print('\n'.join(resultNames))
    else:
        print('无')
    print('---------------------------------------------')

class UnicodeStreamFilter:

    def __init__(self, target):
        if False:
            print('Hello World!')
        self.target = target
        self.encoding = 'utf-8'
        self.errors = 'replace'
        self.encode_to = self.target.encoding

    def write(self, s):
        if False:
            for i in range(10):
                print('nop')
        if type(s) == str:
            try:
                s = s.decode('utf-8')
            except:
                pass
        s = s.encode(self.encode_to, self.errors).decode(self.encode_to)
        self.target.write(s)
if sys.stdout.encoding == 'cp936':
    sys.stdout = UnicodeStreamFilter(sys.stdout)
if __name__ == '__main__':
    print('本程序的查询结果可能会引起一些心理上的不适,请小心使用...')
    print('1小时内只能使用一次，否则会因操作繁忙阻止建群')
    main()
    print('回车键退出...')
    input()