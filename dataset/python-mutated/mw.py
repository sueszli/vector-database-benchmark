import os
import sys
import time
import string
import json
import hashlib
import shlex
import datetime
import subprocess
import glob
import base64
import re
from random import Random
import db

def execShell(cmdstring, cwd=None, timeout=None, shell=True):
    if False:
        for i in range(10):
            print('nop')
    if shell:
        cmdstring_list = cmdstring
    else:
        cmdstring_list = shlex.split(cmdstring)
    if timeout:
        end_time = datetime.datetime.now() + datetime.timedelta(seconds=timeout)
    sub = subprocess.Popen(cmdstring_list, cwd=cwd, stdin=subprocess.PIPE, shell=shell, bufsize=4096, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while sub.poll() is None:
        time.sleep(0.1)
        if timeout:
            if end_time <= datetime.datetime.now():
                raise Exception('Timeout：%s' % cmdstring)
    if sys.version_info[0] == 2:
        return sub.communicate()
    data = sub.communicate()
    if isinstance(data[0], bytes):
        t1 = str(data[0], encoding='utf-8')
    if isinstance(data[1], bytes):
        t2 = str(data[1], encoding='utf-8')
    return (t1, t2)

def getTracebackInfo():
    if False:
        i = 10
        return i + 15
    import traceback
    errorMsg = traceback.format_exc()
    return errorMsg

def getRunDir():
    if False:
        i = 10
        return i + 15
    return os.getcwd()

def getRootDir():
    if False:
        return 10
    return os.path.dirname(os.path.dirname(getRunDir()))

def getPluginDir():
    if False:
        for i in range(10):
            print('nop')
    return getRunDir() + '/plugins'

def getPanelDataDir():
    if False:
        for i in range(10):
            print('nop')
    return getRunDir() + '/data'

def getPanelTmp():
    if False:
        for i in range(10):
            print('nop')
    return getRunDir() + '/tmp'

def getServerDir():
    if False:
        i = 10
        return i + 15
    return getRootDir() + '/server'

def getLogsDir():
    if False:
        while True:
            i = 10
    return getRootDir() + '/wwwlogs'

def getWwwDir():
    if False:
        i = 10
        return i + 15
    file = getRunDir() + '/data/site.pl'
    if os.path.exists(file):
        return readFile(file).strip()
    return getRootDir() + '/wwwroot'

def setWwwDir(wdir):
    if False:
        while True:
            i = 10
    file = getRunDir() + '/data/site.pl'
    return writeFile(file, wdir)

def getBackupDir():
    if False:
        i = 10
        return i + 15
    file = getRunDir() + '/data/backup.pl'
    if os.path.exists(file):
        return readFile(file).strip()
    return getRootDir() + '/backup'

def setBackupDir(bdir):
    if False:
        print('Hello World!')
    file = getRunDir() + '/data/backup.pl'
    return writeFile(file, bdir)

def getAcmeDir():
    if False:
        while True:
            i = 10
    acme = '/root/.acme.sh'
    if isAppleSystem():
        cmd = "who | sed -n '2, 1p' |awk '{print $1}'"
        user = execShell(cmd)[0].strip()
        acme = '/Users/' + user + '/.acme.sh'
    if not os.path.exists(acme):
        acme = '/.acme.sh'
    return acme

def getAcmeDomainDir(domain):
    if False:
        return 10
    acme_dir = getAcmeDir()
    acme_domain = acme_dir + '/' + domain
    acme_domain_ecc = acme_domain + '_ecc'
    if os.path.exists(acme_domain_ecc):
        acme_domain = acme_domain_ecc
    return acme_domain

def fileNameCheck(filename):
    if False:
        for i in range(10):
            print('nop')
    f_strs = [';', '&', '<', '>']
    for fs in f_strs:
        if filename.find(fs) != -1:
            return False
    return True

def triggerTask():
    if False:
        return 10
    isTask = getRunDir() + '/tmp/panelTask.pl'
    writeFile(isTask, 'True')

def systemdCfgDir():
    if False:
        print('Hello World!')
    cfg_dir = '/lib/systemd/system'
    if os.path.exists(cfg_dir):
        return cfg_dir
    cfg_dir = '/usr/lib/systemd/system'
    if os.path.exists(cfg_dir):
        return cfg_dir
    return '/tmp'

def getSslCrt():
    if False:
        i = 10
        return i + 15
    if os.path.exists('/etc/ssl/certs/ca-certificates.crt'):
        return '/etc/ssl/certs/ca-certificates.crt'
    if os.path.exists('/etc/pki/tls/certs/ca-bundle.crt'):
        return '/etc/pki/tls/certs/ca-bundle.crt'
    return ''

def getOs():
    if False:
        print('Hello World!')
    return sys.platform

def getOsName():
    if False:
        while True:
            i = 10
    cmd = 'cat /etc/*-release | grep PRETTY_NAME |awk -F = \'{print $2}\' | awk -F \'"\' \'{print $2}\'| awk \'{print $1}\''
    data = execShell(cmd)
    return data[0].strip().lower()

def getOsID():
    if False:
        print('Hello World!')
    cmd = 'cat /etc/*-release | grep VERSION_ID | awk -F = \'{print $2}\' | awk -F \'"\' \'{print $2}\''
    data = execShell(cmd)
    return data[0].strip()

def getFileSuffix(file):
    if False:
        for i in range(10):
            print('nop')
    tmp = file.split('.')
    ext = tmp[len(tmp) - 1]
    return ext

def isAppleSystem():
    if False:
        for i in range(10):
            print('nop')
    if getOs() == 'darwin':
        return True
    return False

def isDebugMode():
    if False:
        while True:
            i = 10
    if isAppleSystem():
        return True
    debugPath = getRunDir() + '/data/debug.pl'
    if os.path.exists(debugPath):
        return True
    return False

def isNumber(s):
    if False:
        print('Hello World!')
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def deleteFile(file):
    if False:
        i = 10
        return i + 15
    if os.path.exists(file):
        os.remove(file)

def isInstalledWeb():
    if False:
        i = 10
        return i + 15
    path = getServerDir() + '/openresty/nginx/sbin/nginx'
    if os.path.exists(path):
        return True
    return False

def isIpAddr(ip):
    if False:
        while True:
            i = 10
    check_ip = re.compile('^(1\\d{2}|2[0-4]\\d|25[0-5]|[1-9]\\d|[1-9])\\.(1\\d{2}|2[0-4]\\d|25[0-5]|[1-9]\\d|\\d)\\.(1\\d{2}|2[0-4]\\d|25[0-5]|[1-9]\\d|\\d)\\.(1\\d{2}|2[0-4]\\d|25[0-5]|[1-9]\\d|\\d)$')
    if check_ip.match(ip):
        return True
    else:
        return False

def getWebStatus():
    if False:
        for i in range(10):
            print('nop')
    pid = getServerDir() + '/openresty/nginx/logs/nginx.pid'
    if os.path.exists(pid):
        return True
    return False

def restartWeb():
    if False:
        while True:
            i = 10
    return opWeb('reload')

def opWeb(method):
    if False:
        print('Hello World!')
    if not isInstalledWeb():
        return False
    systemd = '/lib/systemd/system/openresty.service'
    if os.path.exists(systemd):
        execShell('systemctl ' + method + ' openresty')
        return True
    initd = getServerDir() + '/openresty/init.d/openresty'
    if os.path.exists(initd):
        execShell(initd + ' ' + method)
        return True
    return False

def opLuaMake(cmd_name):
    if False:
        for i in range(10):
            print('nop')
    path = getServerDir() + '/web_conf/nginx/lua/lua.conf'
    root_dir = getServerDir() + '/web_conf/nginx/lua/' + cmd_name
    dst_path = getServerDir() + '/web_conf/nginx/lua/' + cmd_name + '.lua'
    def_path = getServerDir() + '/web_conf/nginx/lua/empty.lua'
    if not os.path.exists(root_dir):
        execShell('mkdir -p ' + root_dir)
    files = []
    for fl in os.listdir(root_dir):
        suffix = getFileSuffix(fl)
        if suffix != 'lua':
            continue
        flpath = os.path.join(root_dir, fl)
        files.append(flpath)
    if len(files) > 0:
        def_path = dst_path
        content = ''
        for f in files:
            t = readFile(f)
            f_base = os.path.basename(f)
            content += '-- ' + '*' * 20 + ' ' + f_base + ' start ' + '*' * 20 + '\n'
            content += t
            content += '\n' + '-- ' + '*' * 20 + ' ' + f_base + ' end ' + '*' * 20 + '\n'
        writeFile(dst_path, content)
    elif os.path.exists(dst_path):
        os.remove(dst_path)
    conf = readFile(path)
    conf = re.sub(cmd_name + ' (.*);', cmd_name + ' ' + def_path + ';', conf)
    writeFile(path, conf)

def opLuaInitFile():
    if False:
        for i in range(10):
            print('nop')
    opLuaMake('init_by_lua_file')

def opLuaInitWorkerFile():
    if False:
        print('Hello World!')
    opLuaMake('init_worker_by_lua_file')

def opLuaInitAccessFile():
    if False:
        print('Hello World!')
    opLuaMake('access_by_lua_file')

def opLuaMakeAll():
    if False:
        while True:
            i = 10
    opLuaInitFile()
    opLuaInitWorkerFile()
    opLuaInitAccessFile()

def restartMw():
    if False:
        return 10
    import system_api
    system_api.system_api().restartMw()

def checkWebConfig():
    if False:
        return 10
    op_dir = getServerDir() + '/openresty/nginx'
    cmd = op_dir + '/sbin/nginx -t -c ' + op_dir + '/conf/nginx.conf'
    result = execShell(cmd)
    searchStr = 'test is successful'
    if result[1].find(searchStr) == -1:
        msg = getInfo('配置文件错误: {1}', (result[1],))
        writeLog('软件管理', msg)
        return result[1]
    return True

def M(table):
    if False:
        return 10
    sql = db.Sql()
    return sql.table(table)

def getPage(args, result='1,2,3,4,5,8'):
    if False:
        print('Hello World!')
    data = getPageObject(args, result)
    return data[0]

def getPageObject(args, result='1,2,3,4,5,8'):
    if False:
        print('Hello World!')
    import page
    page = page.Page()
    info = {}
    info['count'] = 0
    if 'count' in args:
        info['count'] = int(args['count'])
    info['row'] = 10
    if 'row' in args:
        info['row'] = int(args['row'])
    info['p'] = 1
    if 'p' in args:
        info['p'] = int(args['p'])
    info['uri'] = {}
    info['return_js'] = ''
    if 'tojs' in args:
        info['return_js'] = args['tojs']
    return (page.GetPage(info, result), page)

def md5(content):
    if False:
        while True:
            i = 10
    try:
        m = hashlib.md5()
        m.update(content.encode('utf-8'))
        return m.hexdigest()
    except Exception as ex:
        return False

def getFileMd5(filename):
    if False:
        return 10
    if not os.path.isfile(filename):
        return False
    myhash = hashlib.md5()
    f = file(filename, 'rb')
    while True:
        b = f.read(8096)
        if not b:
            break
        myhash.update(b)
    f.close()
    return myhash.hexdigest()

def getRandomString(length):
    if False:
        for i in range(10):
            print('nop')
    rnd_str = ''
    chars = 'AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789'
    chrlen = len(chars) - 1
    random = Random()
    for i in range(length):
        rnd_str += chars[random.randint(0, chrlen)]
    return rnd_str

def getUniqueId():
    if False:
        return 10
    '\n    根据时间生成唯一ID\n    :return:\n    '
    current_time = datetime.datetime.now()
    str_time = current_time.strftime('%Y%m%d%H%M%S%f')[:-3]
    unique_id = '{0}'.format(str_time)
    return unique_id

def getJson(data):
    if False:
        i = 10
        return i + 15
    import json
    return json.dumps(data)

def returnData(status, msg, data=None):
    if False:
        while True:
            i = 10
    return {'status': status, 'msg': msg, 'data': data}

def returnJson(status, msg, data=None):
    if False:
        for i in range(10):
            print('nop')
    if data == None:
        return getJson({'status': status, 'msg': msg})
    return getJson({'status': status, 'msg': msg, 'data': data})

def getLanguage():
    if False:
        return 10
    path = 'data/language.pl'
    if not os.path.exists(path):
        return 'Simplified_Chinese'
    return readFile(path).strip()

def getStaticJson(name='public'):
    if False:
        print('Hello World!')
    file = 'static/language/' + getLanguage() + '/' + name + '.json'
    if not os.path.exists(file):
        file = 'route/static/language/' + getLanguage() + '/' + name + '.json'
    return file

def returnMsg(status, msg, args=()):
    if False:
        for i in range(10):
            print('nop')
    pjson = getStaticJson('public')
    logMessage = json.loads(readFile(pjson))
    keys = logMessage.keys()
    if msg in keys:
        msg = logMessage[msg]
        for i in range(len(args)):
            rep = '{' + str(i + 1) + '}'
            msg = msg.replace(rep, args[i])
    return {'status': status, 'msg': msg, 'data': args}

def getInfo(msg, args=()):
    if False:
        while True:
            i = 10
    for i in range(len(args)):
        rep = '{' + str(i + 1) + '}'
        msg = msg.replace(rep, args[i])
    return msg

def getMsg(key, args=()):
    if False:
        while True:
            i = 10
    try:
        pjson = getStaticJson('public')
        logMessage = json.loads(pjson)
        keys = logMessage.keys()
        msg = None
        if key in keys:
            msg = logMessage[key]
            for i in range(len(args)):
                rep = '{' + str(i + 1) + '}'
                msg = msg.replace(rep, args[i])
        return msg
    except:
        return key

def getLan(key):
    if False:
        while True:
            i = 10
    pjson = getStaticJson('public')
    logMessage = json.loads(pjson)
    keys = logMessage.keys()
    msg = None
    if key in keys:
        msg = logMessage[key]
    return msg

def readFile(filename):
    if False:
        i = 10
        return i + 15
    try:
        fp = open(filename, 'r')
        fBody = fp.read()
        fp.close()
        return fBody
    except Exception as e:
        return False

def getDate():
    if False:
        i = 10
        return i + 15
    import time
    return time.strftime('%Y-%m-%d %X', time.localtime())

def getDateFromNow(tf_format='%Y-%m-%d %H:%M:%S', time_zone='Asia/Shanghai'):
    if False:
        for i in range(10):
            print('nop')
    import time
    os.environ['TZ'] = time_zone
    time.tzset()
    return time.strftime(tf_format, time.localtime())

def getDataFromInt(val):
    if False:
        for i in range(10):
            print('nop')
    time_format = '%Y-%m-%d %H:%M:%S'
    time_str = time.localtime(val)
    return time.strftime(time_format, time_str)

def writeLog(stype, msg, args=()):
    if False:
        i = 10
        return i + 15
    uid = 1
    try:
        from flask import session
        if 'uid' in session:
            uid = session['uid']
    except Exception as e:
        pass
    return writeDbLog(stype, msg, args, uid)

def writeFileLog(msg, path=None, limit_size=50 * 1024 * 1024, save_limit=3):
    if False:
        i = 10
        return i + 15
    log_file = getServerDir() + '/mdserver-web/logs/debug.log'
    if path != None:
        log_file = path
    if os.path.exists(log_file):
        size = os.path.getsize(log_file)
        if size > limit_size:
            log_file_rename = log_file + '_' + time.strftime('%Y-%m-%d_%H%M%S') + '.log'
            os.rename(log_file, log_file_rename)
            logs = sorted(glob.glob(log_file + '_*'))
            count = len(logs)
            save_limit = count - save_limit
            for i in range(count):
                if i > save_limit:
                    break
                os.remove(logs[i])
    f = open(log_file, 'ab+')
    msg += '\n'
    if __name__ == '__main__':
        print(msg)
    f.write(msg.encode('utf-8'))
    f.close()
    return True

def writeDbLog(stype, msg, args=(), uid=1):
    if False:
        return 10
    try:
        import time
        import db
        import json
        sql = db.Sql()
        mdate = time.strftime('%Y-%m-%d %X', time.localtime())
        wmsg = getInfo(msg, args)
        data = (stype, wmsg, uid, mdate)
        result = sql.table('logs').add('type,log,uid,addtime', data)
        return True
    except Exception as e:
        return False

def writeFile(filename, content, mode='w+'):
    if False:
        print('Hello World!')
    try:
        fp = open(filename, mode)
        fp.write(content)
        fp.close()
        return True
    except Exception as e:
        return False

def backFile(file, act=None):
    if False:
        return 10
    '\n        @name 备份配置文件\n        @param file 需要备份的文件\n        @param act 如果存在，则备份一份作为默认配置\n    '
    file_type = '_bak'
    if act:
        file_type = '_def'
    execShell('cp -p {0} {1}'.format(file, file + file_type))

def removeBackFile(file, act=None):
    if False:
        for i in range(10):
            print('nop')
    '\n        @name 删除备份配置文件\n        @param file 需要删除备份文件\n        @param act 如果存在，则还原默认配置\n    '
    file_type = '_bak'
    if act:
        file_type = '_def'
    execShell('rm -rf {0}'.format(file + file_type))

def restoreFile(file, act=None):
    if False:
        i = 10
        return i + 15
    '\n        @name 还原配置文件\n        @param file 需要还原的文件\n        @param act 如果存在，则还原默认配置\n    '
    file_type = '_bak'
    if act:
        file_type = '_def'
    execShell('cp -p {1} {0}'.format(file, file + file_type))

def enPunycode(domain):
    if False:
        for i in range(10):
            print('nop')
    if sys.version_info[0] == 2:
        domain = domain.encode('utf8')
    tmp = domain.split('.')
    newdomain = ''
    for dkey in tmp:
        if dkey == '*':
            continue
        match = re.search(u'[\x80-ÿ]+', dkey)
        if not match:
            match = re.search(u'[一-龥]+', dkey)
        if not match:
            newdomain += dkey + '.'
        elif sys.version_info[0] == 2:
            newdomain += 'xn--' + dkey.decode('utf-8').encode('punycode') + '.'
        else:
            newdomain += 'xn--' + dkey.encode('punycode').decode('utf-8') + '.'
    if tmp[0] == '*':
        newdomain = '*.' + newdomain
    return newdomain[0:-1]

def dePunycode(domain):
    if False:
        return 10
    tmp = domain.split('.')
    newdomain = ''
    for dkey in tmp:
        if dkey.find('xn--') >= 0:
            newdomain += dkey.replace('xn--', '').encode('utf-8').decode('punycode') + '.'
        else:
            newdomain += dkey + '.'
    return newdomain[0:-1]

def enCrypt(key, strings):
    if False:
        for i in range(10):
            print('nop')
    try:
        import base64
        _key = key.encode('utf-8')
        _key = base64.urlsafe_b64encode(_key)
        if type(strings) != bytes:
            strings = strings.encode('utf-8')
        import cryptography
        from cryptography.fernet import Fernet
        f = Fernet(_key)
        result = f.encrypt(strings)
        return result.decode('utf-8')
    except:
        writeFileLog(getTracebackInfo())
        return strings

def deCrypt(key, strings):
    if False:
        while True:
            i = 10
    try:
        import base64
        _key = key.encode('utf-8')
        _key = base64.urlsafe_b64encode(_key)
        if type(strings) != bytes:
            strings = strings.encode('utf-8')
        from cryptography.fernet import Fernet
        f = Fernet(_key)
        result = f.decrypt(strings).decode('utf-8')
        return result
    except:
        writeFileLog(getTracebackInfo())
        return strings

def enDoubleCrypt(key, strings):
    if False:
        while True:
            i = 10
    try:
        import base64
        _key = md5(key).encode('utf-8')
        _key = base64.urlsafe_b64encode(_key)
        if type(strings) != bytes:
            strings = strings.encode('utf-8')
        import cryptography
        from cryptography.fernet import Fernet
        f = Fernet(_key)
        result = f.encrypt(strings)
        return result.decode('utf-8')
    except:
        writeFileLog(getTracebackInfo())
        return strings

def deDoubleCrypt(key, strings):
    if False:
        return 10
    try:
        import base64
        _key = md5(key).encode('utf-8')
        _key = base64.urlsafe_b64encode(_key)
        if type(strings) != bytes:
            strings = strings.encode('utf-8')
        from cryptography.fernet import Fernet
        f = Fernet(_key)
        result = f.decrypt(strings).decode('utf-8')
        return result
    except:
        writeFileLog(getTracebackInfo())
        return strings

def aesEncrypt(data, key='ABCDEFGHIJKLMNOP', vi='0102030405060708'):
    if False:
        i = 10
        return i + 15
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    if not isinstance(data, bytes):
        data = data.encode()
    AES_CBC_KEY = key.encode()
    AES_CBC_IV = vi.encode()
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(data) + padder.finalize()
    cipher = Cipher(algorithms.AES(AES_CBC_KEY), modes.CBC(AES_CBC_IV), backend=default_backend())
    encryptor = cipher.encryptor()
    edata = encryptor.update(padded_data)
    return edata

def aesDecrypt(data, key='ABCDEFGHIJKLMNOP', vi='0102030405060708'):
    if False:
        while True:
            i = 10
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    if not isinstance(data, bytes):
        data = data.encode()
    AES_CBC_KEY = key.encode()
    AES_CBC_IV = vi.encode()
    cipher = Cipher(algorithms.AES(AES_CBC_KEY), modes.CBC(AES_CBC_IV), backend=default_backend())
    decryptor = cipher.decryptor()
    ddata = decryptor.update(data)
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    data = unpadder.update(ddata)
    try:
        uppadded_data = data + unpadder.finalize()
    except ValueError:
        raise Exception('无效的加密信息!')
    return uppadded_data

def aesEncrypt_Crypto(data, key, vi):
    if False:
        for i in range(10):
            print('nop')
    from Crypto.Cipher import AES
    cryptor = AES.new(key.encode('utf8'), AES.MODE_CBC, vi.encode('utf8'))
    zhmodel = re.compile(u'[一-\u9fff]')
    match = zhmodel.search(data)
    if match == None:
        add = 16 - len(data) % 16
        pad = lambda s: s + add * chr(add)
        data = pad(data)
        enctext = cryptor.encrypt(data.encode('utf8'))
    else:
        data = data.encode()
        add = 16 - len(data) % 16
        data = data + add * chr(add).encode()
        enctext = cryptor.encrypt(data)
    encodestrs = base64.b64encode(enctext).decode('utf8')
    return encodestrs

def aesDecrypt_Crypto(data, key, vi):
    if False:
        print('Hello World!')
    from crypto.Cipher import AES
    data = data.encode('utf8')
    encodebytes = base64.urlsafe_b64decode(data)
    cipher = AES.new(key.encode('utf8'), AES.MODE_CBC, vi.encode('utf8'))
    text_decrypted = cipher.decrypt(encodebytes)
    zhmodel = re.compile(u'[一-\u9fff]')
    match = zhmodel.search(text_decrypted)
    if match == False:
        unpad = lambda s: s[0:-s[-1]]
        text_decrypted = unpad(text_decrypted)
    text_decrypted = text_decrypted.decode('utf8').rstrip()
    return text_decrypted

def buildSoftLink(src, dst, force=False):
    if False:
        print('Hello World!')
    '\n    建立软连接\n    '
    if not os.path.exists(src):
        return False
    if os.path.exists(dst) and force:
        os.remove(dst)
    if not os.path.exists(dst):
        execShell('ln -sf "' + src + '" "' + dst + '"')
        return True
    return False

def HttpGet(url, timeout=10):
    if False:
        i = 10
        return i + 15
    '\n    发送GET请求\n    @url 被请求的URL地址(必需)\n    @timeout 超时时间默认60秒\n    return string\n    '
    if sys.version_info[0] == 2:
        try:
            import urllib2
            import ssl
            if sys.version_info[0] == 2:
                reload(urllib2)
                reload(ssl)
            try:
                ssl._create_default_https_context = ssl._create_unverified_context
            except:
                pass
            response = urllib2.urlopen(url, timeout=timeout)
            return response.read()
        except Exception as ex:
            return str(ex)
    else:
        try:
            import urllib.request
            import ssl
            try:
                ssl._create_default_https_context = ssl._create_unverified_context
            except:
                pass
            response = urllib.request.urlopen(url, timeout=timeout)
            result = response.read()
            if type(result) == bytes:
                result = result.decode('utf-8')
            return result
        except Exception as ex:
            return str(ex)

def HttpGet2(url, timeout):
    if False:
        print('Hello World!')
    import urllib.request
    try:
        import ssl
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
        except:
            pass
        req = urllib.request.urlopen(url, timeout=timeout)
        result = req.read().decode('utf-8')
        return result
    except Exception as e:
        return str(e)

def httpGet(url, timeout=10):
    if False:
        for i in range(10):
            print('nop')
    return HttpGet2(url, timeout)

def HttpPost(url, data, timeout=10):
    if False:
        print('Hello World!')
    '\n    发送POST请求\n    @url 被请求的URL地址(必需)\n    @data POST参数，可以是字符串或字典(必需)\n    @timeout 超时时间默认60秒\n    return string\n    '
    if sys.version_info[0] == 2:
        try:
            import urllib
            import urllib2
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            data = urllib.urlencode(data)
            req = urllib2.Request(url, data)
            response = urllib2.urlopen(req, timeout=timeout)
            return response.read()
        except Exception as ex:
            return str(ex)
    else:
        try:
            import urllib.request
            import ssl
            try:
                ssl._create_default_https_context = ssl._create_unverified_context
            except:
                pass
            data = urllib.parse.urlencode(data).encode('utf-8')
            req = urllib.request.Request(url, data)
            response = urllib.request.urlopen(req, timeout=timeout)
            result = response.read()
            if type(result) == bytes:
                result = result.decode('utf-8')
            return result
        except Exception as ex:
            return str(ex)

def httpPost(url, data, timeout=10):
    if False:
        while True:
            i = 10
    return HttpPost(url, data, timeout)

def writeSpeed(title, used, total, speed=0):
    if False:
        i = 10
        return i + 15
    if not title:
        data = {'title': None, 'progress': 0, 'total': 0, 'used': 0, 'speed': 0}
    else:
        progress = int(100.0 * used / total)
        data = {'title': title, 'progress': progress, 'total': total, 'used': used, 'speed': speed}
    writeFile('/tmp/panelSpeed.pl', json.dumps(data))
    return True

def getSpeed():
    if False:
        for i in range(10):
            print('nop')
    path = getRootDir()
    data = readFile(path + '/tmp/panelSpeed.pl')
    if not data:
        data = json.dumps({'title': None, 'progress': 0, 'total': 0, 'used': 0, 'speed': 0})
        writeFile(path + '/tmp/panelSpeed.pl', data)
    return json.loads(data)

def getLastLineBk(inputfile, lineNum):
    if False:
        i = 10
        return i + 15
    try:
        fp = open(inputfile, 'rb')
        lastLine = ''
        lines = fp.readlines()
        count = len(lines)
        if count > lineNum:
            num = lineNum
        else:
            num = count
        i = 1
        lastre = []
        for i in range(1, num + 1):
            n = -i
            try:
                lastLine = lines[n].decode('utf-8', 'ignore').strip()
            except Exception as e:
                lastLine = ''
            lastre.append(lastLine)
        fp.close()
        result = ''
        num -= 1
        while num >= 0:
            result += lastre[num] + '\n'
            num -= 1
        return result
    except Exception as e:
        return str(e)

def getLastLine(path, num, p=1):
    if False:
        return 10
    pyVersion = sys.version_info[0]
    try:
        import html
        if not os.path.exists(path):
            return ''
        start_line = (p - 1) * num
        count = start_line + num
        fp = open(path, 'rb')
        buf = ''
        fp.seek(0, 2)
        if fp.read(1) == '\n':
            fp.seek(0, 2)
        data = []
        b = True
        n = 0
        for i in range(count):
            while True:
                newline_pos = str.rfind(str(buf), '\n')
                pos = fp.tell()
                if newline_pos != -1:
                    if n >= start_line:
                        line = buf[newline_pos + 1:]
                        try:
                            data.insert(0, html.escape(line))
                        except Exception as e:
                            pass
                    buf = buf[:newline_pos]
                    n += 1
                    break
                else:
                    if pos == 0:
                        b = False
                        break
                    to_read = min(4096, pos)
                    fp.seek(-to_read, 1)
                    t_buf = fp.read(to_read)
                    if pyVersion == 3:
                        if type(t_buf) == bytes:
                            t_buf = t_buf.decode('utf-8', 'ignore').strip()
                    buf = t_buf + buf
                    fp.seek(-to_read, 1)
                    if pos - to_read == 0:
                        buf = '\n' + buf
            if not b:
                break
        fp.close()
    except Exception as e:
        return str(e)
    return '\n'.join(data)

def downloadFile(url, filename):
    if False:
        print('Hello World!')
    import urllib
    urllib.urlretrieve(url, filename=filename, reporthook=downloadHook)

def downloadHook(count, blockSize, totalSize):
    if False:
        return 10
    speed = {'total': totalSize, 'block': blockSize, 'count': count}
    print('%02d%%' % (100.0 * count * blockSize / totalSize))

def getLocalIpBack():
    if False:
        i = 10
        return i + 15
    try:
        import re
        filename = 'data/iplist.txt'
        ipaddress = readFile(filename)
        if not ipaddress or ipaddress == '127.0.0.1':
            import urllib
            url = 'http://pv.sohu.com/cityjson?ie=utf-8'
            req = urllib.request.urlopen(url, timeout=10)
            content = req.read().decode('utf-8')
            ipaddress = re.search('\\d+.\\d+.\\d+.\\d+', content).group(0)
            writeFile(filename, ipaddress)
        ipaddress = re.search('\\d+.\\d+.\\d+.\\d+', ipaddress).group(0)
        return ipaddress
    except Exception as ex:
        return '127.0.0.1'

def getClientIp():
    if False:
        while True:
            i = 10
    from flask import request
    return request.remote_addr.replace('::ffff:', '')

def getLocalIp():
    if False:
        for i in range(10):
            print('nop')
    filename = 'data/iplist.txt'
    try:
        ipaddress = readFile(filename)
        if not ipaddress or ipaddress == '127.0.0.1':
            cmd = 'curl --insecure -4 -sS --connect-timeout 5 -m 60 https://v6r.ipip.net/?format=text'
            ip = execShell(cmd)
            result = ip[0].strip()
            if result == '':
                raise Exception('ipv4 is empty!')
            writeFile(filename, result)
            return result
        return ipaddress
    except Exception as e:
        cmd = 'curl --insecure -6 -sS --connect-timeout 5 -m 60 https://v6r.ipip.net/?format=text'
        ip = execShell(cmd)
        result = ip[0].strip()
        if result == '':
            return '127.0.0.1'
        writeFile(filename, result)
        return result
    finally:
        pass
    return '127.0.0.1'

def inArray(arrays, searchStr):
    if False:
        print('Hello World!')
    for key in arrays:
        if key == searchStr:
            return True
    return False

def formatDate(format='%Y-%m-%d %H:%M:%S', times=None):
    if False:
        while True:
            i = 10
    if not times:
        times = int(time.time())
    time_local = time.localtime(times)
    return time.strftime(format, time_local)

def strfToTime(sdate):
    if False:
        i = 10
        return i + 15
    import time
    return time.strftime('%Y-%m-%d', time.strptime(sdate, '%b %d %H:%M:%S %Y %Z'))

def checkIp(ip):
    if False:
        i = 10
        return i + 15
    import re
    p = re.compile('^((25[0-5]|2[0-4]\\d|[01]?\\d\\d?)\\.){3}(25[0-5]|2[0-4]\\d|[01]?\\d\\d?)$')
    if p.match(ip):
        return True
    else:
        return False

def getHost(port=False):
    if False:
        i = 10
        return i + 15
    from flask import request
    host_tmp = request.headers.get('host')
    if not host_tmp:
        if request.url_root:
            tmp = re.findall('(https|http)://([\\w:\\.-]+)', request.url_root)
            if tmp:
                host_tmp = tmp[0][1]
    if not host_tmp:
        host_tmp = getLocalIp() + ':' + readFile('data/port.pl').strip()
    try:
        if host_tmp.find(':') == -1:
            host_tmp += ':80'
    except:
        host_tmp = '127.0.0.1:8888'
    h = host_tmp.split(':')
    if port:
        return h[-1]
    return ':'.join(h[0:-1])

def getClientIp():
    if False:
        while True:
            i = 10
    from flask import request
    return request.remote_addr.replace('::ffff:', '')

def checkDomainPanel():
    if False:
        while True:
            i = 10
    tmp = getHost()
    domain = readFile('data/bind_domain.pl')
    port = readFile('data/port.pl').strip()
    npid = getServerDir() + '/openresty/nginx/logs/nginx.pid'
    if not os.path.exists(npid):
        return False
    nconf = getServerDir() + '/web_conf/nginx/vhost/panel.conf'
    if os.path.exists(nconf):
        port = '80'
    if domain:
        client_ip = getClientIp()
        if client_ip in ['127.0.0.1', 'localhost', '::1']:
            return False
        if tmp.strip().lower() != domain.strip().lower():
            from flask import Flask, redirect, request, url_for
            to = 'http://' + domain + ':' + str(port)
            return redirect(to, code=302)
    return False

def createLinuxUser(user, group):
    if False:
        while True:
            i = 10
    execShell('groupadd {}'.format(group))
    execShell('useradd -s /sbin/nologin -g {} {}'.format(user, group))
    return True

def setOwn(filename, user, group=None):
    if False:
        print('Hello World!')
    if isAppleSystem():
        return True
    if not os.path.exists(filename):
        return False
    from pwd import getpwnam
    try:
        user_info = getpwnam(user)
        user = user_info.pw_uid
        if group:
            user_info = getpwnam(group)
        group = user_info.pw_gid
    except:
        if user == 'www':
            createLinuxUser(user)
        try:
            user_info = getpwnam('www')
        except:
            createLinuxUser(user)
            user_info = getpwnam('www')
        user = user_info.pw_uid
        group = user_info.pw_gid
    os.chown(filename, user, group)
    return True

def setMode(filename, mode):
    if False:
        print('Hello World!')
    if not os.path.exists(filename):
        return False
    mode = int(str(mode), 8)
    os.chmod(filename, mode)
    return True

def checkPort(port):
    if False:
        for i in range(10):
            print('nop')
    ports = ['21', '443', '888']
    if port in ports:
        return False
    intport = int(port)
    if intport < 1 or intport > 65535:
        return False
    return True

def getStrBetween(startStr, endStr, srcStr):
    if False:
        return 10
    start = srcStr.find(startStr)
    if start == -1:
        return None
    end = srcStr.find(endStr)
    if end == -1:
        return None
    return srcStr[start + 1:end]

def getCpuType():
    if False:
        return 10
    cpuType = ''
    if isAppleSystem():
        cmd = "system_profiler SPHardwareDataType | grep 'Processor Name' | awk -F ':' '{print $2}'"
        cpuinfo = execShell(cmd)
        return cpuinfo[0].strip()
    current_os = getOs()
    if current_os.startswith('freebsd'):
        cmd = "sysctl -a | egrep -i 'hw.model' | awk -F ':' '{print $2}'"
        cpuinfo = execShell(cmd)
        return cpuinfo[0].strip()
    cpuinfo = open('/proc/cpuinfo', 'r').read()
    rep = 'model\\s+name\\s+:\\s+(.+)'
    tmp = re.search(rep, cpuinfo, re.I)
    if tmp:
        cpuType = tmp.groups()[0]
    else:
        cpuinfo = execShell('LANG="en_US.UTF-8" && lscpu')[0]
        rep = 'Model\\s+name:\\s+(.+)'
        tmp = re.search(rep, cpuinfo, re.I)
        if tmp:
            cpuType = tmp.groups()[0]
    return cpuType

def isRestart():
    if False:
        print('Hello World!')
    num = M('tasks').where('status!=?', ('1',)).count()
    if num > 0:
        return False
    return True

def isUpdateLocalSoft():
    if False:
        while True:
            i = 10
    num = M('tasks').where('status!=?', ('1',)).count()
    if os.path.exists('mdserver-web.zip'):
        return True
    if num > 0:
        data = M('tasks').where('status!=?', ('1',)).field('id,type,execstr').limit('1').select()
        argv = data[0]['execstr'].split('|dl|')
        if data[0]['type'] == 'download' and argv[1] == 'mdserver-web.zip':
            return True
    return False

def hasPwd(password):
    if False:
        i = 10
        return i + 15
    import crypt
    return crypt.crypt(password, password)

def getTimeout(url):
    if False:
        return 10
    start = time.time()
    result = httpGet(url)
    if result != 'True':
        return False
    return int((time.time() - start) * 1000)

def makeConf():
    if False:
        return 10
    file = getRunDir() + '/data/json/config.json'
    if not os.path.exists(file):
        c = {}
        c['title'] = '老子面板'
        c['home'] = 'http://github/midoks/mdserver-web'
        c['recycle_bin'] = True
        c['template'] = 'default'
        writeFile(file, json.dumps(c))
        return c
    c = readFile(file)
    return json.loads(c)

def getConfig(k):
    if False:
        print('Hello World!')
    c = makeConf()
    return c[k]

def setConfig(k, v):
    if False:
        while True:
            i = 10
    c = makeConf()
    c[k] = v
    file = getRunDir() + '/data/json/config.json'
    return writeFile(file, json.dumps(c))

def getHostAddr():
    if False:
        i = 10
        return i + 15
    if os.path.exists('data/iplist.txt'):
        return readFile('data/iplist.txt').strip()
    return '127.0.0.1'

def setHostAddr(addr):
    if False:
        return 10
    file = getRunDir() + '/data/iplist.txt'
    return writeFile(file, addr)

def getHostPort():
    if False:
        return 10
    if os.path.exists('data/port.pl'):
        return readFile('data/port.pl').strip()
    return '7200'

def setHostPort(port):
    if False:
        i = 10
        return i + 15
    file = getRunDir() + '/data/port.pl'
    return writeFile(file, port)

def auth_decode(data):
    if False:
        while True:
            i = 10
    token = GetToken()
    if not token:
        return returnMsg(False, 'REQUEST_ERR')
    if token['access_key'] != data['btauth_key']:
        return returnMsg(False, 'REQUEST_ERR')
    import binascii
    import hashlib
    import urllib
    import hmac
    import json
    tdata = binascii.unhexlify(data['data'])
    signature = binascii.hexlify(hmac.new(token['secret_key'], tdata, digestmod=hashlib.sha256).digest())
    if signature != data['signature']:
        return returnMsg(False, 'REQUEST_ERR')
    return json.loads(urllib.unquote(tdata))

def auth_encode(data):
    if False:
        print('Hello World!')
    token = GetToken()
    pdata = {}
    if not token:
        return returnMsg(False, 'REQUEST_ERR')
    import binascii
    import hashlib
    import urllib
    import hmac
    import json
    tdata = urllib.quote(json.dumps(data))
    pdata['signature'] = binascii.hexlify(hmac.new(token['secret_key'], tdata, digestmod=hashlib.sha256).digest())
    pdata['btauth_key'] = token['access_key']
    pdata['data'] = binascii.hexlify(tdata)
    pdata['timestamp'] = time.time()
    return pdata

def checkToken(get):
    if False:
        print('Hello World!')
    tempFile = 'data/tempToken.json'
    if not os.path.exists(tempFile):
        return False
    import json
    import time
    tempToken = json.loads(readFile(tempFile))
    if time.time() > tempToken['timeout']:
        return False
    if get.token != tempToken['token']:
        return False
    return True

def checkInput(data):
    if False:
        while True:
            i = 10
    if not data:
        return data
    if type(data) != str:
        return data
    checkList = [{'d': '<', 'r': '＜'}, {'d': '>', 'r': '＞'}, {'d': "'", 'r': '‘'}, {'d': '"', 'r': '“'}, {'d': '&', 'r': '＆'}, {'d': '#', 'r': '＃'}, {'d': '<', 'r': '＜'}]
    for v in checkList:
        data = data.replace(v['d'], v['r'])
    return data

def checkCert(certPath='ssl/certificate.pem'):
    if False:
        return 10
    openssl = '/usr/bin/openssl'
    if not os.path.exists(openssl):
        openssl = '/usr/local/openssl/bin/openssl'
    if not os.path.exists(openssl):
        openssl = 'openssl'
    certPem = readFile(certPath)
    s = '\n-----BEGIN CERTIFICATE-----'
    tmp = certPem.strip().split(s)
    for tmp1 in tmp:
        if tmp1.find('-----BEGIN CERTIFICATE-----') == -1:
            tmp1 = s + tmp1
        writeFile(certPath, tmp1)
        result = execShell(openssl + ' x509 -in ' + certPath + ' -noout -subject')
        if result[1].find('-bash:') != -1:
            return True
        if len(result[1]) > 2:
            return False
        if result[0].find('error:') != -1:
            return False
    return True

def getPathSize(path):
    if False:
        i = 10
        return i + 15
    if not os.path.exists(path):
        return 0
    if not os.path.isdir(path):
        return os.path.getsize(path)
    size_total = 0
    for nf in os.walk(path):
        for f in nf[2]:
            filename = nf[0] + '/' + f
            size_total += os.path.getsize(filename)
    return size_total

def toSize(size):
    if False:
        while True:
            i = 10
    d = ('b', 'KB', 'MB', 'GB', 'TB')
    s = d[0]
    for b in d:
        if size < 1024:
            return str(round(size, 2)) + ' ' + b
        size = float(size) / 1024.0
        s = b
    return str(round(size, 2)) + ' ' + b

def getPathSuffix(path):
    if False:
        print('Hello World!')
    return os.path.splitext(path)[-1]

def getMacAddress():
    if False:
        i = 10
        return i + 15
    import uuid
    mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
    return ':'.join([mac[e:e + 2] for e in range(0, 11, 2)])

def get_string(t):
    if False:
        return 10
    if t != -1:
        max = 126
        m_types = [{'m': 122, 'n': 97}, {'m': 90, 'n': 65}, {'m': 57, 'n': 48}, {'m': 47, 'n': 32}, {'m': 64, 'n': 58}, {'m': 96, 'n': 91}, {'m': 125, 'n': 123}]
    else:
        max = 256
        t = 0
        m_types = [{'m': 255, 'n': 0}]
    arr = []
    for i in range(max):
        if i < m_types[t]['n'] or i > m_types[t]['m']:
            continue
        arr.append(chr(i))
    return arr

def get_string_find(t):
    if False:
        return 10
    if type(t) != list:
        t = [t]
    return_str = ''
    for s1 in t:
        return_str += get_string(int(s1[0]))[int(s1[1:])]
    return return_str

def get_string_arr(t):
    if False:
        for i in range(10):
            print('nop')
    s_arr = {}
    t_arr = []
    for s1 in t:
        for i in range(6):
            if not i in s_arr:
                s_arr[i] = get_string(i)
            for j in range(len(s_arr[i])):
                if s1 == s_arr[i][j]:
                    t_arr.append(str(i) + str(j))
    return t_arr

def strfDate(sdate):
    if False:
        return 10
    return time.strftime('%Y-%m-%d', time.strptime(sdate, '%Y%m%d%H%M%S'))

def getCertName(certPath):
    if False:
        return 10
    if not os.path.exists(certPath):
        return None
    try:
        import OpenSSL
        result = {}
        x509 = OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_PEM, readFile(certPath))
        issuer = x509.get_issuer()
        result['issuer'] = ''
        if hasattr(issuer, 'CN'):
            result['issuer'] = issuer.CN
        if not result['issuer']:
            is_key = [b'0', '0']
            issue_comp = issuer.get_components()
            if len(issue_comp) == 1:
                is_key = [b'CN', 'CN']
            for iss in issue_comp:
                if iss[0] in is_key:
                    result['issuer'] = iss[1].decode()
                    break
        if not result['issuer']:
            if hasattr(issuer, 'O'):
                result['issuer'] = issuer.O
        result['notAfter'] = strfDate(bytes.decode(x509.get_notAfter())[:-1])
        result['notBefore'] = strfDate(bytes.decode(x509.get_notBefore())[:-1])
        result['dns'] = []
        for i in range(x509.get_extension_count()):
            s_name = x509.get_extension(i)
            if s_name.get_short_name() in [b'subjectAltName', 'subjectAltName']:
                s_dns = str(s_name).split(',')
                for d in s_dns:
                    result['dns'].append(d.split(':')[1])
        subject = x509.get_subject().get_components()
        if len(subject) == 1:
            result['subject'] = subject[0][1].decode()
        elif not result['dns']:
            for sub in subject:
                if sub[0] == b'CN':
                    result['subject'] = sub[1].decode()
                    break
            if 'subject' in result:
                result['dns'].append(result['subject'])
        else:
            result['subject'] = result['dns'][0]
        result['endtime'] = int(int(time.mktime(time.strptime(result['notAfter'], '%Y-%m-%d')) - time.time()) / 86400)
        return result
    except Exception as e:
        writeFileLog(getTracebackInfo())
        return None

def createSSL():
    if False:
        while True:
            i = 10
    if os.path.exists('ssl/input.pl'):
        return True
    import OpenSSL
    key = OpenSSL.crypto.PKey()
    key.generate_key(OpenSSL.crypto.TYPE_RSA, 2048)
    cert = OpenSSL.crypto.X509()
    cert.set_serial_number(0)
    cert.get_subject().CN = getLocalIp()
    cert.set_issuer(cert.get_subject())
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(86400 * 3650)
    cert.set_pubkey(key)
    cert.sign(key, 'md5')
    cert_ca = OpenSSL.crypto.dump_certificate(OpenSSL.crypto.FILETYPE_PEM, cert)
    private_key = OpenSSL.crypto.dump_privatekey(OpenSSL.crypto.FILETYPE_PEM, key)
    if len(cert_ca) > 100 and len(private_key) > 100:
        writeFile('ssl/cert.pem', cert_ca, 'wb+')
        writeFile('ssl/private.pem', private_key, 'wb+')
        return True
    return False

def getSSHPort():
    if False:
        for i in range(10):
            print('nop')
    try:
        file = '/etc/ssh/sshd_config'
        conf = readFile(file)
        rep = '(#*)?Port\\s+([0-9]+)\\s*\n'
        port = re.search(rep, conf).groups(0)[1]
        return int(port)
    except:
        return 22

def getSSHStatus():
    if False:
        return 10
    if os.path.exists('/usr/bin/apt-get'):
        status = execShell("service ssh status | grep -P '(dead|stop)'")
    else:
        import system_api
        version = system_api.system_api().getSystemVersion()
        if version.find(' Mac ') != -1:
            return True
        if version.find(' 7.') != -1:
            status = execShell("systemctl status sshd.service | grep 'dead'")
        else:
            status = execShell("/etc/init.d/sshd status | grep -e 'stopped' -e '已停'")
    if len(status[0]) > 3:
        status = False
    else:
        status = True
    return status

def requestFcgiPHP(sock, uri, document_root='/tmp', method='GET', pdata=b''):
    if False:
        i = 10
        return i + 15
    sys.path.append(os.getcwd() + '/class/plugin')
    import fpm
    p = fpm.fpm(sock, document_root)
    if type(pdata) == dict:
        pdata = url_encode(pdata)
    result = p.load_url_public(uri, pdata, method)
    return result

def getMyORM():
    if False:
        for i in range(10):
            print('nop')
    '\n    获取MySQL资源的ORM\n    '
    sys.path.append(os.getcwd() + '/class/plugin')
    import orm
    o = orm.ORM()
    return o

def getMyORMDb():
    if False:
        return 10
    '\n    获取MySQL资源的ORM pip install mysqlclient==2.0.3 | pip install mysql-python\n    '
    sys.path.append(os.getcwd() + '/class/plugin')
    import ormDb
    o = ormDb.ORM()
    return o

def initNotifyConfig():
    if False:
        print('Hello World!')
    p = getNotifyPath()
    if not os.path.exists(p):
        writeFile(p, '{}')
    return True

def getNotifyPath():
    if False:
        for i in range(10):
            print('nop')
    path = 'data/notify.json'
    return path

def getNotifyData(is_parse=False):
    if False:
        while True:
            i = 10
    initNotifyConfig()
    notify_file = getNotifyPath()
    notify_data = readFile(notify_file)
    data = json.loads(notify_data)
    if is_parse:
        tag_list = ['tgbot', 'email']
        for t in tag_list:
            if t in data and 'cfg' in data[t]:
                data[t]['data'] = json.loads(deDoubleCrypt(t, data[t]['cfg']))
    return data

def writeNotify(data):
    if False:
        print('Hello World!')
    p = getNotifyPath()
    return writeFile(p, json.dumps(data))

def tgbotNotifyChatID():
    if False:
        i = 10
        return i + 15
    data = getNotifyData(True)
    if 'tgbot' in data and 'enable' in data['tgbot']:
        if data['tgbot']['enable']:
            t = data['tgbot']['data']
            return t['chat_id']
    return ''

def tgbotNotifyObject():
    if False:
        print('Hello World!')
    data = getNotifyData(True)
    if 'tgbot' in data and 'enable' in data['tgbot']:
        if data['tgbot']['enable']:
            t = data['tgbot']['data']
            import telebot
            bot = telebot.TeleBot(app_token)
            return (True, bot)
    return (False, None)

def tgbotNotifyMessage(app_token, chat_id, msg):
    if False:
        return 10
    import telebot
    bot = telebot.TeleBot(app_token)
    try:
        data = bot.send_message(chat_id, msg)
        return True
    except Exception as e:
        writeFileLog(str(e))
    return False

def tgbotNotifyHttpPost(app_token, chat_id, msg):
    if False:
        return 10
    try:
        url = 'https://api.telegram.org/bot' + app_token + '/sendMessage'
        post_data = {'chat_id': chat_id, 'text': msg}
        rdata = httpPost(url, post_data)
        return True
    except Exception as e:
        writeFileLog(str(e))
    return False

def tgbotNotifyTest(app_token, chat_id):
    if False:
        return 10
    msg = 'MW-通知验证测试OK'
    return tgbotNotifyHttpPost(app_token, chat_id, msg)

def emailNotifyMessage(data):
    if False:
        print('Hello World!')
    '\n    邮件通知\n    '
    sys.path.append(os.getcwd() + '/class/plugin')
    import memail
    try:
        if data['smtp_ssl'] == 'ssl':
            memail.sendSSL(data['smtp_host'], data['smtp_port'], data['username'], data['password'], data['to_mail_addr'], data['subject'], data['content'])
        else:
            memail.send(data['smtp_host'], data['smtp_port'], data['username'], data['password'], data['to_mail_addr'], data['subject'], data['content'])
        return True
    except Exception as e:
        print(getTracebackInfo())
    return False

def emailNotifyTest(data):
    if False:
        for i in range(10):
            print('nop')
    data['subject'] = 'MW通知测试'
    data['content'] = data['mail_test']
    return emailNotifyMessage(data)

def notifyMessageTry(msg, stype='common', trigger_time=300, is_write_log=True):
    if False:
        i = 10
        return i + 15
    lock_file = getPanelTmp() + '/notify_lock.json'
    if not os.path.exists(lock_file):
        writeFile(lock_file, '{}')
    lock_data = json.loads(readFile(lock_file))
    if stype in lock_data:
        diff_time = time.time() - lock_data[stype]['do_time']
        if diff_time >= trigger_time:
            lock_data[stype]['do_time'] = time.time()
        else:
            return False
    else:
        lock_data[stype] = {'do_time': time.time()}
    writeFile(lock_file, json.dumps(lock_data))
    if is_write_log:
        writeLog('通知管理[' + stype + ']', msg)
    data = getNotifyData(True)
    do_notify = False
    if 'tgbot' in data and 'enable' in data['tgbot']:
        if data['tgbot']['enable']:
            t = data['tgbot']['data']
            i = sys.version_info
            if i[0] < 3 or i[1] < 7:
                do_notify = tgbotNotifyHttpPost(t['app_token'], t['chat_id'], msg)
            else:
                do_notify = tgbotNotifyMessage(t['app_token'], t['chat_id'], msg)
    if 'email' in data and 'enable' in data['email']:
        if data['email']['enable']:
            t = data['email']['data']
            t['subject'] = 'MW通知'
            t['content'] = msg
            do_notify = emailNotifyMessage(t)
    return do_notify

def notifyMessage(msg, stype='common', trigger_time=300, is_write_log=True):
    if False:
        return 10
    try:
        return notifyMessageTry(msg, stype, trigger_time, is_write_log)
    except Exception as e:
        writeFileLog(getTracebackInfo())
        return False

def getSshDir():
    if False:
        i = 10
        return i + 15
    if isAppleSystem():
        user = execShell("who | sed -n '2, 1p' |awk '{print $1}'")[0].strip()
        return '/Users/' + user + '/.ssh'
    return '/root/.ssh'

def processExists(pname, exe=None, cmdline=None):
    if False:
        for i in range(10):
            print('nop')
    try:
        import psutil
        pids = psutil.pids()
        for pid in pids:
            try:
                p = psutil.Process(pid)
                if p.name() == pname:
                    if not exe and (not cmdline):
                        return True
                    else:
                        if exe:
                            if p.exe() == exe:
                                return True
                        if cmdline:
                            if cmdline in p.cmdline():
                                return True
            except:
                pass
        return False
    except:
        return True

def createRsa():
    if False:
        for i in range(10):
            print('nop')
    ssh_dir = getSshDir()
    if not os.path.exists(ssh_dir + '/authorized_keys'):
        execShell('touch ' + ssh_dir + '/authorized_keys')
    if not os.path.exists(ssh_dir + '/id_rsa.pub') and os.path.exists(ssh_dir + '/id_rsa'):
        execShell('echo y | ssh-keygen -q -t rsa -P "" -f ' + ssh_dir + '/id_rsa')
    else:
        execShell('ssh-keygen -q -t rsa -P "" -f ' + ssh_dir + '/id_rsa')
    execShell('cat ' + ssh_dir + '/id_rsa.pub >> ' + ssh_dir + '/authorized_keys')
    execShell('chmod 600 ' + ssh_dir + '/authorized_keys')

def createSshInfo():
    if False:
        while True:
            i = 10
    ssh_dir = getSshDir()
    if not os.path.exists(ssh_dir + '/id_rsa') or not os.path.exists(ssh_dir + '/id_rsa.pub'):
        createRsa()
    data = execShell('cat ' + ssh_dir + "/id_rsa.pub | awk '{print $3}'")
    if data[0] != '':
        cmd = 'cat ' + ssh_dir + '/authorized_keys | grep ' + data[0]
        ak_data = execShell(cmd)
        if ak_data[0] == '':
            cmd = 'cat ' + ssh_dir + '/id_rsa.pub >> ' + ssh_dir + '/authorized_keys'
            execShell(cmd)
            execShell('chmod 600 ' + ssh_dir + '/authorized_keys')

def connectSsh():
    if False:
        i = 10
        return i + 15
    import paramiko
    ssh = paramiko.SSHClient()
    createSshInfo()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    port = getSSHPort()
    try:
        ssh.connect('127.0.0.1', port, timeout=5)
    except Exception as e:
        ssh.connect('localhost', port, timeout=5)
    except Exception as e:
        ssh.connect(getHostAddr(), port, timeout=30)
    except Exception as e:
        return False
    shell = ssh.invoke_shell(term='xterm', width=83, height=21)
    shell.setblocking(0)
    return shell

def clearSsh():
    if False:
        while True:
            i = 10
    ip = getHostAddr()
    sh = '\n#!/bin/bash\nPLIST=`who | grep localhost | awk \'{print $2}\'`\nfor i in $PLIST\ndo\n    ps -t /dev/$i |grep -v TTY | awk \'{print $1}\' | xargs kill -9\ndone\n\n# getHostAddr\nPLIST=`who | grep "${ip}" | awk \'{print $2}\'`\nfor i in $PLIST\ndo\n    ps -t /dev/$i |grep -v TTY | awk \'{print $1}\' | xargs kill -9\ndone\n'
    if not isAppleSystem():
        info = execShell(sh)
        print(info[0], info[1])

def echoStart(tag):
    if False:
        i = 10
        return i + 15
    print('=' * 89)
    print('★开始{}[{}]'.format(tag, formatDate()))
    print('=' * 89)

def echoEnd(tag):
    if False:
        return 10
    print('=' * 89)
    print('☆{}完成[{}]'.format(tag, formatDate()))
    print('=' * 89)
    print('\n')

def echoInfo(msg):
    if False:
        print('Hello World!')
    print('|-{}'.format(msg))