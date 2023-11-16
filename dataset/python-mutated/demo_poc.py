from collections import OrderedDict
from pocsuite3.api import Output, POCBase, POC_CATEGORY, register_poc, requests, VUL_TYPE, get_listener_ip, get_listener_port
from pocsuite3.lib.core.interpreter_option import OptString, OptDict, OptIP, OptPort, OptBool, OptInteger, OptFloat, OptItems
from pocsuite3.modules.listener import REVERSE_PAYLOAD

class DemoPOC(POCBase):
    vulID = '1571'
    version = '1'
    author = 'seebug'
    vulDate = '2014-10-16'
    createDate = '2014-10-16'
    updateDate = '2014-10-16'
    references = ['https://xxx.xx.com.cn']
    name = 'XXXX SQL注入漏洞 PoC'
    appPowerLink = 'https://www.drupal.org/'
    appName = 'Drupal'
    appVersion = '7.x'
    vulType = VUL_TYPE.UNAUTHORIZED_ACCESS
    category = POC_CATEGORY.EXPLOITS.WEBAPP
    samples = []
    install_requires = []
    desc = '\n            Drupal 在处理 IN 语句时，展开数组时 key 带入 SQL 语句导致 SQL 注入，\n            可以添加管理员、造成信息泄露。\n        '
    pocDesc = '\n            poc的用法描述\n        '

    def _options(self):
        if False:
            return 10
        opt = OrderedDict()
        opt['string'] = OptString('', description='这个poc需要用户登录，请输入登录账号', require=True)
        opt['integer'] = OptInteger('', description='这个poc需要用户密码，请输出用户密码', require=False)
        return opt

    def _verify(self):
        if False:
            print('Hello World!')
        output = Output(self)
        result = {'Result': {'DBInfo': {'Username': 'xxx', 'Password': 'xxx', 'Salt': 'xxx', 'Uid': 'xxx', 'Groupid': 'xxx'}, 'ShellInfo': {'URL': 'xxx', 'Content': 'xxx'}, 'FileInfo': {'Filename': 'xxx', 'Content': 'xxx'}, 'XSSInfo': {'URL': 'xxx', 'Payload': 'xxx'}, 'AdminInfo': {'Uid': 'xxx', 'Username': 'xxx', 'Password': 'xxx'}, 'Database': {'Hostname': 'xxx', 'Username': 'xxx', 'Password': 'xxx', 'DBname': 'xxx'}, 'VerifyInfo': {'URL': 'xxx', 'Postdata': 'xxx', 'Path': 'xxx'}, 'SiteAttr': {'Process': 'xxx'}, 'Stdout': 'result output string'}}
        if result:
            output.success(result)
        else:
            output.fail('target is not vulnerable')
        return output

    def _attack(self):
        if False:
            return 10
        output = Output(self)
        result = {}
        pass

    def _shell(self):
        if False:
            while True:
                i = 10
        '\n        shell模式下，只能运行单个PoC脚本，控制台会进入shell交互模式执行命令及输出\n        '
        cmd = REVERSE_PAYLOAD.BASH.format(get_listener_ip(), get_listener_port())
        pass

def other_fuc():
    if False:
        for i in range(10):
            print('nop')
    pass

def other_utils_func():
    if False:
        i = 10
        return i + 15
    pass
register_poc(DemoPOC)