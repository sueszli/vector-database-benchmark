import os
import unittest
from pocsuite3.api import init_pocsuite
from pocsuite3.api import load_file_to_module, paths, load_string_to_module

class TestCase(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        pass

    def tearDown(self):
        if False:
            print('Hello World!')
        pass

    def test_get_info(self):
        if False:
            return 10
        init_pocsuite({})
        poc_filename = os.path.join(paths.POCSUITE_POCS_PATH, '20190404_WEB_Confluence_path_traversal.py')
        mod = load_file_to_module(poc_filename)
        print(mod.get_infos())
        self.assertTrue(len(mod.get_infos()) > 0)

    def test_get_info_from_string(self):
        if False:
            i = 10
            return i + 15
        source = '\nfrom collections import OrderedDict\n\nfrom pocsuite3.api import Output, POCBase, POC_CATEGORY, register_poc, requests\nfrom pocsuite3.api import OptString\n\n\nclass DemoPOC(POCBase):\n    vulID = \'00000\'  # ssvid\n    version = \'1.0\'\n    author = [\'chenghs\']\n    vulDate = \'2019-2-26\'\n    createDate = \'2019-2-26\'\n    updateDate = \'2019-2-25\'\n    references = [\'\']\n    name = \'自定义命令参数登录例子\'\n    appPowerLink = \'http://www.knownsec.com/\'\n    appName = \'test\'\n    appVersion = \'test\'\n    vulType = \'demo\'\n    desc = \'\'\'这个例子说明了你可以使用console模式设置一些参数或者使用命令中的\'--\'来设置自定义的参数\'\'\'\n    samples = []\n    category = POC_CATEGORY.EXPLOITS.WEBAPP\n\n    def _options(self):\n        o = OrderedDict()\n        o["username"] = OptString(\'\', description=\'这个poc需要用户登录，请输入登录账号\', require=True)\n        o["password"] = OptString(\'\', description=\'这个poc需要用户密码，请输出用户密码\', require=False)\n        return o\n\n    def _verify(self):\n        result = {}\n        payload = "username={0}&password={1}".format(self.get_option("username"), self.get_option("password"))\n        r = requests.post(self.url, data=payload)\n        print(r.text)\n        if r.status_code == 200:\n            result[\'VerifyInfo\'] = {}\n            result[\'VerifyInfo\'][\'URL\'] = self.url\n            result[\'VerifyInfo\'][\'Postdata\'] = payload\n\n        return self.parse_output(result)\n\n    def _attack(self):\n        return self._verify()\n\n    def parse_output(self, result):\n        output = Output(self)\n        if result:\n            output.success(result)\n        else:\n            output.fail(\'target is not vulnerable\')\n        return output\n\n\nregister_poc(DemoPOC)\n        '.strip()
        init_pocsuite({})
        mod = load_string_to_module(source)
        print(mod.get_infos())
        self.assertTrue(len(mod.get_infos()) > 0)