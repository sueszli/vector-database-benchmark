import json
from projectModel.base import projectBase
import os, re, time
import public
from BTPanel import cache

class main(projectBase):
    __O00O0O0000000O0OO = public.Md5('vulnerability_scanning' + time.strftime('%Y-%m-%d'))
    __OO0O0OOOOOO0O0OO0 = '/www/server/panel/config/vulnerability_scanning.json'
    __OO0O000OOOO0O00OO = public.to_string([27492, 21151, 33021, 20026, 20225, 19994, 29256, 19987, 20139, 21151, 33021, 65292, 35831, 20808, 36141, 20080, 20225, 19994, 29256])

    def __O0000O00000O000O0(OO0OOOO00000O0OO0):
        if False:
            while True:
                i = 10
        from pluginAuth import Plugin
        OO0O0OOO000O00O00 = Plugin(False)
        OO0O00O00O00OOO0O = OO0O0OOO000O00O00.get_plugin_list()
        if int(OO0O00O00O00OOO0O['ltd']) > time.time():
            return True
        return False

    def write_config(OOO0000O00OO00O0O, config=False):
        if False:
            return 10
        ''
        if config:
            public.WriteFile(OOO0000O00OO00O0O.__OO0O0OOOOOO0O0OO0, json.dumps(config))
        else:
            public.WriteFile(OOO0000O00OO00O0O.__OO0O0OOOOOO0O0OO0, json.dumps(OOO0000O00OO00O0O.getDefaultCms()))

    def get_config(O00OOOOO0000OOO00):
        if False:
            print('Hello World!')
        ''
        if not os.path.exists(O00OOOOO0000OOO00.__OO0O0OOOOOO0O0OO0):
            O00OOOOO0000OOO00.write_config()
            return O00OOOOO0000OOO00.getDefaultCms()
        else:
            try:
                O00OO00O0OOOO00O0 = json.loads(public.ReadFile(O00OOOOO0000OOO00.__OO0O0OOOOOO0O0OO0))
            except:
                O00OOOOO0000OOO00.write_config()
                return O00OOOOO0000OOO00.getDefaultCms()
        if not cache.get(O00OOOOO0000OOO00.__O00O0O0000000O0OO):
            try:
                import requests
                O00OO00O0OOOO00O0 = requests.get('https://www.bt.cn/api/bt_waf/scanRules').json()
                cache.set(O00OOOOO0000OOO00.__O00O0O0000000O0OO, '1', 3600)
                O00OOOOO0000OOO00.write_config(O00OO00O0OOOO00O0)
            except:
                return O00OOOOO0000OOO00.getDefaultCms()
            return O00OO00O0OOOO00O0
        else:
            return O00OO00O0OOOO00O0

    def getDefaultCms(OOOOO00OOO0O0O0O0):
        if False:
            i = 10
            return i + 15
        ''
        OO00O0O00O00O00O0 = [{'cms_list': [], 'dangerous': '2', 'cms_name': '迅睿CMS', 'ps': '迅睿CMS 版本过低', 'name': '迅睿CMS 版本过低', 'determine': ['dayrui/My/Config/Version.php'], 'version': {'type': 'file', 'file': 'dayrui/My/Config/Version.php', 'regular': "version.+'(\\d+.\\d+.\\d+)'", 'regular_len': 0, 'vul_version': '3.2.0~4.5.4', 'ver_type': 'range'}, 'repair_file': {'type': 'file', 'file': [{'file': 'dayrui/My/Config/Version.php', 'regular': " if (preg_match('/(php|jsp|asp|exe|sh|cmd|vb|vbs|phtml)/i', $value)) {"}]}, 'repair': '修复建议 : https://www.xunruicms.com/bug/ \n升级到最新版'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'pbootcms', 'ps': 'pbootcms 3.0.0~3.0.4 存在多个高危漏洞CNVD-2020-48981,CNVD-2020-48677,CNVD-2020-48469,CNVD-2020-57593,CNVD-2020-56006,CNVD-2021-00794,CNVD-2021-30081,CNVD-2021-30113,CNVD-2021-32163', 'name': 'pbootcms 2.0.0~2.0.8 存在多个高危漏洞CNVD-2020-48981,CNVD-2020-48677,CNVD-2020-48469,CNVD-2020-57593,CNVD-2020-56006,CNVD-2021-00794,CNVD-2021-30081,CNVD-2021-30113,CNVD-2021-32163', 'determine': ['apps/common/version.php', 'core/basic/Config.php', 'apps/admin/view/default/js/mylayui.js', 'apps/api/controller/ContentController.php'], 'version': {'type': 'file', 'file': 'apps/common/version.php', 'regular': "app_version.+'(\\d+.\\d+.\\d+)'", 'regular_len': 0, 'vul_version': '3.0.0~3.0.4', 'ver_type': 'range'}, 'repair_file': {'type': 'file', 'file': [{'file': 'apps/admin/controller/system/ConfigController.php', 'regular': " if (preg_match('/(php|jsp|asp|exe|sh|cmd|vb|vbs|phtml)/i', $value)) {"}]}, 'repair': '修复建议 : https://www.pbootcms.com/changelog/ \n升级到最新版'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'pbootcms', 'ps': 'pbootcms 2.0.0~2.0.8 存在多个高危漏洞CNVD-2020-04104,CNVD-2020-13536,CNVD-2020-24744,CNVD-2020-32198,CNVD-2020-32180,CNVD-2020-32177,CNVD-2020-31495,CNVD-2019-43060', 'name': 'pbootcms 2.0.0~2.0.8 存在多个高危漏洞CNVD-2020-04104,CNVD-2020-13536,CNVD-2020-24744,CNVD-2020-32198,CNVD-2020-32180,CNVD-2020-32177,CNVD-2020-31495,CNVD-2019-43060', 'determine': ['apps/common/version.php', 'core/basic/Config.php', 'apps/admin/view/default/js/mylayui.js', 'apps/api/controller/ContentController.php'], 'version': {'type': 'file', 'file': 'apps/common/version.php', 'regular': "app_version.+'(\\d+.\\d+.\\d+)'", 'regular_len': 0, 'vul_version': '2.0.0~2.0.8', 'ver_type': 'range'}, 'repair_file': {'type': 'file', 'file': [{'file': 'apps/home/controller/ParserController.php', 'regular': " if (preg_match('/(\\$_GET\\[)|(\\$_POST\\[)|(\\$_REQUEST\\[)|(\\$_COOKIE\\[)|(\\$_SESSION\\[)|(file_put_contents)|(file_get_contents)|(fwrite)|(phpinfo)|(base64)|(`)|(shell_exec)|(eval)|(assert)|(system)|(exec)|(passthru)|(print_r)|(urldecode)|(chr)|(include)|(request)|(__FILE__)|(__DIR__)|(copy)/i', $matches[1][$i]))"}]}, 'repair': '修复建议 : https://www.pbootcms.com/changelog/ \n升级到最新版'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'pbootcms', 'ps': 'pbootcms 1.3.0~1.3.8 存在多个高危漏洞CNVD-2018-26355,CNVD-2018-24253,CNVD-2018-26938,CNVD-2019-14855,CNVD-2019-27743,CNVD-2020-23841', 'name': 'pbootcms 1.3.0~1.3.8 存在多个高危漏洞CNVD-2018-26355,CNVD-2018-24253,CNVD-2018-26938,CNVD-2019-14855,CNVD-2019-27743,CNVD-2020-23841', 'determine': ['apps/common/version.php', 'core/basic/Config.php', 'apps/admin/view/default/js/mylayui.js', 'apps/api/controller/ContentController.php'], 'version': {'type': 'file', 'file': 'apps/common/version.php', 'regular': "app_version.+'(\\d+.\\d+.\\d+)'", 'regular_len': 0, 'vul_version': '1.3.0~1.3.8', 'ver_type': 'range'}, 'repair_file': {'type': 'file', 'file': [{'file': 'apps/admin/controller/system/ConfigController.php', 'regular': '$config = preg_replace(\'/(\'\' . $key . \'\'([\\s]+)?=>([\\s]+)?)[\\w\'"\\s,]+,/\', \'${1}\'\' . $value . \'\',\', $config);'}]}, 'repair': '修复建议 : https://www.pbootcms.com/changelog/ \n升级到最新版'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'pbootcms', 'ps': 'pbootcms 1.2.0~1.2.2 存在多个高危漏洞CNVD-2018-21503,CNVD-2018-19945,CNVD-2018-22854,CNVD-2018-22142,CNVD-2018-26780,CNVD-2018-24845', 'name': 'pbootcms 1.0.1~1.2.2 存在多个高危漏洞CNVD-2018-21503,CNVD-2018-19945,CNVD-2018-22854,CNVD-2018-22142,CNVD-2018-26780,CNVD-2018-24845', 'determine': ['apps/common/version.php', 'core/basic/Config.php', 'apps/admin/view/default/js/mylayui.js', 'apps/api/controller/ContentController.php'], 'version': {'type': 'file', 'file': 'apps/common/version.php', 'regular': "app_version.+'(\\d+.\\d+.\\d+)'", 'regular_len': 0, 'vul_version': ['1.2.0', '1.2.1', '1.2.2'], 'ver_type': 'list'}, 'repair_file': {'type': 'file', 'file': [{'file': 'apps/admin/controller/system/DatabaseController.php', 'regular': "if ($value && ! preg_match('/(^|[\\s]+)(drop|truncate|set)[\\s]+/i', $value)) {"}]}, 'repair': '修复建议 : https://www.pbootcms.com/changelog/ \n升级到最新版'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'pbootcms', 'ps': 'pbootcms 1.1.9 存在SQL注入漏洞CNVD-2018-18069', 'name': 'pbootcms 1.1.9 存在SQL注入漏洞CNVD-2018-18069', 'determine': ['apps/common/version.php', 'core/basic/Config.php', 'apps/admin/view/default/js/mylayui.js', 'apps/api/controller/ContentController.php'], 'version': {'type': 'file', 'file': 'apps/common/version.php', 'regular': "app_version.+'(\\d+.\\d+.\\d+)'", 'regular_len': 0, 'vul_version': ['1.1.9'], 'ver_type': 'list'}, 'repair_file': {'type': 'file', 'file': [{'file': 'core/function/handle.php', 'regular': "if (Config::get('url_type') == 2 && strrpos($indexfile, 'index.php') !== false)"}]}, 'repair': '修复建议 : https://www.pbootcms.com/changelog/ \n升级到最新版'}, {'cms_list': [], 'dangerous': '4', 'cms_name': 'pbootcms', 'ps': 'pbootcms 1.1.6~1.1.8 存在前台代码执行漏洞、存在多个SQL注入漏洞 CNVD-2018-17412,CNVD-2018-17741,CNVD-2018-17747,CNVD-2018-17750,CNVD-2018-17751,CNVD-2018-17752,CNVD-2018-17753,CNVD-2018-17754', 'name': 'pbootcms 1.1.6~1.1.8  存在前台代码执行漏洞、存在多个SQL注入漏洞 CNVD-2018-17412,CNVD-2018-17741,CNVD-2018-17747,CNVD-2018-17750,CNVD-2018-17751,CNVD-2018-17752,CNVD-2018-17753,CNVD-2018-17754', 'determine': ['apps/common/version.php', 'core/basic/Config.php', 'apps/admin/view/default/js/mylayui.js', 'apps/api/controller/ContentController.php'], 'version': {'type': 'file', 'file': 'apps/common/version.php', 'regular': "app_version.+'(\\d+.\\d+.\\d+)'", 'regular_len': 0, 'vul_version': ['1.1.6', '1.1.7', '1.1.8'], 'ver_type': 'list'}, 'repair_file': {'type': 'file', 'file': [{'file': 'core/function/handle.php', 'regular': 'if (is_array($string)) { // 数组处理\n        foreach ($string as $key => $value) {\n            $string[$key] = decode_slashes($value);\n        }'}]}, 'repair': '修复建议 : https://www.pbootcms.com/changelog/ \n升级到最新版'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'pbootcms', 'ps': 'pbootcms 1.1.4 存在SQL注入漏洞CNVD-2018-13335,CNVD-2018-13336', 'name': 'pbootcms 1.1.4 存在SQL注入漏洞CNVD-2018-13335,CNVD-2018-13336', 'determine': ['apps/common/version.php', 'core/basic/Config.php', 'apps/admin/view/default/js/mylayui.js', 'apps/api/controller/ContentController.php'], 'version': {'type': 'file', 'file': 'apps/common/version.php', 'regular': "app_version.+'(\\d+.\\d+.\\d+)'", 'regular_len': 0, 'vul_version': ['1.1.4'], 'ver_type': 'list'}, 'repair_file': {'type': 'file', 'file': [{'file': 'core/extend/ueditor/php/controller.php', 'regular': "if (! ini_get('session.auto_start') && ! isset($_SESSION)"}]}, 'repair': '修复建议 : https://www.pbootcms.com/changelog/ \n升级到最新版'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'maccms10', 'ps': 'maccms10 <=2022.1000.3025 存在ssrf漏洞、存在XSS漏洞', 'name': 'maccms10 <=2022.1000.3025 存在ssrf漏洞、存在XSS漏洞', 'determine': ['application/extra/version.php', 'application/api/controller/Wechat.php', 'thinkphp/library/think/Route.php', 'application/admin/controller/Upload.php'], 'version': {'type': 'file', 'file': 'application/extra/version.php', 'regular': "code.+'(\\d+.\\d+.\\d+)'", 'regular_len': 0, 'vul_version': ['2022.1000.3025', '2022.1000.3005', '2022.1000.3024', '2022.1000.3020', '2022.1000.3023', '2022.1000.3002', '2022.1000.1099', '2021.1000.1081'], 'ver_type': 'list'}, 'repair_file': {'type': 'file', 'file': [{'file': 'application/common/model/Actor.php', 'regular': '$data[$filter_field] = mac_filter_xss($data[$filter_field]);'}]}, 'repair': '修复建议 : https://github.com/magicblack/maccms10/releases \n升级到最新版'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'maccms10', 'ps': 'maccms10 <=2022.1000.3024 存在前台任意用户登陆、后台会话验证绕过、后台任意文件写入、任意文件删除漏洞', 'name': 'maccms10 <=2022.1000.3024 存在前台任意用户登陆、后台会话验证绕过、后台任意文件写入、任意文件删除漏洞', 'determine': ['application/extra/version.php', 'application/api/controller/Wechat.php', 'thinkphp/library/think/Route.php', 'application/admin/controller/Upload.php'], 'version': {'type': 'file', 'file': 'application/extra/version.php', 'regular': "code.+'(\\d+.\\d+.\\d+)'", 'regular_len': 0, 'vul_version': ['2022.1000.3005', '2022.1000.3024', '2022.1000.3020', '2022.1000.3023', '2022.1000.3002', '2022.1000.1099', '2021.1000.1081'], 'ver_type': 'list'}, 'repair_file': {'type': 'file', 'file': [{'file': 'application/common/model/Annex.php', 'regular': "if (stripos($v['annex_file'], '../') !== false)"}]}, 'repair': '修复建议 : https://github.com/magicblack/maccms10/releases \n升级到最新版'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'eyoucms', 'ps': 'eyoucms 1.5.5~1.5.7 存在多个安全漏洞', 'name': 'eyoucms 1.5.1~1.5.4 存在多个安全漏洞', 'determine': ['data/conf/version.txt', 'application/api/controller/Uploadify.php', 'application/extra/extra_cache_key.php', 'application/admin/controller/Uploadify.php'], 'version': {'type': 'file', 'file': 'data/conf/version.txt', 'regular': '(\\d+.\\d+.\\d+)', 'regular_len': 0, 'vul_version': '1.5.5~1.5.7', 'ver_type': 'range'}, 'repair_file': {'type': 'file', 'file': [{'file': 'application/common.php', 'regular': "$login_errnum_key = 'adminlogin_'.md5('login_errnum_'.$admin_info['user_name']);"}]}, 'repair': '修复建议 : https://www.eyoucms.com/rizhi/ \n升级eyoucms到最新版'}, {'cms_list': [], 'dangerous': '4', 'cms_name': 'eyoucms', 'ps': 'eyoucms 1.5.1~1.5.4 存在多个高危安全漏洞,CNVD-2021-82431,CNVD-2021-82429,CNVD-2021-72772,CNVD-2021-51838,CNVD-2021-51836,CNVD-2021-41520,CNVD-2021-24745,,CNVD-2021-26007,CNVD-2021-26099,CNVD-2021-41520', 'name': 'eyoucms 1.5.1~1.5.4 存在多个高危安全漏洞,CNVD-2021-82431,CNVD-2021-82429,CNVD-2021-72772,CNVD-2021-51838,CNVD-2021-51836,CNVD-2021-41520,CNVD-2021-24745,,CNVD-2021-26007,CNVD-2021-26099,CNVD-2021-41520', 'determine': ['data/conf/version.txt', 'application/api/controller/Uploadify.php', 'application/extra/extra_cache_key.php', 'application/admin/controller/Uploadify.php'], 'version': {'type': 'file', 'file': 'data/conf/version.txt', 'regular': '(\\d+.\\d+.\\d+)', 'regular_len': 0, 'vul_version': '1.5.1~1.5.4', 'ver_type': 'range'}, 'repair_file': {'type': 'file', 'file': [{'file': 'application/common.php', 'regular': "$citysite_db->where(['domain'=>$s_arr[0]])->cache(true, EYOUCMS_CACHE_TIME, 'citysite')->count()"}]}, 'repair': '修复建议 : https://www.eyoucms.com/rizhi/ \n升级eyoucms到最新版'}, {'cms_list': [], 'dangerous': '4', 'cms_name': 'eyoucms', 'ps': 'eyoucms 1.4.7 存在多个高危安全漏洞,CNVD-2020-46317,CNVD-2020-49065,CNVD-2020-44394,CNVD-2020-44392,CNVD-2020-44391,CNVD-2020-47671,CNVD-2020-50721', 'name': 'eyoucms 1.4.7 存在多个高危安全漏洞,CNVD-2020-46317,CNVD-2020-49065,CNVD-2020-44394,CNVD-2020-44392,CNVD-2020-44391,CNVD-2020-47671,CNVD-2020-50721', 'determine': ['data/conf/version.txt', 'application/api/controller/Uploadify.php', 'application/extra/extra_cache_key.php', 'application/admin/controller/Uploadify.php'], 'version': {'type': 'file', 'file': 'data/conf/version.txt', 'regular': '(\\d+.\\d+.\\d+)', 'regular_len': 0, 'vul_version': '1.4.7~1.4.7', 'ver_type': 'range'}, 'repair_file': {'type': 'file', 'file': [{'file': 'application/common.php', 'regular': "function GetTagIndexRanking($limit = 5, $field = 'id, tag')"}]}, 'repair': '修复建议 : https://www.eyoucms.com/rizhi/ \n升级eyoucms到最新版'}, {'cms_list': [], 'dangerous': '4', 'cms_name': 'eyoucms', 'ps': 'eyoucms 1.4.6 存在多个高危安全漏洞,CNVD-2020-44116,CNVD-2020-32622,CNVD-2020-28132,CNVD-2020-28083,CNVD-2020-28064,CNVD-2020-33104', 'name': 'eyoucms 1.4.6 存在多个高危安全漏洞,CNVD-2020-44116,CNVD-2020-32622,CNVD-2020-28132,CNVD-2020-28083,CNVD-2020-28064,CNVD-2020-33104', 'determine': ['data/conf/version.txt', 'application/api/controller/Uploadify.php', 'application/extra/extra_cache_key.php', 'application/admin/controller/Uploadify.php'], 'version': {'type': 'file', 'file': 'data/conf/version.txt', 'regular': '(\\d+.\\d+.\\d+)', 'regular_len': 0, 'vul_version': '1.4.6~1.4.6', 'ver_type': 'range'}, 'repair_file': {'type': 'file', 'file': [{'file': 'application/common.php', 'regular': "preg_replace('#^(/[/\\w]+)?(/uploads/|/public/static/)#i"}]}, 'repair': '修复建议 : https://www.eyoucms.com/rizhi/ \n升级eyoucms到最新版'}, {'cms_list': [], 'dangerous': '4', 'cms_name': 'eyoucms', 'ps': 'eyoucms 1.3.9~1.4.4 存在多个安全漏洞CNVD-2020-02271,CNVD-2020-02824,CNVD-2020-18735,CNVD-2020-18677,CNVD-2020-23229,CNVD-2020-23805,CNVD-2020-23820', 'name': 'eyoucms 1.3.9~1.4.4 存在多个安全漏洞CNVD-2020-02271,CNVD-2020-02824,CNVD-2020-18735,CNVD-2020-18677,CNVD-2020-23229,CNVD-2020-23805,CNVD-2020-23820', 'determine': ['data/conf/version.txt', 'application/api/controller/Uploadify.php', 'application/extra/extra_cache_key.php', 'application/admin/controller/Uploadify.php'], 'version': {'type': 'file', 'file': 'data/conf/version.txt', 'regular': '(\\d+.\\d+.\\d+)', 'regular_len': 0, 'vul_version': '1.3.9~1.4.4', 'ver_type': 'range'}, 'repair_file': {'type': 'file', 'file': [{'file': 'application/common.php', 'regular': "$TimingTaskRow = model('Weapp')->getWeappList('TimingTask');"}]}, 'repair': '修复建议 : https://www.eyoucms.com/rizhi/ \n升级eyoucms到最新版'}, {'cms_list': [], 'dangerous': '4', 'cms_name': 'eyoucms', 'ps': 'eyoucms 1.4.1 存在命令执行漏洞', 'name': 'eyoucms 1.4.1 存在命令执行漏洞', 'determine': ['data/conf/version.txt', 'application/api/controller/Uploadify.php', 'application/extra/extra_cache_key.php', 'application/admin/controller/Uploadify.php'], 'version': {'type': 'file', 'file': 'data/conf/version.txt', 'regular': '(\\d+.\\d+.\\d+)', 'regular_len': 0, 'vul_version': '1.4.1~1.4.1', 'ver_type': 'range'}, 'repair_file': {'type': 'file', 'file': [{'file': 'application/route.php', 'regular': "$weapp_route_file = 'plugins/route.php';"}]}, 'repair': '修复建议 : https://www.eyoucms.com/rizhi/ \n升级eyoucms到最新版'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'eyoucms', 'ps': 'eyoucms<=1.3.8 存在SQL注入、存在插件上传漏洞', 'name': 'eyoucms<=1.3.8 存在SQL注入、存在插件上传漏洞', 'determine': ['data/conf/version.txt', 'application/api/controller/Uploadify.php', 'application/extra/extra_cache_key.php', 'application/admin/controller/Uploadify.php'], 'version': {'type': 'file', 'file': 'data/conf/version.txt', 'regular': '(\\d+.\\d+.\\d+)', 'regular_len': 0, 'vul_version': '1.0.0~1.3.8', 'ver_type': 'range'}, 'repair_file': {'type': 'file', 'file': [{'file': 'core/library/think/template/taglib/Eyou.php', 'regular': "$notypeid  = !empty($tag['notypeid']) ? $tag['notypeid'] : '';"}]}, 'repair': '修复建议 : https://www.eyoucms.com/rizhi/ \n升级eyoucms到最新版'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'eyoucms', 'ps': 'eyoucms<=1.3.4 存在后台文件上传漏洞', 'name': 'eyoucms<=1.3.4 存在后台文件上传漏洞', 'determine': ['data/conf/version.txt', 'application/api/controller/Uploadify.php', 'application/extra/extra_cache_key.php', 'application/admin/controller/Uploadify.php'], 'version': {'type': 'file', 'file': 'data/conf/version.txt', 'regular': '(\\d+.\\d+.\\d+)', 'regular_len': 0, 'vul_version': '1.0.0~1.3.4', 'ver_type': 'range'}, 'repair_file': {'type': 'file', 'file': [{'file': 'application/common.php', 'regular': 'include_once EXTEND_PATH."function.php";'}]}, 'repair': '修复建议 : https://www.eyoucms.com/rizhi/ \n升级eyoucms到最新版'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'eyoucms', 'ps': 'eyoucms 1.0 存在任意文件上传漏洞', 'name': 'eyoucms 1.0 存在任意文件上传漏洞', 'determine': ['data/conf/version.txt', 'application/api/controller/Uploadify.php', 'application/extra/extra_cache_key.php', 'application/admin/controller/Uploadify.php'], 'version': {'type': 'file', 'file': 'data/conf/version.txt', 'regular': '(\\d+.\\d+.\\d+)', 'regular_len': 0, 'vul_version': '1.0.0~1.1.0', 'ver_type': 'range'}, 'repair_file': {'type': 'file', 'file': [{'file': 'application/api/controller/Uploadify.php', 'regular': '目前没用到这个api接口'}]}, 'repair': '修复建议 : https://www.eyoucms.com/rizhi/ \n升级eyoucms到最新版'}, {'cms_list': [], 'dangerous': '2', 'cms_name': '海洋CMS', 'ps': '海洋CMS 版本过低', 'name': '海洋CMS 版本过低', 'determine': ['data/admin/ver.txt', 'include/common.php', 'include/main.class.php', 'detail/index.php'], 'version': {'type': 'file', 'file': 'data/admin/ver.txt', 'regular': '(\\d+.\\d+?|\\d+)', 'regular_len': 0, 'vul_version': ['6.28', '6.54', '7.2', '8.4', '8.5', '8.6', '8.7', '8.8', '8.9', '9', '9.1', '9.2', '9.3', '9.4', '9.5', '9.6', '9.7', '9.8', '9.9', '9.91', '9.92', '9.93', '9.94', '9.96', '9.97', '9.98', '9.99', '10', '10.1', '10.2', '10.3', '10.4', '10.5', '10.6', '10.7', '10.8', '10.9', '11', '11.1', '11.2', '11.3', '11.4', '11.5'], 'ver_type': 'list'}, 'repair_file': {'type': 'version', 'file': []}, 'repair': '修复建议:升级到海洋CMS 最新版 https://www.seacms.net/p-549'}, {'cms_list': [], 'dangerous': '3', 'cms_name': '海洋CMS', 'ps': '海洋CMS <=9.95存在前台RCE', 'name': '海洋CMS <=9.95存在前台RCE', 'determine': ['data/admin/ver.txt', 'include/common.php', 'include/main.class.php', 'detail/index.php'], 'version': {'type': 'file', 'file': 'data/admin/ver.txt', 'regular': '(\\d+.\\d+?|\\d+)', 'regular_len': 0, 'vul_version': ['6.28', '6.54', '7.2', '8.4', '8.5', '8.6', '8.7', '8.8', '8.9', '9', '9.1', '9.2', '9.3', '9.4', '9.5', '9.6', '9.7', '9.8', '9.9', '9.91', '9.92', '9.93', '9.94'], 'ver_type': 'list'}, 'repair_file': {'type': 'file', 'file': [{'file': 'include/common.php', 'regular': "'$jpurl='//'.$_SERVER['SERVER_NAME']"}]}, 'repair': '修复建议:升级到海洋CMS 最新版 https://www.seacms.net/p-549'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'ThinkCMF', 'ps': 'ThinkCMF CVE-2019-6713漏洞', 'name': 'ThinkCMF CVE-2019-6713', 'determine': ['public/index.php', 'app/admin/hooks.php', 'app/admin/controller/NavMenuController.php', 'simplewind/cmf/hooks.php'], 'version': {'type': 'file', 'file': 'public/index.php', 'regular': "THINKCMF_VERSION.+'(\\d+.\\d+.\\d+)'", 'regular_len': 0, 'vul_version': ['5.0.190111', '5.0.181231', '5.0.181212', '5.0.180901', '5.0.180626', '5.0.180525', '5.0.180508'], 'ver_type': 'list'}, 'repair_file': {'type': 'file', 'file': [{'file': 'app/admin/validate/RouteValidate.php', 'regular': 'protected function checkUrl($value, $rule, $data)'}]}, 'repair': '修复建议:1.修改代码 https://github.com/thinkcmf/thinkcmf/commit/217b6f8ad77a2917634bb9dd9c1f4ccf2c4c2930\n2.升级到最新版 https://github.com/thinkcmf/thinkcmf/releases/tag/5.0.1904193.升级到6.0 https://github.com/thinkcmf/thinkcmf/releases'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'ThinkCMF', 'ps': 'ThinkCMF templateFile远程代码执行漏洞', 'name': 'ThinkCMF templateFile远程代码执行漏洞', 'determine': ['simplewind/Core/ThinkPHP.php', 'index.php', 'data/conf/db.php', 'application/Admin/Controller/NavcatController.class.php', 'application/Comment/Controller/WidgetController.class.php'], 'version': {'type': 'file', 'file': 'index.php', 'regular': "THINKCMF_VERSION.+(\\d+.\\d+.\\d+)'", 'regular_len': 0, 'vul_version': '1.6.0~2.2.2', 'ver_type': 'range'}, 'repair_file': {'type': 'file', 'file': [{'file': 'application/Comment/Controller/WidgetController.class.php', 'regular': 'protected function display('}]}, 'repair': '修复建议:1.修改代码 https://gitee.com/thinkcmf/ThinkCMFX/commit/559b868283bc491cf858d2f85bcd5b6cfa425d63\n2.升级到最新版 https://gitee.com/thinkcmf/ThinkCMFX/releases/X2.2.43.升级到ThinkCMF5.X 或者ThinkCMF6.X'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'zfaka', 'ps': 'zfaka存在SQL注入漏洞', 'name': 'zfaka存在SQL注入漏洞', 'determine': ['application/init.php', 'application/function/F_Network.php', 'application/controllers/Error.php', 'application/modules/Admin/controllers/Profiles.php'], 'version': {'type': 'file', 'file': 'application/init.php', 'regular': "VERSION.+'(\\d+.\\d+.\\d+)'", 'regular_len': 0, 'vul_version': '1.0.0~1.4.4', 'ver_type': 'range'}, 'repair_file': {'type': 'file', 'file': [{'file': 'application/function/F_Network.php', 'regular': 'if(filter_var($ip, FILTER_VALIDATE_IP, FILTER_FLAG_IPV4'}]}, 'repair': '修复建议:\nhttps://github.com/zlkbdotnet/zfaka/commit/f0f504528347a758fc34fb4b8dbc69377b099b8e?branch=f0f504528347a758fc34fb4b8dbc69377b099b8e&diff=split\n 或者升级到1.4.5'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'dedecms', 'ps': 'dedecms 20210719安全更新', 'name': 'dedecms 20210719安全更新', 'determine': ['data/admin/ver.txt', 'data/common.inc.php', 'dede/shops_operations_userinfo.php', 'member/edit_space_info.php'], 'version': {'type': 'file', 'file': 'data/admin/ver.txt', 'regular': '(\\d+)', 'regular_len': 0, 'vul_version': ['20180109'], 'ver_type': 'list'}, 'repair_file': {'type': 'file', 'file': [{'file': 'include/dedemodule.class.php', 'regular': 'if(preg_match("#[^a-z]+(eval|assert)[\\s]*[(]#i"'}]}, 'repair': '修复建议:https://www.dedecms.com/package.html?t=1626652800\n 也可以在后台进行更新'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'dedecms', 'ps': 'dedecms 20220125安全更新', 'name': 'dedecms 20220125安全更新', 'determine': ['data/admin/ver.txt', 'data/common.inc.php', 'dede/shops_operations_userinfo.php', 'member/edit_space_info.php'], 'version': {'type': 'file', 'file': 'data/admin/ver.txt', 'regular': '(\\d+)', 'regular_len': 0, 'vul_version': ['20180109', '20220325', '20210201', '20210806'], 'ver_type': 'list'}, 'repair_file': {'type': 'file', 'file': [{'file': 'include/downmix.inc.php', 'regular': '上海卓卓网络科技有限公司'}]}, 'repair': '修复建议:https://www.dedecms.com/package.html?t=1643068800\n 也可以在后台进行更新'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'dedecms', 'ps': 'dedecms 20220218安全更新', 'name': 'dedecms 20220218安全更新', 'determine': ['data/admin/ver.txt', 'data/common.inc.php', 'dede/shops_operations_userinfo.php', 'member/edit_space_info.php'], 'version': {'type': 'file', 'file': 'data/admin/ver.txt', 'regular': '(\\d+)', 'regular_len': 0, 'vul_version': ['20180109', '20220325', '20210201', '20210806'], 'ver_type': 'list'}, 'repair_file': {'type': 'file', 'file': [{'file': 'dede/file_manage_control.php', 'regular': 'phpinfo,eval,assert,exec,passthru,shell_exec,system,proc_open,popen'}]}, 'repair': '修复建议:https://www.dedecms.com/package.html?t=1645142400\n 也可以在后台进行更新'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'dedecms', 'ps': 'dedecms 20220310安全更新', 'name': 'dedecms 20220310安全更新', 'determine': ['data/admin/ver.txt', 'data/common.inc.php', 'dede/shops_operations_userinfo.php', 'member/edit_space_info.php'], 'version': {'type': 'file', 'file': 'data/admin/ver.txt', 'regular': '(\\d+)', 'regular_len': 0, 'vul_version': ['20180109', '20220325', '20210201', '20210806'], 'ver_type': 'list'}, 'repair_file': {'type': 'file', 'file': [{'file': 'dede/file_manage_control.php', 'regular': 'phpinfo,eval,assert,exec,passthru,shell_exec,system,proc_open,popen'}]}, 'repair': '修复建议:https://www.dedecms.com/package.html?t=1646870400\n 也可以在后台进行更新'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'dedecms', 'ps': 'dedecms 20220325安全更新', 'name': 'dedecms 20220325安全更新', 'determine': ['data/admin/ver.txt', 'data/common.inc.php', 'dede/shops_operations_userinfo.php', 'member/edit_space_info.php'], 'version': {'type': 'file', 'file': 'data/admin/ver.txt', 'regular': '(\\d+)', 'regular_len': 0, 'vul_version': ['20180109', '20220325', '20210201', '20210806'], 'ver_type': 'list'}, 'repair_file': {'type': 'file', 'file': [{'file': 'plus/mytag_js.php', 'regular': 'phpinfo,eval,assert,exec,passthru,shell_exec,system,proc_open,popen'}]}, 'repair': '修复建议:https://www.dedecms.com/package.html?t=1648166400\n 也可以在后台进行更新'}, {'cms_list': [], 'dangerous': '2', 'cms_name': 'dedecms', 'ps': 'dedecms 已开启会员注册功能', 'name': 'dedecms 已开启会员注册功能', 'determine': ['data/admin/ver.txt', 'data/common.inc.php', 'dede/shops_operations_userinfo.php', 'member/edit_space_info.php'], 'version': {'type': 'file', 'file': 'data/admin/ver.txt', 'regular': '(\\d+)', 'regular_len': 0, 'vul_version': ['20180109', '20220325', '20210201', '20210806'], 'ver_type': 'list'}, 'repair_file': {'type': 'phpshell', 'file': [{'file': 'member/get_user_cfg_mb_open.php', 'phptext': "<?php require_once(dirname(__FILE__).'/../include/common.inc.php');echo 'start'.$cfg_mb_open.'end';?>", 'regular': 'start(\\w)end', 'reulst_type': 'str', 'result': 'startYend'}]}, 'repair': '修复建议:建议在后台关闭会员注册功能'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'MetInfo', 'ps': 'MetInfo 7.5.0存在SQL注入漏洞', 'name': 'MetInfo7.5.0存在SQL注入漏洞', 'determine': ['cache/config/config_metinfo.php', 'app/system/entrance.php', 'app/system/databack/admin/index.class.php', 'cache/config/app_config_metinfo.php'], 'version': {'type': 'file', 'file': 'cache/config/config_metinfo.php', 'regular': "value.+'(\\d+.\\d+.\\d+)'", 'vul_version': '7.5.0~7.5.0', 'ver_type': 'range'}, 'repair_file': {'type': 'version', 'file': []}, 'repair': '漏洞参考:https://www.metinfo.cn/log/ 建议升级到最新版'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'MetInfo', 'ps': 'MetInfo 7.3.0存在SQL注入漏洞、XSS漏洞', 'name': 'MetInfo 7.3.0存在SQL注入漏洞、XSS漏洞', 'determine': ['app/system/entrance.php', 'app/system/admin/admin/index.class.php', 'app/system/admin/admin/templates/admin_add.php'], 'version': {'type': 'file', 'file': 'app/system/entrance.php', 'regular': "SYS_VER.+'(\\d+.\\d+.\\d+)'", 'vul_version': '7.3.0~7.3.0', 'ver_type': 'range'}, 'repair_file': {'type': 'version', 'file': []}, 'repair': '漏洞参考:https://www.metinfo.cn/log/ 建议升级到最新版'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'MetInfo', 'ps': 'MetInfo 7.2.0存在SQL注入漏洞、XSS漏洞', 'name': 'MetInfo 7.2.0存在SQL注入漏洞、XSS漏洞', 'determine': ['app/system/entrance.php', 'app/system/admin/admin/index.class.php', 'app/system/admin/admin/templates/admin_add.php'], 'version': {'type': 'file', 'file': 'app/system/entrance.php', 'regular': "SYS_VER.+'(\\d+.\\d+.\\d+)'", 'vul_version': '7.2.0~7.2.0', 'ver_type': 'range'}, 'repair_file': {'type': 'version', 'file': []}, 'repair': '漏洞参考:https://www.metinfo.cn/log/ 建议升级到最新版'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'MetInfo', 'ps': 'MetInfo 7.1.0存在文件上传漏洞、SQL注入漏洞、XSS漏洞', 'name': 'MetInfo 7.1.0存在文件上传漏洞、SQL注入漏洞、XSS漏洞', 'determine': ['app/system/entrance.php', 'app/system/admin/admin/index.class.php', 'app/system/admin/admin/templates/admin_add.php'], 'version': {'type': 'file', 'file': 'app/system/entrance.php', 'regular': "SYS_VER.+'(\\d+.\\d+.\\d+)'", 'vul_version': '7.1.0~7.1.0', 'ver_type': 'range'}, 'repair_file': {'type': 'version', 'file': []}, 'repair': '漏洞参考:https://www.metinfo.cn/log/ 建议升级到最新版'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'MetInfo', 'ps': 'MetInfo 7.0.0 存在SQL注入漏洞', 'name': 'MetInfo7.0.0存在SQL注入漏洞', 'determine': ['app/system/entrance.php', 'app/system/admin/admin/index.class.php', 'app/system/admin/admin/templates/admin_add.php'], 'version': {'type': 'file', 'file': 'app/system/entrance.php', 'regular': "SYS_VER.+'(\\d+.\\d+.\\d+)'", 'vul_version': '7.0.0~7.0.0', 'ver_type': 'range'}, 'repair_file': {'type': 'version', 'file': []}, 'repair': '漏洞参考:https://www.metinfo.cn/log/ 建议升级到最新版'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'MetInfo', 'ps': 'MetInfo 6.1.2存在SQL注入漏洞', 'name': 'MetInfo 6.1.2存在SQL注入漏洞', 'determine': ['app/system/entrance.php', 'app/system/admin/admin/templates/admin_add.php'], 'version': {'type': 'file', 'file': 'app/system/entrance.php', 'regular': "SYS_VER.+'(\\d+.\\d+.\\d+)'", 'vul_version': '6.1.2~6.1.2', 'ver_type': 'range'}, 'repair_file': {'type': 'version', 'file': []}, 'repair': '漏洞参考:https://www.metinfo.cn/log/ 建议升级到最新版'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'MetInfo', 'ps': 'MetInfo 6.1.1 存在已知后台权限可以获取webshell漏洞', 'name': 'MetInfo 6.1.1 存在已知后台权限可以获取webshell漏洞', 'determine': ['app/system/entrance.php', 'app/system/admin/admin/templates/admin_add.php'], 'version': {'type': 'file', 'file': 'app/system/entrance.php', 'regular': "SYS_VER.+'(\\d+.\\d+.\\d+)'", 'vul_version': '6.1.1~6.1.1', 'ver_type': 'range'}, 'repair_file': {'type': 'version', 'file': []}, 'repair': '漏洞参考:https://www.metinfo.cn/log/ 建议升级到最新版'}, {'cms_list': [], 'dangerous': '2', 'cms_name': 'emlog', 'ps': 'emlog版本太低建议升级到Pro版本', 'name': 'emlog版本太低建议升级到Pro版本', 'determine': ['include/lib/option.php', 'admin/views/template_install.php', 'include/lib/checkcode.php', 'include/controller/author_controller.php'], 'version': {'type': 'file', 'file': 'include/lib/option.php', 'regular': "EMLOG_VERSION.+'(\\d+.\\d+.\\d+)'", 'vul_version': '5.3.1~6.0.0', 'ver_type': 'range'}, 'repair_file': {'type': 'version', 'file': []}, 'repair': '漏洞参考:https://www.emlog.net/docs/#/531toPro'}, {'cms_list': [], 'dangerous': '1', 'cms_name': '帝国CMS', 'ps': 'EmpireCMs7.0后台XSS漏洞', 'name': 'EmpireCMs7.0后台XSS漏洞', 'determine': ['e/class/EmpireCMS_version.php', 'e/search/index.php', 'e/member/EditInfo/index.php', 'e/ViewImg/index.html'], 'version': {'type': 'file', 'file': 'e/class/EmpireCMS_version.php', 'regular': "EmpireCMS_VERSION.+'(\\d+.\\d+)'", 'vul_version': '7.0~7.0', 'ver_type': 'range'}, 'repair_file': {'type': 'version', 'file': []}, 'repair': '漏洞参考:/e/admin/openpage/AdminPage.php?ehash_f9Tj7=ZMhwowHjtSwqyRuiOylK&mainfile=javascript:alert(1)'}, {'cms_list': [], 'dangerous': '2', 'cms_name': '帝国CMS', 'ps': 'EmpireCMs6.0~7.5 后台代码执行', 'name': 'EmpireCMs6.0~7.5 后台代码执行', 'determine': ['e/class/EmpireCMS_version.php', 'e/search/index.php', 'e/member/EditInfo/index.php', 'e/ViewImg/index.html'], 'version': {'type': 'file', 'file': 'e/class/EmpireCMS_version.php', 'regular': "EmpireCMS_VERSION.+'(\\d+.\\d+)'", 'vul_version': '6.0~7.5', 'ver_type': 'range'}, 'repair_file': {'type': 'version', 'file': []}, 'repair': '漏洞参考:https://blog.csdn.net/ws13129/article/details/90071260 官方暂未有修复方案'}, {'cms_list': [], 'dangerous': '2', 'cms_name': '帝国CMS', 'ps': 'EmpireCMs6.0~7.5 后台导入模型代码执行', 'name': 'EmpireCMs6.0~7.5 后台导入模型代码执行', 'determine': ['e/class/EmpireCMS_version.php', 'e/search/index.php', 'e/member/EditInfo/index.php', 'e/ViewImg/index.html'], 'version': {'type': 'file', 'file': 'e/class/EmpireCMS_version.php', 'regular': "EmpireCMS_VERSION.+'(\\d+.\\d+)'", 'vul_version': '6.0~7.5', 'ver_type': 'range'}, 'repair_file': {'type': 'version', 'file': []}, 'repair': '漏洞参考:https://blog.csdn.net/ws13129/article/details/90071260 官方暂未有修复方案'}, {'cms_list': [], 'dangerous': '2', 'cms_name': 'discuz', 'ps': 'Discuz utility组件对外访问', 'name': 'Discuz utility组件对外访问', 'determine': ['uc_client/client.php', 'uc_server/lib/uccode.class.php', 'uc_server/model/version.php', 'source/discuz_version.php'], 'version': {'type': 'single_file', 'file': 'utility/convert/index.php', 'regular': "DISCUZ_RELEASE.+'(\\d+)'", 'regular_len': 0, 'vul_version': ['1'], 'ver_type': 'list'}, 'repair_file': {'type': 'single_file', 'file': [{'file': 'utility/convert/index.php', 'regular': "$source = getgpc('source') ? getgpc('source') : getgpc('s');"}]}, 'repair': '修复漏洞参考删除utility目录'}, {'cms_list': [], 'dangerous': '2', 'cms_name': 'discuz', 'ps': 'Discuz邮件认证入口CSRF以及时间限制可绕过漏洞', 'name': 'Discuz邮件认证入口CSRF以及时间限制可绕过漏洞', 'determine': ['uc_client/client.php', 'uc_server/lib/uccode.class.php', 'uc_server/model/version.php', 'source/discuz_version.php'], 'version': {'type': 'file', 'file': 'source/discuz_version.php', 'regular': "DISCUZ_RELEASE.+'(\\d+)'", 'regular_len': 0, 'vul_version': ['20210816', '20210630', '20210520', '20210320', '20210119', '20200818', '20191201', '20190917'], 'ver_type': 'list'}, 'repair_file': {'type': 'file', 'file': [{'file': 'source/admincp/admincp_setting.php', 'regular': "showsetting('setting_permissions_mailinterval', 'settingnew[mailinterval]', $setting['mailinterval'], 'text');"}]}, 'repair': '修复漏洞参考https://gitee.com/Discuz/DiscuzX/pulls/1276/commits'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'discuz', 'ps': 'Discuz 报错注入SQL', 'name': 'Discuz 报错注入SQL', 'determine': ['uc_client/client.php', 'uc_server/lib/uccode.class.php', 'uc_server/model/version.php', 'source/discuz_version.php'], 'version': {'type': 'file', 'file': 'source/discuz_version.php', 'regular': "DISCUZ_RELEASE.+'(\\d+)'", 'regular_len': 0, 'vul_version': ['20211124', '20211022', '20210926', '20210917', '20210816', '20210630', '20210520', '20210320', '20210119', '20200818', '20191201', '20190917'], 'ver_type': 'list'}, 'repair_file': {'type': 'file', 'file': [{'file': 'api/uc.php', 'regular': 'if($len > 22 || $len < 3 || preg_match("/\\s+|^c:\\con\\con|[%,\\*"\\s\\<\\>\\&\\(\\)\']/is", $get[\'newusername\']))'}]}, 'repair': '修复漏洞参考https://gitee.com/Discuz/DiscuzX/pulls/1349'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'discuz', 'ps': 'Discuz备份恢复功能执行任意SQL漏洞', 'name': 'Discuz备份恢复功能执行任意SQL漏洞', 'determine': ['uc_client/client.php', 'uc_server/lib/uccode.class.php', 'uc_server/model/version.php', 'source/discuz_version.php'], 'version': {'type': 'file', 'file': 'source/discuz_version.php', 'regular': "DISCUZ_RELEASE.+'(\\d+)'", 'regular_len': 0, 'vul_version': ['20211231', '20211124', '20211022', '20210926', '20210917', '20210816', '20210630', '20210520', '20210320', '20210119', '20200818', '20191201', '20190917'], 'ver_type': 'list'}, 'repair_file': {'type': 'file', 'file': [{'file': 'api/db/dbbak.php', 'regular': "if(!preg_match('/^backup_(\\d+)_\\w+$/', $get['sqlpath']) || !preg_match('/^\\d+_\\w+\\-(\\d+).sql$/', $get['dumpfile']))"}]}, 'repair': '修复漏洞参考https://gitee.com/Discuz/DiscuzX/releases/v3.4-20220131'}, {'cms_list': ['maccms10'], 'dangerous': '4', 'cms_name': 'Thinkphp', 'ps': 'thinkphp5.0.X漏洞', 'name': 'Thinkphp5.X代码执行', 'determine': ['thinkphp/base.php', 'thinkphp/library/think/App.php', 'thinkphp/library/think/Request.php'], 'version': {'type': 'file', 'file': 'thinkphp/base.php', 'regular': 'THINK_VERSION.+(\\d+.\\d+.\\d+)', 'vul_version': '5.0.0~5.0.24', 'ver_type': 'range'}, 'repair_file': {'type': 'file', 'file': [{'file': 'thinkphp/library/think/App.php', 'regular': "(!preg_match('/^[A-Za-z](\\w|\\.)*$/"}, {'file': 'thinkphp/library/think/Request.php', 'regular': "if (in_array($method, ['GET', 'POST', 'DELETE', 'PUT', 'PATCH']))"}]}, 'repair': '修复漏洞参考https://www.thinkphp.cn/topic/60693.html\n修复漏洞参考https://www.thinkphp.cn/topic/60693.html'}, {'cms_list': ['maccms10'], 'dangerous': '3', 'cms_name': 'Thinkphp', 'ps': 'Thinkphp5.0.15 SQL注入漏洞', 'name': 'Thinkphp5.0.15 SQL注入漏洞', 'determine': ['thinkphp/base.php', 'thinkphp/library/think/App.php', 'thinkphp/library/think/Request.php'], 'version': {'type': 'file', 'file': 'thinkphp/base.php', 'regular': 'THINK_VERSION.+(\\d+.\\d+.\\d+)', 'vul_version': '5.0.13~5.0.15', 'ver_type': 'range'}, 'repair_file': {'type': 'file', 'file': [{'file': 'thinkphp/library/think/db/Builder.php', 'regular': "if ($key == $val[1]) {\n                        $result[$item] = $this->parseKey($val[1]) . '+' . floatval($val[2]);\n                    }"}]}, 'repair': '修复漏洞https://github.com/top-think/framework/commit/363fd4d90312f2cfa427535b7ea01a097ca8db1b'}, {'cms_list': ['maccms10'], 'dangerous': '3', 'cms_name': 'Thinkphp', 'ps': 'Thinkphp5.0.10 SQL注入漏洞', 'name': 'Thinkphp5.0.10 SQL注入漏洞', 'determine': ['thinkphp/base.php', 'thinkphp/library/think/App.php', 'thinkphp/library/think/Request.php'], 'version': {'type': 'file', 'file': 'thinkphp/base.php', 'regular': 'THINK_VERSION.+(\\d+.\\d+.\\d+)', 'vul_version': '5.0.10~5.0.10', 'ver_type': 'range'}, 'repair_file': {'type': 'file', 'file': [{'file': 'thinkphp/library/think/Request.php', 'regular': "preg_match('/^(EXP|NEQ|GT|EGT|LT|ELT|OR|XOR|LIKE|NOTLIKE|NOT LIKE|NOT BETWEEN|NOTBETWEEN|BETWEEN|NOTIN|NOT IN|IN)$/i"}]}, 'repair': '修复漏洞https://github.com/top-think/framework/commit/495020b7b0c16de40f20b08f2ab3be0a2b816b96'}, {'cms_list': ['maccms10'], 'dangerous': '3', 'cms_name': 'Thinkphp', 'ps': 'Thinkphp5.0.0 到 Thinkphp5.0.21 SQL注入漏洞', 'name': 'Thinkphp5.0.21 SQL注入漏洞', 'determine': ['thinkphp/base.php', 'thinkphp/library/think/App.php', 'thinkphp/library/think/Request.php'], 'version': {'type': 'file', 'file': 'thinkphp/base.php', 'regular': 'THINK_VERSION.+(\\d+.\\d+.\\d+)', 'vul_version': '5.0.0~5.0.21', 'ver_type': 'range'}, 'repair_file': {'type': 'file', 'file': [{'file': 'thinkphp/library/think/db/builder/Mysql.php', 'regular': "if ($strict && !preg_match('/^[\\w\\.\\*]+$/', $key))"}]}, 'repair': '修复漏洞https://github.com/top-think/framework/commit/8652c83ea10661483217c4088b582b9f05b90c20#diff-680218f330b44eb8db590f77c9307503cd312225c5ada4dbff65b9af382498bb'}, {'cms_list': ['maccms10'], 'dangerous': '3', 'cms_name': 'Thinkphp', 'ps': 'Thinkphp5.0.18文件包含漏洞', 'name': 'Thinkphp5.0.18文件包含漏洞', 'determine': ['thinkphp/base.php', 'thinkphp/library/think/App.php', 'thinkphp/library/think/Request.php'], 'version': {'type': 'file', 'file': 'thinkphp/base.php', 'regular': 'THINK_VERSION.+(\\d+.\\d+.\\d+)', 'vul_version': '5.0.0~5.0.18', 'ver_type': 'range'}, 'repair_file': {'type': 'file', 'file': [{'file': 'thinkphp/library/think/template/driver/File.php', 'regular': '$this->cacheFile = $cacheFile;'}]}, 'repair': '修复漏洞https://github.com/top-think/framework/commit/e255100c7f162c48a22f1c2bf0890469f54f061b#diff-89dd11f3d7c96572fd8218c6566241bb206b77d17220fa569a35d45cda7b5f59'}, {'cms_list': ['maccms10'], 'dangerous': '4', 'cms_name': 'Thinkphp', 'ps': 'Thinkphp5.0.10远程代码执行', 'name': 'Thinkphp5.0.10远程代码执行', 'determine': ['thinkphp/base.php', 'thinkphp/library/think/App.php', 'thinkphp/library/think/Request.php'], 'version': {'type': 'file', 'file': 'thinkphp/base.php', 'regular': 'THINK_VERSION.+(\\d+.\\d+.\\d+)', 'vul_version': '5.0.0~5.0.10', 'ver_type': 'range'}, 'repair_file': {'type': 'file', 'file': [{'file': 'thinkphp/library/think/App.php', 'regular': '$data   = "<?php\n//" . sprintf(\'%012d\', $expire) . "\n exit();?>;'}]}, 'repair': '修复漏洞参考https://github.com/top-think/framework/commit/a217d88e38a0ec2dd33ba9d5fd53ac509f93c91a#diff-c945c42842520443a3b7bdd49df3a6ca5df44a07e9a957d85ca1475ea74f8564'}, {'cms_list': [], 'dangerous': '3', 'cms_name': 'Wordpress', 'ps': 'CVE-2022–21661 Wordpress SQL注入', 'name': 'CVE-2022–21661 Wordpress SQL注入', 'determine': ['wp-includes/version.php', 'wp-settings.php', 'wp-comments-post.php', 'wp-includes/class-wp-hook.php'], 'version': {'type': 'file', 'file': 'wp-includes/version.php', 'regular': 'wp_version.+(\\d+.\\d+.\\d+)', 'vul_version': '4.1.0~5.8.2', 'ver_type': 'range'}, 'repair_file': {'type': 'file', 'file': [{'file': 'wp-includes/class-wp-tax-query.php', 'regular': "if ( 'slug' === $query['field'] || 'name' === $query['field'] )"}]}, 'repair': '修复漏洞参考https://wordpress.org/news/2022/01/wordpress-5-8-3-security-release'}]
        return OO00O0O00O00O00O0

    def getCmsType(O0000000O0000O0O0, O00OOO0000000O0OO, OO0OOO00OO0OO000O):
        if False:
            return 10
        ''
        for O0OO0OOO0OO0OOOOO in OO0OOO00OO0OO000O['determine']:
            OO00OO0O0000O0OO0 = O00OOO0000000O0OO['path'] + '/' + O0OO0OOO0OO0OOOOO
            if not os.path.exists(OO00OO0O0000O0OO0):
                return False
        if 'cms_name' in O00OOO0000000O0OO:
            if O00OOO0000000O0OO['cms_name'] != OO0OOO00OO0OO000O['cms_name']:
                if not OO0OOO00OO0OO000O['cms_name'] in OO0OOO00OO0OO000O['cms_list']:
                    return False
        OOO0OOOOOO0000000 = O0000000O0000O0O0.getCmsVersion(O00OOO0000000O0OO, OO0OOO00OO0OO000O)
        if not OOO0OOOOOO0000000:
            return False
        O00OOO0000000O0OO['version_info'] = OOO0OOOOOO0000000
        if not O0000000O0000O0O0.getVersionInfo(OOO0OOOOOO0000000, OO0OOO00OO0OO000O['version']):
            return False
        O00OOO0000000O0OO['cms_name'] = OO0OOO00OO0OO000O['cms_name']
        OO0OO0000OOOOOOO0 = O0000000O0000O0O0.getCmsVersionVulFix(O00OOO0000000O0OO, OO0OOO00OO0OO000O)
        if not OO0OO0000OOOOOOO0:
            return False
        O00OOO0000000O0OO['is_vufix'] = True
        return True

    def getVersionInfo(O0O0O0O00O00O0OO0, O0O0OO0O0OOO0OOOO, O0OO0OOOOOO0O00O0):
        if False:
            for i in range(10):
                print('nop')
        ''
        if O0OO0OOOOOO0O00O0['ver_type'] == 'range':
            try:
                O0OO0OOOOOO0O00O0 = O0OO0OOOOOO0O00O0['vul_version']
                (OOOO00O0OO00O0OOO, OOO000O00OOOO0000) = O0OO0OOOOOO0O00O0.split('~')
                if O0O0OO0O0OOO0OOOO.split('.')[0] >= OOOO00O0OO00O0OOO.split('.')[0] and O0O0OO0O0OOO0OOOO.split('.')[0] <= OOO000O00OOOO0000.split('.')[0]:
                    OOOO00O0OO00O0OOO = ''.join(OOOO00O0OO00O0OOO.split('.'))
                    OOO000O00OOOO0000 = ''.join(OOO000O00OOOO0000.split('.'))
                    O0O0OO0O0OOO0OOOO = ''.join(O0O0OO0O0OOO0OOOO.split('.'))
                    if O0O0OO0O0OOO0OOOO >= OOOO00O0OO00O0OOO and O0O0OO0O0OOO0OOOO <= OOO000O00OOOO0000:
                        return True
                return False
            except:
                return False
        elif O0OO0OOOOOO0O00O0['ver_type'] == 'list':
            if O0O0OO0O0OOO0OOOO in O0OO0OOOOOO0O00O0['vul_version']:
                return True
            return False

    def getCmsVersion(OO000O0O00OOOOOOO, O000O000000OO00OO, OOO0OO0OO00OOO0O0):
        if False:
            while True:
                i = 10
        ''
        OOO0OO00OOOO0OOO0 = OOO0OO0OO00OOO0O0['version']
        if 'regular_len' in OOO0OO00OOOO0OOO0:
            OOO000000O0O00O00 = OOO0OO00OOOO0OOO0['regular_len']
        else:
            OOO000000O0O00O00 = 0
        if OOO0OO00OOOO0OOO0['type'] == 'file':
            O000OOOO0O0O000O0 = O000O000000OO00OO['path'] + '/' + OOO0OO00OOOO0OOO0['file']
            if os.path.exists(O000OOOO0O0O000O0):
                OO000O0OO0O000O00 = public.ReadFile(O000OOOO0O0O000O0)
                if OO000O0OO0O000O00 and re.search(OOO0OO00OOOO0OOO0['regular'], OO000O0OO0O000O00):
                    if not 'cms_name' in O000O000000OO00OO:
                        O000O000000OO00OO['cms_name'] = OOO0OO0OO00OOO0O0['cms_name']
                    return re.findall(OOO0OO00OOOO0OOO0['regular'], OO000O0OO0O000O00)[OOO000000O0O00O00]
        elif OOO0OO00OOOO0OOO0['type'] == 'single_file':
            return '1'
        elif OOO0OO00OOOO0OOO0['type'] == 'is_file':
            O000OOOO0O0O000O0 = O000O000000OO00OO['path'] + '/' + OOO0OO00OOOO0OOO0['file']
            if os.path.exists(O000OOOO0O0O000O0):
                return '1'
        return False

    def getCmsVersionVulFix(OO00OO000OOOO0OO0, O0OOO0OOO000O000O, OO000O0O0O0O0000O):
        if False:
            while True:
                i = 10
        ''
        OOOO0O00O00000O0O = OO000O0O0O0O0000O['repair_file']
        if OOOO0O00O00000O0O['type'] == 'file':
            for O0OOOOOOOOOO00OOO in OOOO0O00O00000O0O['file']:
                OOO0O0OOOO0OOOO00 = O0OOO0OOO000O000O['path'] + '/' + O0OOOOOOOOOO00OOO['file']
                if os.path.exists(OOO0O0OOOO0OOOO00):
                    O0OO0OOO00OOOOOOO = public.ReadFile(OOO0O0OOOO0OOOO00)
                    if not O0OOOOOOOOOO00OOO['regular'] in O0OO0OOO00OOOOOOO:
                        return True
        elif OOOO0O00O00000O0O['type'] == 'single_file':
            for O0OOOOOOOOOO00OOO in OOOO0O00O00000O0O['file']:
                OOO0O0OOOO0OOOO00 = O0OOO0OOO000O000O['path'] + '/' + O0OOOOOOOOOO00OOO['file']
                if os.path.exists(OOO0O0OOOO0OOOO00):
                    O0OO0OOO00OOOOOOO = public.ReadFile(OOO0O0OOOO0OOOO00)
                    if O0OOOOOOOOOO00OOO['regular'] in O0OO0OOO00OOOOOOO:
                        return True
        elif OOOO0O00O00000O0O['type'] == 'version':
            return True
        elif OOOO0O00O00000O0O['type'] == 'is_file':
            for O0OOOOOOOOOO00OOO in OOOO0O00O00000O0O['file']:
                OOO0O0OOOO0OOOO00 = O0OOO0OOO000O000O['path'] + '/' + O0OOOOOOOOOO00OOO['file']
                if os.path.exists(OOO0O0OOOO0OOOO00):
                    return True
        elif OOOO0O00O00000O0O['type'] == 'phpshell':
            for O0OOOOOOOOOO00OOO in OOOO0O00O00000O0O['file']:
                try:
                    OOO0O0OOOO0OOOO00 = O0OOO0OOO000O000O['path'] + '/' + O0OOOOOOOOOO00OOO['file']
                    public.WriteFile(OOO0O0OOOO0OOOO00, O0OOOOOOOOOO00OOO['phptext'])
                    O0O00OO0OOO0O0OO0 = os.path.dirname(OOO0O0OOOO0OOOO00)
                    OOOO000OO0O0O00O0 = os.path.basename(OOO0O0OOOO0OOOO00)
                    OOO0O0OO00O0OO0O0 = public.ExecShell('cd %s && php %s' % (O0O00OO0OOO0O0OO0, OOOO000OO0O0O00O0))
                    if len(OOO0O0OO00O0OO0O0) <= 0:
                        return False
                    if O0OOOOOOOOOO00OOO['result'] in OOO0O0OO00O0OO0O0[0]:
                        os.remove(OOO0O0OOOO0OOOO00)
                        return True
                    else:
                        os.remove(OOO0O0OOOO0OOOO00)
                except:
                    continue
        return False

    def getWebInfo(O00OO00OO0O0000OO, O0OO0O0O0OO0O0OO0):
        if False:
            return 10
        ''
        return public.M('sites').where('project_type=?', 'PHP').select()

    def startScan(O00OO0OO00OO000O0, OOOO0O00O0O0OOO00):
        if False:
            while True:
                i = 10
        ''
        OO0O0OOO0OO0OO0O0 = O00OO0OO00OO000O0.__O0000O00000O000O0()
        O0O00O00000O00OO0 = int(time.time())
        O0O0000OOO000OOO0 = O00OO0OO00OO000O0.getWebInfo(None)
        OO000O0000O000OOO = O00OO0OO00OO000O0.get_config()
        O00O0OO0O0OO00OO0 = []
        for OO0O00O0O00O0OOOO in O0O0000OOO000OOO0:
            for O0OOOOO0OOO000OO0 in OO000O0000O000OOO:
                OOOO0000OOO0O0000 = O0OOOOO0OOO000OO0
                if 'cms_name' in OO0O00O0O00O0OOOO:
                    if OO0O00O0O00O0OOOO['cms_name'] != O0OOOOO0OOO000OO0['cms_name']:
                        if not OO0O00O0O00O0OOOO['cms_name'] in O0OOOOO0OOO000OO0['cms_list']:
                            continue
                if O00OO0OO00OO000O0.getCmsType(OO0O00O0O00O0OOOO, OOOO0000OOO0O0000):
                    if not 'cms' in OO0O00O0O00O0OOOO:
                        OO0O00O0O00O0OOOO['cms'] = []
                        OO0O00O0O00O0OOOO['cms'].append(O0OOOOO0OOO000OO0)
                    else:
                        OO0O00O0O00O0OOOO['cms'].append(O0OOOOO0OOO000OO0)
                elif not 'cms' in OO0O00O0O00O0OOOO:
                    OO0O00O0O00O0OOOO['cms'] = []
            if not 'is_vufix' in OO0O00O0O00O0OOOO:
                OO0O00O0O00O0OOOO['is_vufix'] = False
        for O0O0O0OOOO0000O0O in O0O0000OOO000OOO0:
            if O0O0O0OOOO0000O0O['is_vufix']:
                O00O0OO0O0OO00OO0.append(O0O0O0OOOO0000O0O)
        cache.set('scaing_info', O00O0OO0O0OO00OO0, 1600)
        cache.set('scaing_info_time', O0O00O00000O00OO0, 1600)
        O00OOOOOOOO0OOOO0 = {'info': O00O0OO0O0OO00OO0, 'time': O0O00O00000O00OO0, 'is_pay': OO0O0OOO0OO0OO0O0}
        return O00OOOOOOOO0OOOO0

    def list(OOO00000O000OO00O, OO00O00O00O0OO000):
        if False:
            while True:
                i = 10
        ''
        O000OO00OOOOOO000 = OOO00000O000OO00O.__O0000O00000O000O0()
        if not cache.get('scaing_info') or not cache.get('scaing_info_time'):
            return OOO00000O000OO00O.startScan(None)
        O0O00OOO00O00OO00 = {'info': cache.get('scaing_info'), 'time': cache.get('scaing_info_time'), 'is_pay': O000OO00OOOOOO000}
        return O0O00OOO00O00OO00

    def startAweb(O000OOO0O0OOO00O0, OO000O00OO000OO00):
        if False:
            while True:
                i = 10
        ''
        O0OOO0O0OO00O0OOO = public.M('sites').where('project_type=? and name=?', ('PHP', OO000O00OO000OO00.name)).count()
        if not O0OOO0O0OO00O0OOO:
            return public.returnMsg(False, '当前网站不存在')
        return O000OOO0O0OOO00O0.startScan(None)

    def startScanWeb(OO0O00OOO0OOO0OO0, OO000O000OOO0O0OO):
        if False:
            for i in range(10):
                print('nop')
        ''
        O000O0O0OOO000000 = public.M('sites').where('project_type=? and name=?', ('PHP', OO000O000OOO0O0OO.name)).count()
        if not O000O0O0OOO000000:
            return public.returnMsg(False, '当前网站不存在')
        O000O0O0OOO000000 = public.M('sites').where('project_type=? and name=?', ('PHP', OO000O000OOO0O0OO.name)).select()
        OO00OO0OO000OOO0O = OO0O00OOO0OOO0OO0.get_config()
        for OO0O000O0O00000O0 in O000O0O0OOO000000:
            for O00O000OOO00OOO00 in OO00OO0OO000OOO0O:
                OO0O0OO0OOO000O0O = O00O000OOO00OOO00
                if OO0O00OOO0OOO0OO0.getCmsType(OO0O000O0O00000O0, OO0O0OO0OOO000O0O):
                    if not 'cms' in OO0O000O0O00000O0:
                        OO0O000O0O00000O0['cms'] = []
                        OO0O000O0O00000O0['cms'].append(O00O000OOO00OOO00)
                    else:
                        OO0O000O0O00000O0['cms'].append(O00O000OOO00OOO00)
                    break
                elif not 'cms' in OO0O000O0O00000O0:
                    OO0O000O0O00000O0['cms'] = []
            if not 'is_vufix' in OO0O000O0O00000O0:
                OO0O000O0O00000O0['is_vufix'] = False
        return public.returnMsg(True, O000O0O0OOO000000)

    def startPath(OOO0O0O0OOO00000O, O0O0OO0O00O00O0OO):
        if False:
            print('Hello World!')
        ''
        OOO0OOOO0OOOO0O00 = O0O0OO0O00O00O0OO.path.strip()
        if not os.path.exists(OOO0OOOO0OOOO0O00):
            return public.returnMsg(False, '目录不存在')
        OOO00O00OOOOOOO00 = OOO0O0O0OOO00000O.get_config()
        OO0O0OOOOO0OO0OO0 = [{'path': OOO0OOOO0OOOO0O00, 'name': OOO0OOOO0OOOO0O00}]
        for O000O00OO00OOO00O in OO0O0OOOOO0OO0OO0:
            for O00O0O000O000OO0O in OOO00O00OOOOOOO00:
                OO0O00O0O00000OO0 = O00O0O000O000OO0O
                if OOO0O0O0OOO00000O.getCmsType(O000O00OO00OOO00O, OO0O00O0O00000OO0):
                    if not 'cms' in O000O00OO00OOO00O:
                        O000O00OO00OOO00O['cms'] = []
                        O000O00OO00OOO00O['cms'].append(O00O0O000O000OO0O)
                    else:
                        O000O00OO00OOO00O['cms'].append(O00O0O000O000OO0O)
                    break
                elif not 'cms' in O000O00OO00OOO00O:
                    O000O00OO00OOO00O['cms'] = []
            if not 'is_vufix' in O000O00OO00OOO00O:
                O000O00OO00OOO00O['is_vufix'] = False
        return public.returnMsg(True, OO0O0OOOOO0OO0OO0)