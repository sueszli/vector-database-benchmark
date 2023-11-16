import json
import os, time
from projectModel.base import projectBase
import public
from BTPanel import cache

class main(projectBase):
    __O000OO000OOOOOO00 = public.Md5('clear' + time.strftime('%Y-%m-%d'))
    __OOOO0O0OOO00O0O0O = '/www/server/panel/config/clear_log.json'
    __OO000OOOOO0OOO0OO = '/www/server/panel/data/clear'
    __OO00O000000OO0000 = public.to_string([27492, 21151, 33021, 20026, 20225, 19994, 29256, 19987, 20139, 21151, 33021, 65292, 35831, 20808, 36141, 20080, 20225, 19994, 29256])

    def __init__(OO0OO000O0OO0000O):
        if False:
            i = 10
            return i + 15
        if not os.path.exists(OO0OO000O0OO0000O.__OO000OOOOO0OOO0OO):
            os.makedirs(OO0OO000O0OO0000O.__OO000OOOOO0OOO0OO, 384)

    def __O000OO00O0OO0O00O(O0O00O00O00OO0OO0):
        if False:
            return 10
        from pluginAuth import Plugin
        O0O00000O000OO00O = Plugin(False)
        O00000O0O0OOOOOOO = O0O00000O000OO00O.get_plugin_list()
        return int(O00000O0O0OOOOOOO['ltd']) > time.time()

    def get_config(O0OO0O00OOOOO00OO):
        if False:
            print('Hello World!')
        ''
        if not os.path.exists(O0OO0O00OOOOO00OO.__OOOO0O0OOO00O0O0O):
            O0OO0O00OOOOO00OO.write_config()
            return O0OO0O00OOOOO00OO.default_config()
        else:
            try:
                O00OOOOO0O0O00000 = json.loads(public.ReadFile(O0OO0O00OOOOO00OO.__OOOO0O0OOO00O0O0O))
            except:
                O0OO0O00OOOOO00OO.write_config()
                return O0OO0O00OOOOO00OO.default_config()
        if not cache.get(O0OO0O00OOOOO00OO.__O000OO000OOOOOO00):
            try:
                import requests
                O00OOOOO0O0O00000 = requests.get('https://www.bt.cn/api/bt_waf/clearLog').json()
                cache.set(O0OO0O00OOOOO00OO.__O000OO000OOOOOO00, '1', 1800)
                O0OO0O00OOOOO00OO.write_config(O00OOOOO0O0O00000)
            except:
                return O0OO0O00OOOOO00OO.default_config()
            return O00OOOOO0O0O00000
        else:
            return O00OOOOO0O0O00000

    def write_config(O0O0O0OOOO000O0O0, config=False):
        if False:
            while True:
                i = 10
        ''
        if config:
            public.WriteFile(O0O0O0OOOO000O0O0.__OOOO0O0OOO00O0O0O, json.dumps(config))
        else:
            public.WriteFile(O0O0O0OOOO000O0O0.__OOOO0O0OOO00O0O0O, json.dumps(O0O0O0OOOO000O0O0.default_config()))

    def default_config(O0O0OO0OO00O00O0O):
        if False:
            return 10
        ''
        return [{'name': 'recycle1', 'ps': '面板备份文件', 'path': '/www/backup/panel', 'type': 'dir', 'is_del': True, 'find': [], 'exclude': [], 'is_config': False, 'regular': '', 'subdirectory': False, 'result': [], 'size': 0}, {'name': 'recycle2', 'ps': '面板文件备份', 'path': '/www/backup/file_history', 'type': 'dir', 'is_del': True, 'find': [], 'exclude': [], 'is_config': False, 'regular': '', 'subdirectory': False, 'result': [], 'size': 0}, {'name': 'docker', 'ps': 'Docker容器日志', 'path': '/var/lib/docker/containers', 'type': 'file', 'is_del': False, 'find': ['-json.log'], 'exclude': [], 'is_config': False, 'regular': '', 'subdirectory': False, 'result': [], 'size': 0}, {'name': 'openrasp', 'ps': 'openrasp日志', 'path': ['/opt/rasp55/logs/alarm', '/opt/rasp55/logs/policy', '/opt/rasp55/logs/plugin', '/opt/rasp56/logs/alarm', '/opt/rasp56/logs/policy', '/opt/rasp56/logs/plugin', '/opt/rasp70/logs/alarm', '/opt/rasp70/logs/policy', '/opt/rasp70/logs/plugin', '/opt/rasp71/logs/alarm', '/opt/rasp71/logs/policy', '/opt/rasp72/logs/plugin', '/opt/rasp73/logs/alarm', '/opt/rasp73/logs/policy', '/opt/rasp73/logs/plugin', '/opt/rasp74/logs/alarm', '/opt/rasp74/logs/policy', '/opt/rasp74/logs/plugin'], 'type': 'file', 'is_del': True, 'find': ['.log'], 'exclude': [], 'is_config': False, 'regular': '', 'subdirectory': False, 'result': [], 'size': 0}, {'name': 'springboot', 'ps': 'springboot日志', 'path': '/var/tmp/springboot/vhost/logs', 'type': 'file', 'is_del': False, 'find': ['.log'], 'exclude': [], 'is_config': False, 'regular': '', 'subdirectory': False, 'result': [], 'size': 0}, {'name': 'aliyun', 'ps': '阿里云Agent日志', 'path': ['/usr/local/share/aliyun-assist/2.2.3.247/log', '/usr/local/share/aliyun-assist/2.2.3.256/log'], 'type': 'file', 'is_del': True, 'find': ['.log'], 'exclude': [], 'is_config': False, 'regular': '', 'subdirectory': False, 'result': [], 'size': 0}, {'name': 'qcloud', 'ps': '腾讯云Agent日志', 'path': ['/usr/local/qcloud/tat_agent/log', '/usr/local/qcloud/YunJing/log', '/usr/local/qcloud/stargate/logs', '/usr/local/qcloud/monitor/barad/log'], 'type': 'file', 'is_del': True, 'find': ['.log'], 'exclude': [], 'is_config': False, 'regular': '', 'subdirectory': False, 'result': [], 'size': 0}, {'name': 'crontab', 'ps': '计划任务日志', 'path': '/www/server/cron', 'type': 'file', 'is_del': True, 'find': ['.log'], 'exclude': [], 'is_config': False, 'regular': '', 'subdirectory': False, 'result': [], 'size': 0}, {'name': 'tomcat', 'ps': 'tomcat日志', 'path': ['/usr/local/bttomcat/tomcat8/logs', '/usr/local/bttomcat/tomcat9/logs', '/usr/local/bttomcat/tomcat7/logs'], 'type': 'file', 'is_del': True, 'find': ['.log'], 'exclude': [], 'is_config': False, 'regular': '', 'subdirectory': False, 'result': [], 'size': 0}, {'name': 'panellog', 'ps': '面板日志', 'path': '/www/server/panel/logs/request', 'type': 'file', 'is_del': True, 'find': ['.json.gz'], 'exclude': [], 'is_config': False, 'regular': '', 'subdirectory': False, 'result': [], 'size': 0}, {'name': 'recycle3', 'ps': '回收站', 'path': '/www/Recycle_bin', 'type': 'dir', 'is_del': True, 'find': [], 'exclude': [], 'is_config': False, 'regular': '', 'subdirectory': False, 'result': [], 'size': 0}, {'name': 'maillog', 'ps': '邮件日志', 'path': '/var/spool/mail', 'type': 'file', 'is_del': True, 'find': [], 'exclude': [], 'is_config': False, 'regular': '', 'subdirectory': False, 'result': [], 'size': 0}, {'name': 'btwaflog', 'ps': '防火墙日志', 'path': ['/www/wwwlogs/btwaf', '/www/server/btwaf/totla_db/http_log'], 'type': 'file', 'is_del': True, 'find': [], 'exclude': [], 'is_config': False, 'regular': '', 'subdirectory': False, 'result': [], 'size': 0}, {'name': 'weblog', 'ps': '网站日志', 'path': '/www/wwwlogs', 'type': 'file', 'is_del': True, 'find': [], 'exclude': [], 'is_config': False, 'regular': '', 'subdirectory': False, 'result': [], 'size': 0}, {'name': 'syslog', 'ps': '系统日志', 'path': ['/var/log/audit', '/var/log'], 'type': 'file', 'is_del': True, 'find': [], 'exclude': [], 'is_config': False, 'regular': '', 'subdirectory': False, 'result': [], 'size': 0}, {'name': 'package', 'ps': '面板遗留文件', 'path': '/www/server/panel/package', 'type': 'file', 'is_del': True, 'find': ['.zip'], 'exclude': [], 'is_config': False, 'regular': '', 'subdirectory': False, 'result': [], 'size': 0}, {'name': 'mysqllog', 'ps': '数据库日志', 'path': '/www/server/data', 'type': 'file', 'is_del': True, 'find': ['mysql-bin.00'], 'exclude': [], 'is_config': False, 'regular': '', 'subdirectory': False, 'result': [], 'size': 0}, {'name': 'session', 'ps': 'session日志', 'path': '/tmp', 'type': 'file', 'is_del': True, 'find': ['sess_'], 'exclude': [], 'is_config': False, 'regular': '', 'subdirectory': False, 'result': [], 'size': 0}, {'name': 'bttotal', 'ps': '网站监控报表日志', 'path': '/www/server/total/logs', 'type': 'file', 'is_del': True, 'find': ['_bt_'], 'exclude': [], 'is_config': False, 'regular': '', 'subdirectory': False, 'result': [], 'size': 0}]

    def tosize(O00O00O00O000O0OO, O0OOO00O00OOOOOO0):
        if False:
            return 10
        ''
        O000O0O0OOOO0O0OO = ['b', 'KB', 'MB', 'GB', 'TB']
        for OO0O00OOO000O0O0O in O000O0O0OOOO0O0OO:
            if O0OOO00O00OOOOOO0 < 1024:
                return str(int(O0OOO00O00OOOOOO0)) + OO0O00OOO000O0O0O
            O0OOO00O00OOOOOO0 = O0OOO00O00OOOOOO0 / 1024
        return '0b'

    def any_size(OO00O0OOO0O0OOOOO, OOOO0O000O000O0O0):
        if False:
            i = 10
            return i + 15
        ''
        OOOO0O000O000O0O0 = str(OOOO0O000O000O0O0)
        OO0OOO0OOOO0OO0O0 = OOOO0O000O000O0O0[-1]
        try:
            OOO00OO0O00OO000O = float(OOOO0O000O000O0O0[0:-1])
        except:
            OOO00OO0O00OO000O = 0
            return OOOO0O000O000O0O0
        O00000O0O0000OOOO = ['b', 'K', 'M', 'G', 'T']
        if OO0OOO0OOOO0OO0O0 in O00000O0O0000OOOO:
            if OO0OOO0OOOO0OO0O0 == 'b':
                return int(OOO00OO0O00OO000O)
            elif OO0OOO0OOOO0OO0O0 == 'K':
                OOO00OO0O00OO000O = OOO00OO0O00OO000O * 1024
                return int(OOO00OO0O00OO000O)
            elif OO0OOO0OOOO0OO0O0 == 'M':
                OOO00OO0O00OO000O = OOO00OO0O00OO000O * 1024 * 1024
                return int(OOO00OO0O00OO000O)
            elif OO0OOO0OOOO0OO0O0 == 'G':
                OOO00OO0O00OO000O = OOO00OO0O00OO000O * 1024 * 1024 * 1024
                return int(OOO00OO0O00OO000O)
            elif OO0OOO0OOOO0OO0O0 == 'T':
                OOO00OO0O00OO000O = OOO00OO0O00OO000O * 1024 * 1024 * 1024 * 1024
                return int(OOO00OO0O00OO000O)
            else:
                return int(OOO00OO0O00OO000O)
        else:
            return '0b'

    def get_path_file(O000OO0OO0OOOO0OO, O00O000O0OOOOO0OO):
        if False:
            while True:
                i = 10
        ''
        if type(O00O000O0OOOOO0OO['path']) == list:
            for O00000OO0000OO00O in O00O000O0OOOOO0OO['path']:
                if os.path.exists(O00000OO0000OO00O):
                    for OOOO0OOO0OOOO0OO0 in os.listdir(O00000OO0000OO00O):
                        OOOO0O0000OOOO0OO = O00000OO0000OO00O + '/' + OOOO0OOO0OOOO0OO0
                        if O00O000O0OOOOO0OO['type'] == 'file':
                            if os.path.isfile(OOOO0O0000OOOO0OO):
                                O000O000O0O0OOOO0 = {}
                                O0OOO0O000O0O0OOO = os.path.getsize(OOOO0O0000OOOO0OO)
                                if O0OOO0O000O0O0OOO >= 100:
                                    O000O000O0O0OOOO0['size'] = O000OO0OO0OOOO0OO.tosize(O0OOO0O000O0O0OOO)
                                    O000O000O0O0OOOO0['count_size'] = O0OOO0O000O0O0OOO
                                    O00O000O0OOOOO0OO['size'] += O0OOO0O000O0O0OOO
                                    O000O000O0O0OOOO0['name'] = OOOO0O0000OOOO0OO
                                    O00O000O0OOOOO0OO['result'].append(O000O000O0O0OOOO0)
        elif os.path.exists(O00O000O0OOOOO0OO['path']):
            for OOOO0OOO0OOOO0OO0 in os.listdir(O00O000O0OOOOO0OO['path']):
                OOOO0O0000OOOO0OO = O00O000O0OOOOO0OO['path'] + '/' + OOOO0OOO0OOOO0OO0
                if O00O000O0OOOOO0OO['type'] == 'file':
                    if os.path.isfile(OOOO0O0000OOOO0OO):
                        O000O000O0O0OOOO0 = {}
                        O0OOO0O000O0O0OOO = os.path.getsize(OOOO0O0000OOOO0OO)
                        if O0OOO0O000O0O0OOO >= 100:
                            O000O000O0O0OOOO0['size'] = O000OO0OO0OOOO0OO.tosize(O0OOO0O000O0O0OOO)
                            O000O000O0O0OOOO0['count_size'] = O0OOO0O000O0O0OOO
                            O00O000O0OOOOO0OO['size'] += O0OOO0O000O0O0OOO
                            O000O000O0O0OOOO0['name'] = OOOO0O0000OOOO0OO
                            O00O000O0OOOOO0OO['result'].append(O000O000O0O0OOOO0)
                elif O00O000O0OOOOO0OO['type'] == 'dir':
                    if os.path.isdir(OOOO0O0000OOOO0OO):
                        O0OOO0O000O0O0OOO = public.ExecShell('du -sh  %s' % OOOO0O0000OOOO0OO)[0].split()[0]
                        O000O000O0O0OOOO0 = {}
                        O000O000O0O0OOOO0['dir'] = 'dir'
                        O000O000O0O0OOOO0['size'] = O0OOO0O000O0O0OOO
                        O000O000O0O0OOOO0['count_size'] = O000OO0OO0OOOO0OO.any_size(O0OOO0O000O0O0OOO)
                        if O000O000O0O0OOOO0['count_size'] < 100:
                            continue
                        if O00O000O0OOOOO0OO['path'] == '/www/Recycle_bin':
                            OOOO0O0000OOOO0OO = OOOO0OOO0OOOO0OO0.split('_t_')[0].replace('_bt_', '/')
                            O000O000O0O0OOOO0['filename'] = O00O000O0OOOOO0OO['path'] + '/' + OOOO0OOO0OOOO0OO0
                            O000O000O0O0OOOO0['name'] = OOOO0O0000OOOO0OO
                        else:
                            O000O000O0O0OOOO0['filename'] = O00O000O0OOOOO0OO['path'] + '/' + OOOO0OOO0OOOO0OO0
                            O000O000O0O0OOOO0['name'] = os.path.basename(OOOO0O0000OOOO0OO)
                        O00O000O0OOOOO0OO['size'] += O000O000O0O0OOOO0['count_size']
                        O00O000O0OOOOO0OO['result'].append(O000O000O0O0OOOO0)
                    else:
                        O0OO000O0OO000000 = os.path.getsize(OOOO0O0000OOOO0OO)
                        if O0OO000O0OO000000 < 100:
                            continue
                        O000O000O0O0OOOO0 = {}
                        O000O000O0O0OOOO0['filename'] = OOOO0O0000OOOO0OO
                        OOOO0O0000OOOO0OO = os.path.basename(OOOO0O0000OOOO0OO)
                        O000O000O0O0OOOO0['count_size'] = O0OO000O0OO000000
                        if O00O000O0OOOOO0OO['path'] == '/www/Recycle_bin':
                            OOOO0O0000OOOO0OO = OOOO0OOO0OOOO0OO0.split('_t_')[0].replace('_bt_', '/')
                            O000O000O0O0OOOO0['name'] = OOOO0O0000OOOO0OO
                        else:
                            O000O000O0O0OOOO0['name'] = OOOO0O0000OOOO0OO
                        O000O000O0O0OOOO0['size'] = O000OO0OO0OOOO0OO.tosize(O0OO000O0OO000000)
                        O00O000O0OOOOO0OO['size'] += O000O000O0O0OOOO0['count_size']
                        O00O000O0OOOOO0OO['result'].append(O000O000O0O0OOOO0)
        return O00O000O0OOOOO0OO

    def get_path_find(OO0O0OO0O00OO0000, O00O00000OOOO0O00):
        if False:
            for i in range(10):
                print('nop')
        ''
        if type(O00O00000OOOO0O00['path']) == list:
            for O000000OO00OO00O0 in O00O00000OOOO0O00['path']:
                if os.path.exists(O000000OO00OO00O0):
                    for OOOOOOO00O00O0O0O in os.listdir(O000000OO00OO00O0):
                        for O0000O0OOOO0OO0O0 in O00O00000OOOO0O00['find']:
                            if OOOOOOO00O00O0O0O.find(O0000O0OOOO0OO0O0) == -1:
                                continue
                            O0OOOOO00O0O00O0O = O000000OO00OO00O0 + '/' + OOOOOOO00O00O0O0O
                            if not os.path.exists(O0OOOOO00O0O00O0O):
                                continue
                            OO000OOO00OOO0O00 = os.path.getsize(O0OOOOO00O0O00O0O)
                            if OO000OOO00OOO0O00 < 1024:
                                continue
                            OO00OOOOOO0000000 = {}
                            OO00OOOOOO0000000['name'] = O0OOOOO00O0O00O0O
                            OO00OOOOOO0000000['count_size'] = OO000OOO00OOO0O00
                            O00O00000OOOO0O00['size'] += OO000OOO00OOO0O00
                            OO00OOOOOO0000000['size'] = OO0O0OO0O00OO0000.tosize(OO000OOO00OOO0O00)
                            O00O00000OOOO0O00['result'].append(OO00OOOOOO0000000)
        elif os.path.exists(O00O00000OOOO0O00['path']):
            for OOOOOOO00O00O0O0O in os.listdir(O00O00000OOOO0O00['path']):
                for O0000O0OOOO0OO0O0 in O00O00000OOOO0O00['find']:
                    O0OOOOO00O0O00O0O = O00O00000OOOO0O00['path'] + '/' + OOOOOOO00O00O0O0O
                    if O00O00000OOOO0O00['path'] == '/var/lib/docker/containers':
                        O0OOOOO00O0O00O0O = O0OOOOO00O0O00O0O + '/' + OOOOOOO00O00O0O0O + '-json.log'
                        if os.path.exists(O0OOOOO00O0O00O0O):
                            OO000OOO00OOO0O00 = os.path.getsize(O0OOOOO00O0O00O0O)
                            if OO000OOO00OOO0O00 < 1024:
                                continue
                            OO00OOOOOO0000000 = {}
                            OO00OOOOOO0000000['name'] = O0OOOOO00O0O00O0O
                            OO00OOOOOO0000000['count_size'] = OO000OOO00OOO0O00
                            O00O00000OOOO0O00['size'] += OO000OOO00OOO0O00
                            OO00OOOOOO0000000['size'] = OO0O0OO0O00OO0000.tosize(OO000OOO00OOO0O00)
                            O00O00000OOOO0O00['result'].append(OO00OOOOOO0000000)
                    else:
                        if OOOOOOO00O00O0O0O.find(O0000O0OOOO0OO0O0) == -1:
                            continue
                        if not os.path.exists(O0OOOOO00O0O00O0O):
                            continue
                        OO000OOO00OOO0O00 = os.path.getsize(O0OOOOO00O0O00O0O)
                        if OO000OOO00OOO0O00 < 1024:
                            continue
                        OO00OOOOOO0000000 = {}
                        OO00OOOOOO0000000['name'] = O0OOOOO00O0O00O0O
                        OO00OOOOOO0000000['count_size'] = OO000OOO00OOO0O00
                        O00O00000OOOO0O00['size'] += OO000OOO00OOO0O00
                        OO00OOOOOO0000000['size'] = OO0O0OO0O00OO0000.tosize(OO000OOO00OOO0O00)
                        O00O00000OOOO0O00['result'].append(OO00OOOOOO0000000)
        return O00O00000OOOO0O00

    def scanning(O0000O0O000OO000O, OO0O0O0O000O00OOO):
        if False:
            while True:
                i = 10
        ''
        if not O0000O0O000OO000O.__O000OO00O0OO0O00O():
            return public.returnMsg(False, O0000O0O000OO000O.__OO00O000000OO0000)
        O0000O000OO0OO00O = 0
        O0OOO000O0OOOO00O = O0000O0O000OO000O.get_config()
        OOOO0O0OOO00O0O0O = int(time.time())
        public.WriteFile(O0000O0O000OO000O.__OO000OOOOO0OOO0OO + '/scanning', str(OOOO0O0OOO00O0O0O))
        for O00OOO00O0O00O000 in O0OOO000O0OOOO00O:
            O0O0OOOO0O000O0O0 = O0000O0O000OO000O.__OO000OOOOO0OOO0OO + '/' + O00OOO00O0O00O000['name'] + '.pl'
            if not O00OOO00O0O00O000['find'] and (not O00OOO00O0O00O000['exclude']) and (not O00OOO00O0O00O000['is_config']):
                O0000O0O000OO000O.get_path_file(O00OOO00O0O00O000)
                O00OOO00O0O00O000['time'] = int(time.time())
                O0000O000OO0OO00O += O00OOO00O0O00O000['size']
                O00OOO00O0O00O000['size_info'] = O0000O0O000OO000O.tosize(O00OOO00O0O00O000['size'])
            if O00OOO00O0O00O000['find'] and (not O00OOO00O0O00O000['exclude']) and (not O00OOO00O0O00O000['is_config']):
                O0000O0O000OO000O.get_path_find(O00OOO00O0O00O000)
                O00OOO00O0O00O000['time'] = int(time.time())
                O0000O000OO0OO00O += O00OOO00O0O00O000['size']
                O00OOO00O0O00O000['size_info'] = O0000O0O000OO000O.tosize(O00OOO00O0O00O000['size'])
            public.WriteFile(O0O0OOOO0O000O0O0, json.dumps(O00OOO00O0O00O000))
        O0O0OOOOOO0OO0OO0 = {'info': O0OOO000O0OOOO00O, 'size': O0000O000OO0OO00O, 'time': OOOO0O0OOO00O0O0O}
        return O0O0OOOOOO0OO0OO0

    def list(O0O0O0O00O0OO0OO0, OO0OO0OOOOOOO0OOO):
        if False:
            i = 10
            return i + 15
        ''
        O00O00O0O000OO0OO = 0
        OO0O0OOO0O0O0OO0O = []
        O0O0OOOO0O0O00OOO = O0O0O0O00O0OO0OO0.get_config()
        for O0000OOOOO0OOO00O in O0O0OOOO0O0O00OOO:
            OOO000O00OOO0O0O0 = O0O0O0O00O0OO0OO0.__OO000OOOOO0OOO0OO + '/' + O0000OOOOO0OOO00O['name'] + '.pl'
            if os.path.exists(OOO000O00OOO0O0O0):
                OO00O0OO000O0OO0O = json.loads(public.readFile(OOO000O00OOO0O0O0))
                OO0O0OOO0O0O0OO0O.append(OO00O0OO000O0OO0O)
                O00O00O0O000OO0OO += OO00O0OO000O0OO0O['size']
            else:
                if not O0000OOOOO0OOO00O['find'] and (not O0000OOOOO0OOO00O['exclude']) and (not O0000OOOOO0OOO00O['is_config']):
                    O0O0O0O00O0OO0OO0.get_path_file(O0000OOOOO0OOO00O)
                    O0000OOOOO0OOO00O['size_info'] = O0O0O0O00O0OO0OO0.tosize(O0000OOOOO0OOO00O['size'])
                    O00O00O0O000OO0OO += O0000OOOOO0OOO00O['size']
                if O0000OOOOO0OOO00O['find'] and (not O0000OOOOO0OOO00O['exclude']) and (not O0000OOOOO0OOO00O['is_config']):
                    O0O0O0O00O0OO0OO0.get_path_find(O0000OOOOO0OOO00O)
                    O0000OOOOO0OOO00O['time'] = time.time()
                    O00O00O0O000OO0OO += O0000OOOOO0OOO00O['size']
                    O0000OOOOO0OOO00O['size_info'] = O0O0O0O00O0OO0OO0.tosize(O0000OOOOO0OOO00O['size'])
                public.WriteFile(OOO000O00OOO0O0O0, json.dumps(O0000OOOOO0OOO00O))
                OO0O0OOO0O0O0OO0O.append(O0000OOOOO0OOO00O)
        if os.path.exists(O0O0O0O00O0OO0OO0.__OO000OOOOO0OOO0OO + '/scanning'):
            OOOO0O0OOO0000O0O = int(public.ReadFile(O0O0O0O00O0OO0OO0.__OO000OOOOO0OOO0OO + '/scanning'))
        else:
            OOOO0O0OOO0000O0O = int(time.time())
        OOOO000OO0O0000O0 = {'info': OO0O0OOO0O0O0OO0O, 'size': O00O00O0O000OO0OO, 'time': OOOO0O0OOO0000O0O}
        return OOOO000OO0O0000O0

    def remove_file(OO0OOOO00OO000OO0, OO0OOOO00000OO00O):
        if False:
            return 10
        ''
        if not OO0OOOO00OO000OO0.__O000OO00O0OO0O00O():
            return public.returnMsg(False, OO0OOOO00OO000OO0.__OO00O000000OO0000)
        O0OOO0000000O000O = 0
        O0O0O00000000OOOO = OO0OOOO00000OO00O.san_info
        for O00OOO00OOOO0OOOO in O0O0O00000000OOOO:
            if len(O00OOO00OOOO0OOOO['result']) <= 0:
                return
            for O0O00O0O0000OO000 in O00OOO00OOOO0OOOO['result']:
                try:
                    if O00OOO00OOOO0OOOO['type'] == 'dir':
                        if 'filename' in O0O00O0O0000OO000:
                            if os.path.isfile(O0O00O0O0000OO000['filename']):
                                if O00OOO00OOOO0OOOO['is_del']:
                                    os.remove(O0O00O0O0000OO000['filename'])
                                else:
                                    public.ExecShell('echo >%s' % O0O00O0O0000OO000['filename'])
                                O0OOO0000000O000O += O0O00O0O0000OO000['count_size']
                            else:
                                O0OOO0000000O000O += O0O00O0O0000OO000['count_size']
                                public.ExecShell('rm -rf %s' % O0O00O0O0000OO000['filename'])
                        else:
                            return '22'
                    elif os.path.isfile(O0O00O0O0000OO000['name']):
                        if O00OOO00OOOO0OOOO['is_del']:
                            os.remove(O0O00O0O0000OO000['name'])
                        else:
                            public.ExecShell('echo >%s' % O0O00O0O0000OO000['name'])
                        O0OOO0000000O000O += O0O00O0O0000OO000['count_size']
                    else:
                        O0OOO0000000O000O += O0O00O0O0000OO000['count_size']
                        os.rmdir(O0O00O0O0000OO000['name'])
                except:
                    continue
        OO0OOOO00OO000OO0.scanning(None)
        return public.returnMsg(True, OO0OOOO00OO000OO0.tosize(O0OOO0000000O000O))