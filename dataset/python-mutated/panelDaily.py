import os
import sys
import time
import psutil
os.chdir('/www/server/panel')
sys.path.insert(0, '/www/server/panel')
sys.path.insert(0, 'class/')
import public
from system import system
from panelPlugin import panelPlugin
from BTPanel import auth, cache

class panelDaily:

    def check_databases(O000OOOO0OOOO00O0):
        if False:
            print('Hello World!')
        ''
        OOOO000000OO0000O = ['app_usage', 'server_status', 'backup_status', 'daily']
        import sqlite3
        OOO00OOO00000O0OO = sqlite3.connect('/www/server/panel/data/system.db')
        O0OOOOOO000000OOO = OOO00OOO00000O0OO.cursor()
        O00O0O000OOOO0OOO = ','.join(["'" + OOO0O0000O0OO0O00 + "'" for OOO0O0000O0OO0O00 in OOOO000000OO0000O])
        OO0O0OOOO0000O0O0 = O0OOOOOO000000OOO.execute("SELECT name FROM sqlite_master WHERE type='table' and name in ({})".format(O00O0O000OOOO0OOO))
        OOOO000OO000OO00O = OO0O0OOOO0000O0O0.fetchall()
        O000O0OOO0O0OOOO0 = False
        OO0OOO00OOO000OO0 = []
        if OOOO000OO000OO00O:
            OO0OOO00OOO000OO0 = [O000O0OOO000OO00O[0] for O000O0OOO000OO00O in OOOO000OO000OO00O]
        if 'app_usage' not in OO0OOO00OOO000OO0:
            O000O0O000000O0OO = 'CREATE TABLE IF NOT EXISTS `app_usage` (\n                    `time_key` INTEGER PRIMARY KEY,\n                    `app` TEXT,\n                    `disks` TEXT,\n                    `addtime` DATETIME DEFAULT CURRENT_TIMESTAMP\n                )'
            O0OOOOOO000000OOO.execute(O000O0O000000O0OO)
            O000O0OOO0O0OOOO0 = True
        if 'server_status' not in OO0OOO00OOO000OO0:
            print('创建server_status表:')
            O000O0O000000O0OO = 'CREATE TABLE IF NOT EXISTS `server_status` (\n                    `status` TEXT,\n                    `addtime` DATETIME DEFAULT CURRENT_TIMESTAMP\n                )'
            O0OOOOOO000000OOO.execute(O000O0O000000O0OO)
            O000O0OOO0O0OOOO0 = True
        if 'backup_status' not in OO0OOO00OOO000OO0:
            print('创建备份状态表:')
            O000O0O000000O0OO = 'CREATE TABLE IF NOT EXISTS `backup_status` (\n                    `id` INTEGER,\n                    `target` TEXT,\n                    `status` INTEGER,\n                    `msg` TEXT DEFAULT "",\n                    `addtime` DATETIME DEFAULT CURRENT_TIMESTAMP\n                )'
            O0OOOOOO000000OOO.execute(O000O0O000000O0OO)
            O000O0OOO0O0OOOO0 = True
        if 'daily' not in OO0OOO00OOO000OO0:
            O000O0O000000O0OO = 'CREATE TABLE IF NOT EXISTS `daily` (\n                    `time_key` INTEGER,\n                    `evaluate` INTEGER,\n                    `addtime` DATETIME DEFAULT CURRENT_TIMESTAMP\n                )'
            O0OOOOOO000000OOO.execute(O000O0O000000O0OO)
            O000O0OOO0O0OOOO0 = True
        if O000O0OOO0O0OOOO0:
            OOO00OOO00000O0OO.commit()
        O0OOOOOO000000OOO.close()
        OOO00OOO00000O0OO.close()
        return True

    def get_time_key(OO00000O0000OOO00, date=None):
        if False:
            for i in range(10):
                print('nop')
        if date is None:
            date = time.localtime()
        OO00O0OO0OO0OO0O0 = 0
        OO0000OO00000O00O = '%Y%m%d'
        if type(date) == time.struct_time:
            OO00O0OO0OO0OO0O0 = int(time.strftime(OO0000OO00000O00O, date))
        if type(date) == str:
            OO00O0OO0OO0OO0O0 = int(time.strptime(date, OO0000OO00000O00O))
        return OO00O0OO0OO0OO0O0

    def store_app_usage(OOOO0O00O000O0O00, time_key=None):
        if False:
            while True:
                i = 10
        ''
        OOOO0O00O000O0O00.check_databases()
        if time_key is None:
            time_key = OOOO0O00O000O0O00.get_time_key()
        OOO0OOO0O00OO0000 = public.M('system').dbfile('system').table('app_usage')
        OOOO0OOO0O0OOOO00 = OOO0OOO0O00OO0000.field('time_key').where('time_key=?', time_key).find()
        if OOOO0OOO0O0OOOO00 and 'time_key' in OOOO0OOO0O0OOOO00:
            if OOOO0OOO0O0OOOO00['time_key'] == time_key:
                return True
        O000OOO00OOO0O0OO = public.M('sites').field('path').select()
        O00OOO00000OO00O0 = 0
        for O00O0O0OOOOO0OO00 in O000OOO00OOO0O0OO:
            OO00O0OO00OO0O000 = O00O0O0OOOOO0OO00['path']
            if OO00O0OO00OO0O000:
                O00OOO00000OO00O0 += public.get_path_size(OO00O0OO00OO0O000)
        O00000OOOOO0O0O0O = public.get_path_size('/www/server/data')
        O0O000000O0OO00O0 = public.M('ftps').field('path').select()
        OOO00O0OOO00OO0OO = 0
        for O00O0O0OOOOO0OO00 in O0O000000O0OO00O0:
            O0OOO000O0O0O0OO0 = O00O0O0OOOOO0OO00['path']
            if O0OOO000O0O0O0OO0:
                OOO00O0OOO00OO0OO += public.get_path_size(O0OOO000O0O0O0OO0)
        O000OO0OOOOOO0OOO = public.get_path_size('/www/server/panel/plugin')
        O0OOO00000OOOOOOO = ['/www/server/total', '/www/server/btwaf', '/www/server/coll', '/www/server/nginx', '/www/server/apache', '/www/server/redis']
        for OO0O000O0O0OO0000 in O0OOO00000OOOOOOO:
            O000OO0OOOOOO0OOO += public.get_path_size(OO0O000O0O0OO0000)
        OOO000O000OOO00O0 = system().GetDiskInfo2(human=False)
        OO0O000OOO0O0OO00 = ''
        O00O0O0O000O00000 = 0
        O000O00O0O0O0O0OO = 0
        for OOO00OO00000OO0O0 in OOO000O000OOO00O0:
            O0OOO0OOO00000OOO = OOO00OO00000OO0O0['path']
            if OO0O000OOO0O0OO00:
                OO0O000OOO0O0OO00 += '-'
            (O0O0OOOO0OO0OOOO0, OOO0OO00O000O0O00, O0OOOOOOOOO0OO0O0, O0O00OO00O000OOO0) = OOO00OO00000OO0O0['size']
            (OOO00OOOOO00OOO0O, O00000OO00OO0OOO0, _O00O00OO0OOO00OO0, _O0OOOOO0O0OOO000O) = OOO00OO00000OO0O0['inodes']
            OO0O000OOO0O0OO00 = '{},{},{},{},{}'.format(O0OOO0OOO00000OOO, OOO0OO00O000O0O00, O0O0OOOO0OO0OOOO0, O00000OO00OO0OOO0, OOO00OOOOO00OOO0O)
            if O0OOO0OOO00000OOO == '/':
                O00O0O0O000O00000 = O0O0OOOO0OO0OOOO0
                O000O00O0O0O0O0OO = OOO0OO00O000O0O00
        OOOO00O0O0O0OOO00 = '{},{},{},{},{},{}'.format(O00O0O0O000O00000, O000O00O0O0O0O0OO, O00OOO00000OO00O0, O00000OOOOO0O0O0O, OOO00O0OOO00OO0OO, O000OO0OOOOOO0OOO)
        OOO0O0OO000000OOO = public.M('system').dbfile('system').table('app_usage').add('time_key,app,disks', (time_key, OOOO00O0O0O0OOO00, OO0O000OOO0O0OO00))
        if OOO0O0OO000000OOO == time_key:
            return True
        return False

    def parse_app_usage_info(O00O0OO0000O00O00, O00OOO0O0OOOOOOOO):
        if False:
            i = 10
            return i + 15
        ''
        if not O00OOO0O0OOOOOOOO:
            return {}
        print(O00OOO0O0OOOOOOOO)
        (O0000OOO0O00OOOOO, OO00O00OO0OO0OO0O, O0OOO00O0O0000000, OO0O00O00OOOOO000, OO00O0000OO00000O, O0OOOOOOO0000OOO0) = O00OOO0O0OOOOOOOO['app'].split(',')
        OOO0O00O0O0OO0000 = O00OOO0O0OOOOOOOO['disks'].split('-')
        OOO00OOO0O0O00OOO = {}
        for OOO0O0OO0OOO000OO in OOO0O00O0O0OO0000:
            (O00OO00000O000000, OOOO0OOOOOO00OOOO, O0O00000O00OO0OO0, O00000OO00OO0OO00, O0000OOOOOOO0O0O0) = OOO0O0OO0OOO000OO.split(',')
            OOO0O0O000OO000O0 = {}
            OOO0O0O000OO000O0['usage'] = OOOO0OOOOOO00OOOO
            OOO0O0O000OO000O0['total'] = O0O00000O00OO0OO0
            OOO0O0O000OO000O0['iusage'] = O00000OO00OO0OO00
            OOO0O0O000OO000O0['itotal'] = O0000OOOOOOO0O0O0
            OOO00OOO0O0O00OOO[O00OO00000O000000] = OOO0O0O000OO000O0
        return {'apps': {'disk_total': O0000OOO0O00OOOOO, 'disk_usage': OO00O00OO0OO0OO0O, 'sites': O0OOO00O0O0000000, 'databases': OO0O00O00OOOOO000, 'ftps': OO00O0000OO00000O, 'plugins': O0OOOOOOO0000OOO0}, 'disks': OOO00OOO0O0O00OOO}

    def get_app_usage(O0O00O0OO0OO0OO00, O00O0O000OO0O0O0O):
        if False:
            i = 10
            return i + 15
        O000000OO0OO0O0O0 = time.localtime()
        O00OOOOO00000OOOO = O0O00O0OO0OO0OO00.get_time_key()
        O00OOO0000OO00O00 = time.localtime(time.mktime((O000000OO0OO0O0O0.tm_year, O000000OO0OO0O0O0.tm_mon, O000000OO0OO0O0O0.tm_mday - 1, 0, 0, 0, 0, 0, 0)))
        O0OO0O0OO0OO0O0O0 = O0O00O0OO0OO0OO00.get_time_key(O00OOO0000OO00O00)
        O0O000OO0O0OOOO00 = public.M('system').dbfile('system').table('app_usage').where('time_key =? or time_key=?', (O00OOOOO00000OOOO, O0OO0O0OO0OO0O0O0))
        OO00O00O000OO00OO = O0O000OO0O0OOOO00.select()
        if type(OO00O00O000OO00OO) == str or not OO00O00O000OO00OO:
            return {}
        OO00O000O00OO00OO = {}
        OO00O0O0OOO0O0O0O = {}
        for O0OOO00O0OOO0OOOO in OO00O00O000OO00OO:
            if O0OOO00O0OOO0OOOO['time_key'] == O00OOOOO00000OOOO:
                OO00O000O00OO00OO = O0O00O0OO0OO0OO00.parse_app_usage_info(O0OOO00O0OOO0OOOO)
            if O0OOO00O0OOO0OOOO['time_key'] == O0OO0O0OO0OO0O0O0:
                OO00O0O0OOO0O0O0O = O0O00O0OO0OO0OO00.parse_app_usage_info(O0OOO00O0OOO0OOOO)
        if not OO00O000O00OO00OO:
            return {}
        for (O000O0000OO0OOO00, OO0O0OOOO0O0O0000) in OO00O000O00OO00OO['disks'].items():
            O0000OO0O0OO0O00O = int(OO0O0OOOO0O0O0000['total'])
            OOOOOOOO000O00OOO = int(OO0O0OOOO0O0O0000['usage'])
            OOO00O0OOO00O0OO0 = int(OO0O0OOOO0O0O0000['itotal'])
            O0O0OOOOOO0000O00 = int(OO0O0OOOO0O0O0000['iusage'])
            if OO00O0O0OOO0O0O0O and O000O0000OO0OOO00 in OO00O0O0OOO0O0O0O['disks'].keys():
                OO0OO0O0OOO0000O0 = OO00O0O0OOO0O0O0O['disks']
                OO00OOO00OO00O0OO = OO0OO0O0OOO0000O0[O000O0000OO0OOO00]
                O0OOO00OO0O00OOO0 = int(OO00OOO00OO00O0OO['total'])
                if O0OOO00OO0O00OOO0 == O0000OO0O0OO0O00O:
                    OOO0O0OOOO00OOO0O = int(OO00OOO00OO00O0OO['usage'])
                    OO0000O0O00OOOOO0 = 0
                    OO00000OOOOO0OO00 = OOOOOOOO000O00OOO - OOO0O0OOOO00OOO0O
                    if OO00000OOOOO0OO00 > 0:
                        OO0000O0O00OOOOO0 = round(OO00000OOOOO0OO00 / O0000OO0O0OO0O00O, 2)
                    OO0O0OOOO0O0O0000['incr'] = OO0000O0O00OOOOO0
                OOO00000OOO000OO0 = int(OO00OOO00OO00O0OO['itotal'])
                if True:
                    O0O0O0OOOOOOO0O0O = int(OO00OOO00OO00O0OO['iusage'])
                    OOOO00O000000O000 = 0
                    OO00000OOOOO0OO00 = O0O0OOOOOO0000O00 - O0O0O0OOOOOOO0O0O
                    if OO00000OOOOO0OO00 > 0:
                        OOOO00O000000O000 = round(OO00000OOOOO0OO00 / OOO00O0OOO00O0OO0, 2)
                    OO0O0OOOO0O0O0000['iincr'] = OOOO00O000000O000
        OO0O000OO0OO00OO0 = OO00O000O00OO00OO['apps']
        O000OOOOOO0000O00 = int(OO0O000OO0OO00OO0['disk_total'])
        if OO00O0O0OOO0O0O0O and OO00O0O0OOO0O0O0O['apps']['disk_total'] == OO0O000OO0OO00OO0['disk_total']:
            OO00O0OOO0OO0O000 = OO00O0O0OOO0O0O0O['apps']
            for (OOO0OO0OOO00O000O, O0O0O0OO0O000OO00) in OO0O000OO0OO00OO0.items():
                if OOO0OO0OOO00O000O == 'disks':
                    continue
                if OOO0OO0OOO00O000O == 'disk_total':
                    continue
                if OOO0OO0OOO00O000O == 'disk_usage':
                    continue
                OO00O000O000OO0O0 = 0
                O000OO0O0000OO000 = int(O0O0O0OO0O000OO00) - int(OO00O0OOO0OO0O000[OOO0OO0OOO00O000O])
                if O000OO0O0000OO000 > 0:
                    OO00O000O000OO0O0 = round(O000OO0O0000OO000 / O000OOOOOO0000O00, 2)
                OO0O000OO0OO00OO0[OOO0OO0OOO00O000O] = {'val': O0O0O0OO0O000OO00, 'incr': OO00O000O000OO0O0}
        return OO00O000O00OO00OO

    def get_timestamp_interval(O0O0O00000OO0O000, O0O0O0000O0OO00OO):
        if False:
            for i in range(10):
                print('nop')
        OOOO000000O0O0OO0 = None
        O0OOOO0O0000000O0 = None
        OOOO000000O0O0OO0 = time.mktime((O0O0O0000O0OO00OO.tm_year, O0O0O0000O0OO00OO.tm_mon, O0O0O0000O0OO00OO.tm_mday, 0, 0, 0, 0, 0, 0))
        O0OOOO0O0000000O0 = time.mktime((O0O0O0000O0OO00OO.tm_year, O0O0O0000O0OO00OO.tm_mon, O0O0O0000O0OO00OO.tm_mday, 23, 59, 59, 0, 0, 0))
        return (OOOO000000O0O0OO0, O0OOOO0O0000000O0)

    def check_server(OOO0O0O00O00OO000):
        if False:
            for i in range(10):
                print('nop')
        try:
            O000O0OO0OOO0OO00 = ['php', 'nginx', 'apache', 'mysql', 'tomcat', 'pure-ftpd', 'redis', 'memcached']
            O0O0O0O000OOOOOO0 = panelPlugin()
            OO00O00OO000O00OO = public.dict_obj()
            O00OOO00OO00OO0OO = ''
            for OO0O0OOO0OO0O0OOO in O000O0OO0OOO0OO00:
                OOO0OO000OO00OOO0 = False
                OO0OOOO0O0OO0000O = False
                OO00O00OO000O00OO.name = OO0O0OOO0OO0O0OOO
                O0O000O00000OO00O = O0O0O0O000OOOOOO0.getPluginInfo(OO00O00OO000O00OO)
                if not O0O000O00000OO00O:
                    continue
                OOO00OOOO0O00OO00 = O0O000O00000OO00O['versions']
                for O000O0OOO0O00OOO0 in OOO00OOOO0O00OO00:
                    if O000O0OOO0O00OOO0['status']:
                        OO0OOOO0O0OO0000O = True
                    if 'run' in O000O0OOO0O00OOO0.keys() and O000O0OOO0O00OOO0['run']:
                        OO0OOOO0O0OO0000O = True
                        OOO0OO000OO00OOO0 = True
                        break
                OOO00OO0000000O0O = 0
                if OO0OOOO0O0OO0000O:
                    OOO00OO0000000O0O = 1
                    if not OOO0OO000OO00OOO0:
                        OOO00OO0000000O0O = 2
                O00OOO00OO00OO0OO += str(OOO00OO0000000O0O)
            if '2' in O00OOO00OO00OO0OO:
                public.M('system').dbfile('server_status').add('status, addtime', (O00OOO00OO00OO0OO, time.time()))
        except Exception as O00OO00OOOO0000O0:
            return True

    def get_daily_data(OOOOO00OO0O0OO0OO, OO0OOOO0O000OOO00):
        if False:
            for i in range(10):
                print('nop')
        ''
        O00O000O000OOOOOO = 'IS_PRO_OR_LTD_FOR_PANEL_DAILY'
        O00OOOOOOO0O0O0OO = cache.get(O00O000O000OOOOOO)
        if not O00OOOOOOO0O0O0OO:
            try:
                O0O0O0O00O00OOO00 = panelPlugin()
                O0OOOOOOO0OO0000O = O0O0O0O00O00OOO00.get_soft_list(OO0OOOO0O000OOO00)
                if O0OOOOOOO0OO0000O['pro'] < 0 and O0OOOOOOO0OO0000O['ltd'] < 0:
                    if os.path.exists('/www/server/panel/data/start_daily.pl'):
                        os.remove('/www/server/panel/data/start_daily.pl')
                    return {'status': False, 'msg': 'No authorization.', 'data': [], 'date': OO0OOOO0O000OOO00.date}
                cache.set(O00O000O000OOOOOO, True, 86400)
            except:
                return {'status': False, 'msg': '获取不到授权信息，请检查网络是否正常', 'data': [], 'date': OO0OOOO0O000OOO00.date}
        if not os.path.exists('/www/server/panel/data/start_daily.pl'):
            public.writeFile('/www/server/panel/data/start_daily.pl', OO0OOOO0O000OOO00.date)
        return OOOOO00OO0O0OO0OO.get_daily_data_local(OO0OOOO0O000OOO00.date)

    def get_daily_data_local(OOO00O00O000O0O00, OO0OOO0OO0000O000):
        if False:
            print('Hello World!')
        OO00OOO00OOO00OO0 = time.strptime(OO0OOO0OO0000O000, '%Y%m%d')
        O0O000OO0OOOOO00O = OOO00O00O000O0O00.get_time_key(OO00OOO00OOO00OO0)
        OOO00O00O000O0O00.check_databases()
        O0O0O00OO00OO0OOO = time.strftime('%Y-%m-%d', OO00OOO00OOO00OO0)
        O0OOO00O0O00OOOO0 = 0
        (OOOOOO000O00OO00O, O0O00O000O0O00OOO) = OOO00O00O000O0O00.get_timestamp_interval(OO00OOO00OOO00OO0)
        O0OO0OOO0O0OOOOO0 = public.M('system').dbfile('system')
        OO0OO00O00OOOOOO0 = O0OO0OOO0O0OOOOO0.table('process_high_percent')
        O00O0OOO00O0O0000 = OO0OO00O00OOOOOO0.where('addtime>=? and addtime<=?', (OOOOOO000O00OO00O, O0O00O000O0O00OOO)).order('addtime').select()
        O00O0O000O0OO0000 = []
        if len(O00O0OOO00O0O0000) > 0:
            for OOO0000OO0OOOOO0O in O00O0OOO00O0O0000:
                OOO000O000O00O00O = int(OOO0000OO0OOOOO0O['cpu_percent'])
                if OOO000O000O00O00O >= 80:
                    O00O0O000O0OO0000.append({'time': OOO0000OO0OOOOO0O['addtime'], 'name': OOO0000OO0OOOOO0O['name'], 'pid': OOO0000OO0OOOOO0O['pid'], 'percent': OOO000O000O00O00O})
        O0O0OOOO00O0O0O00 = len(O00O0O000O0OO0000)
        OO0000O00O0OOO00O = 0
        OOO0O0O0O0000OOO0 = ''
        if O0O0OOOO00O0O0O00 == 0:
            OO0000O00O0OOO00O = 20
        else:
            OOO0O0O0O0000OOO0 = 'CPU出现过载情况'
        O0OOOOO000O0OOOOO = {'ex': O0O0OOOO00O0O0O00, 'detail': O00O0O000O0OO0000}
        O000000OO0O00000O = []
        if len(O00O0OOO00O0O0000) > 0:
            for OOO0000OO0OOOOO0O in O00O0OOO00O0O0000:
                O0O000O0OOOOO00O0 = float(OOO0000OO0OOOOO0O['memory'])
                O0O0O0O0O0000O0OO = psutil.virtual_memory().total
                OO0000OOO00O0000O = round(100 * O0O000O0OOOOO00O0 / O0O0O0O0O0000O0OO, 2)
                if OO0000OOO00O0000O >= 80:
                    O000000OO0O00000O.append({'time': OOO0000OO0OOOOO0O['addtime'], 'name': OOO0000OO0OOOOO0O['name'], 'pid': OOO0000OO0OOOOO0O['pid'], 'percent': OO0000OOO00O0000O})
        O0000OOO000O0OO00 = len(O000000OO0O00000O)
        OOO00O0OOOOOO0OO0 = ''
        O0OO000O00O0O0000 = 0
        if O0000OOO000O0OO00 == 0:
            O0OO000O00O0O0000 = 20
        elif O0000OOO000O0OO00 > 1:
            OOO00O0OOOOOO0OO0 = '内存在多个时间点出现占用80%'
        else:
            OOO00O0OOOOOO0OO0 = '内存出现占用超过80%'
        O0OOO0OOOOOO00O00 = {'ex': O0000OOO000O0OO00, 'detail': O000000OO0O00000O}
        OOO0OOOOOO0OO0O00 = public.M('system').dbfile('system').table('app_usage').where('time_key=?', (O0O000OO0OOOOO00O,))
        O0000OO0OO0OOOO0O = OOO0OOOOOO0OO0O00.select()
        O000000O0OO00OO0O = {}
        if O0000OO0OO0OOOO0O and type(O0000OO0OO0OOOO0O) != str:
            O000000O0OO00OO0O = OOO00O00O000O0O00.parse_app_usage_info(O0000OO0OO0OOOO0O[0])
        OOO0O000OOO00O0O0 = []
        if O000000O0OO00OO0O:
            OOOO000O00O0OO0O0 = O000000O0OO00OO0O['disks']
            for (OOOOO000OOO000OO0, OOO0OO0O0OO0OOOOO) in OOOO000O00O0OO0O0.items():
                O000OO0OOOOO000O0 = int(OOO0OO0O0OO0OOOOO['usage'])
                O0O0O0O0O0000O0OO = int(OOO0OO0O0OO0OOOOO['total'])
                O00OO00OOO00000O0 = round(O000OO0OOOOO000O0 / O0O0O0O0O0000O0OO, 2)
                O000OOOOOO0O0O0OO = int(OOO0OO0O0OO0OOOOO['iusage'])
                O000000O0O0OO0000 = int(OOO0OO0O0OO0OOOOO['itotal'])
                if O000000O0O0OO0000 > 0:
                    OO0OOO0OOO0O00000 = round(O000OOOOOO0O0O0OO / O000000O0O0OO0000, 2)
                else:
                    OO0OOO0OOO0O00000 = 0
                if O00OO00OOO00000O0 >= 0.8:
                    OOO0O000OOO00O0O0.append({'name': OOOOO000OOO000OO0, 'percent': O00OO00OOO00000O0 * 100, 'ipercent': OO0OOO0OOO0O00000 * 100, 'usage': O000OO0OOOOO000O0, 'total': O0O0O0O0O0000O0OO, 'iusage': O000OOOOOO0O0O0OO, 'itotal': O000000O0O0OO0000})
        OOOO0OO0OO00OO0O0 = len(OOO0O000OOO00O0O0)
        OO0O00OOO0OOOO0O0 = ''
        OO0O0O000000O0O00 = 0
        if OOOO0OO0OO00OO0O0 == 0:
            OO0O0O000000O0O00 = 20
        else:
            OO0O00OOO0OOOO0O0 = '有磁盘空间占用已经超过80%'
        O00O000O000O00O00 = {'ex': OOOO0OO0OO00OO0O0, 'detail': OOO0O000OOO00O0O0}
        O00O0O0OO000O0OOO = public.M('system').dbfile('system').table('server_status').where('addtime>=? and addtime<=?', (OOOOOO000O00OO00O, O0O00O000O0O00OOO)).order('addtime desc').select()
        OO00OO000OOO000O0 = ['php', 'nginx', 'apache', 'mysql', 'tomcat', 'pure-ftpd', 'redis', 'memcached']
        OO00000000O0000O0 = {}
        O0O000OO000O000OO = 0
        O00O0O0OOOOOOOOO0 = ''
        for (OO0O0000OO0OO0OO0, OO00OO0O00O00OO0O) in enumerate(OO00OO000OOO000O0):
            if OO00OO0O00O00OO0O == 'pure-ftpd':
                OO00OO0O00O00OO0O = 'ftpd'
            O0000O0000000O0OO = 0
            O0O00000OO00OOOOO = []
            for OOO0O00000O000O0O in O00O0O0OO000O0OOO:
                _O0OOO0O000000OOOO = OOO0O00000O000O0O['status']
                if OO0O0000OO0OO0OO0 < len(_O0OOO0O000000OOOO):
                    if _O0OOO0O000000OOOO[OO0O0000OO0OO0OO0] == '2':
                        O0O00000OO00OOOOO.append({'time': OOO0O00000O000O0O['addtime'], 'desc': '退出'})
                        O0000O0000000O0OO += 1
                        O0O000OO000O000OO += 1
            OO00000000O0000O0[OO00OO0O00O00OO0O] = {'ex': O0000O0000000O0OO, 'detail': O0O00000OO00OOOOO}
        OOOOO0OO00O00O0O0 = 0
        if O0O000OO000O000OO == 0:
            OOOOO0OO00O00O0O0 = 20
        else:
            O00O0O0OOOOOOOOO0 = '系统级服务有出现异常退出情况'
        O0OO00O0O00OOO0OO = public.M('crontab').field('sName,sType').where('sType in (?, ?)', ('database', 'site')).select()
        O0000O0O000O0O0OO = set((O0OO0O0O0OO000O00['sName'] for O0OO0O0O0OO000O00 in O0OO00O0O00OOO0OO if O0OO0O0O0OO000O00['sType'] == 'database'))
        OO0O0OOO000O000O0 = 'ALL' in O0000O0O000O0O0OO
        OOO00O0OOOOOO0O00 = set((O0O0OOOOOOO0000O0['sName'] for O0O0OOOOOOO0000O0 in O0OO00O0O00OOO0OO if O0O0OOOOOOO0000O0['sType'] == 'site'))
        O0OO00O0O0OOO0000 = 'ALL' in OOO00O0OOOOOO0O00
        O0000OOO00OO00OOO = []
        OOOO00OOO0O000O0O = []
        if not OO0O0OOO000O000O0:
            O000OO00000000O00 = public.M('databases').field('name').select()
            for OO000O00O00O00OOO in O000OO00000000O00:
                O0000000OO0000000 = OO000O00O00O00OOO['name']
                if O0000000OO0000000 not in O0000O0O000O0O0OO:
                    O0000OOO00OO00OOO.append({'name': O0000000OO0000000})
        if not O0OO00O0O0OOO0000:
            O0OO000000O00O0O0 = public.M('sites').field('name').select()
            for O0O00OOO0OOOO000O in O0OO000000O00O0O0:
                OOO0O0OO00O000O00 = O0O00OOO0OOOO000O['name']
                if OOO0O0OO00O000O00 not in OOO00O0OOOOOO0O00:
                    OOOO00OOO0O000O0O.append({'name': OOO0O0OO00O000O00})
        O00OO0O00OOO0O0OO = public.M('system').dbfile('system').table('backup_status').where('addtime>=? and addtime<=?', (OOOOOO000O00OO00O, O0O00O000O0O00OOO)).select()
        OOO000O00OO0O000O = {'database': {'no_backup': O0000OOO00OO00OOO, 'backup': []}, 'site': {'no_backup': OOOO00OOO0O000O0O, 'backup': []}, 'path': {'no_backup': [], 'backup': []}}
        O00OOOO0O0OOO0OOO = 0
        for O000OOO00O0OOO00O in O00OO0O00OOO0O0OO:
            OO00OOO0OOO0O0OOO = O000OOO00O0OOO00O['status']
            if OO00OOO0OOO0O0OOO:
                continue
            O00OOOO0O0OOO0OOO += 1
            O0000OO0O00O00000 = O000OOO00O0OOO00O['id']
            O00OO0O00OOOOO000 = public.M('crontab').where('id=?', O0000OO0O00O00000).find()
            if not O00OO0O00OOOOO000:
                continue
            O000O000OO00OOO00 = O00OO0O00OOOOO000['sType']
            if not O000O000OO00OOO00:
                continue
            OO000O0OOOOOOO000 = O00OO0O00OOOOO000['name']
            OOOOOO0000OO000O0 = O000OOO00O0OOO00O['addtime']
            O0OO0O00O000OO00O = O000OOO00O0OOO00O['target']
            if O000O000OO00OOO00 not in OOO000O00OO0O000O.keys():
                OOO000O00OO0O000O[O000O000OO00OOO00] = {}
                OOO000O00OO0O000O[O000O000OO00OOO00]['backup'] = []
                OOO000O00OO0O000O[O000O000OO00OOO00]['no_backup'] = []
            OOO000O00OO0O000O[O000O000OO00OOO00]['backup'].append({'name': OO000O0OOOOOOO000, 'target': O0OO0O00O000OO00O, 'status': OO00OOO0OOO0O0OOO, 'target': O0OO0O00O000OO00O, 'time': OOOOOO0000OO000O0})
        O0OO0OO00O000O0O0 = ''
        O0O000O0OOO0OOO0O = 0
        if O00OOOO0O0OOO0OOO == 0:
            O0O000O0OOO0OOO0O = 20
        else:
            O0OO0OO00O000O0O0 = '有计划任务备份失败'
        if len(O0000OOO00OO00OOO) == 0:
            O0O000O0OOO0OOO0O += 10
        else:
            if O0OO0OO00O000O0O0:
                O0OO0OO00O000O0O0 += ';'
            O0OO0OO00O000O0O0 += '有数据库未及时备份'
        if len(OOOO00OOO0O000O0O) == 0:
            O0O000O0OOO0OOO0O += 10
        else:
            if O0OO0OO00O000O0O0:
                O0OO0OO00O000O0O0 += ';'
            O0OO0OO00O000O0O0 += '有网站未备份'
        OO00OOOOOOO0O0OO0 = 0
        O0O000O0OOOOOO0O0 = public.M('logs').where('addtime like "{}%" and type=?'.format(O0O0O00OO00OO0OOO), ('用户登录',)).select()
        OOO0000O0OO00O00O = []
        if O0O000O0OOOOOO0O0 and type(O0O000O0OOOOOO0O0) == list:
            for OOOO00O0OO0O00OOO in O0O000O0OOOOOO0O0:
                O0O00O00OO00O0O00 = OOOO00O0OO0O00OOO['log']
                if O0O00O00OO00O0O00.find('失败') >= 0 or O0O00O00OO00O0O00.find('错误') >= 0:
                    OO00OOOOOOO0O0OO0 += 1
                    OOO0000O0OO00O00O.append({'time': time.mktime(time.strptime(OOOO00O0OO0O00OOO['addtime'], '%Y-%m-%d %H:%M:%S')), 'desc': OOOO00O0OO0O00OOO['log'], 'username': OOOO00O0OO0O00OOO['username']})
            OOO0000O0OO00O00O.sort(key=lambda O000O00OOOOOOOO0O: O000O00OOOOOOOO0O['time'])
        O0O00OO0O0OO0OO0O = public.M('logs').where('type=?', ('SSH安全',)).where("addtime like '{}%'".format(O0O0O00OO00OO0OOO), ()).select()
        O000O000OOOOO000O = []
        OO00O0000OOOO0OOO = 0
        if O0O00OO0O0OO0OO0O:
            for OOOO00O0OO0O00OOO in O0O00OO0O0OO0OO0O:
                O0O00O00OO00O0O00 = OOOO00O0OO0O00OOO['log']
                if O0O00O00OO00O0O00.find('存在异常') >= 0:
                    OO00O0000OOOO0OOO += 1
                    O000O000OOOOO000O.append({'time': time.mktime(time.strptime(OOOO00O0OO0O00OOO['addtime'], '%Y-%m-%d %H:%M:%S')), 'desc': OOOO00O0OO0O00OOO['log'], 'username': OOOO00O0OO0O00OOO['username']})
            O000O000OOOOO000O.sort(key=lambda O000OOOO0000O00O0: O000OOOO0000O00O0['time'])
        O0O0OO0OO000O000O = ''
        O0000O0O0OOO0O0O0 = 0
        if OO00O0000OOOO0OOO == 0:
            O0000O0O0OOO0O0O0 = 10
        else:
            O0O0OO0OO000O000O = 'SSH有异常登录'
        if OO00OOOOOOO0O0OO0 == 0:
            O0000O0O0OOO0O0O0 += 10
        else:
            if OO00OOOOOOO0O0OO0 > 10:
                O0000O0O0OOO0O0O0 -= 10
            if O0O0OO0OO000O000O:
                O0O0OO0OO000O000O += ';'
            O0O0OO0OO000O000O += '面板登录有错误'.format(OO00OOOOOOO0O0OO0)
        O00O0O0OO000O0OOO = {'panel': {'ex': OO00OOOOOOO0O0OO0, 'detail': OOO0000O0OO00O00O}, 'ssh': {'ex': OO00O0000OOOO0OOO, 'detail': O000O000OOOOO000O}}
        O0OOO00O0O00OOOO0 = OO0000O00O0OOO00O + O0OO000O00O0O0000 + OO0O0O000000O0O00 + OOOOO0OO00O00O0O0 + O0O000O0OOO0OOO0O + O0000O0O0OOO0O0O0
        OOO00OO00000O0OO0 = [OOO0O0O0O0000OOO0, OOO00O0OOOOOO0OO0, OO0O00OOO0OOOO0O0, O00O0O0OOOOOOOOO0, O0OO0OO00O000O0O0, O0O0OO0OO000O000O]
        O00000OOO00O0O000 = []
        for OO0OO0O0OOOO0000O in OOO00OO00000O0OO0:
            if OO0OO0O0OOOO0000O:
                if OO0OO0O0OOOO0000O.find(';') >= 0:
                    for O000O0OO00000OOO0 in OO0OO0O0OOOO0000O.split(';'):
                        O00000OOO00O0O000.append(O000O0OO00000OOO0)
                else:
                    O00000OOO00O0O000.append(OO0OO0O0OOOO0000O)
        if not O00000OOO00O0O000:
            O00000OOO00O0O000.append('服务器运行正常，请继续保持！')
        OO00O0OOOOO00000O = OOO00O00O000O0O00.evaluate(O0OOO00O0O00OOOO0)
        return {'data': {'cpu': O0OOOOO000O0OOOOO, 'ram': O0OOO0OOOOOO00O00, 'disk': O00O000O000O00O00, 'server': OO00000000O0000O0, 'backup': OOO000O00OO0O000O, 'exception': O00O0O0OO000O0OOO}, 'evaluate': OO00O0OOOOO00000O, 'score': O0OOO00O0O00OOOO0, 'date': O0O000OO0OOOOO00O, 'summary': O00000OOO00O0O000, 'status': True}

    def evaluate(O0O0000O0000OO000, O0O0OOO00OO00O0OO):
        if False:
            i = 10
            return i + 15
        O0OO0000OOO0000O0 = ''
        if O0O0OOO00OO00O0OO >= 100:
            O0OO0000OOO0000O0 = '正常'
        elif O0O0OOO00OO00O0OO >= 80:
            O0OO0000OOO0000O0 = '良好'
        else:
            O0OO0000OOO0000O0 = '一般'
        return O0OO0000OOO0000O0

    def get_daily_list(O00O00OO00OOO0OOO, O0000O0O0O000O0O0):
        if False:
            while True:
                i = 10
        OO0O0OO0000OO0O00 = public.M('system').dbfile('system').table('daily').where('time_key>?', 0).select()
        O000O0OOOO0O0000O = []
        for O0OO000000OOO0OOO in OO0O0OO0000OO0O00:
            O0OO000000OOO0OOO['evaluate'] = O00O00OO00OOO0OOO.evaluate(O0OO000000OOO0OOO['evaluate'])
            O000O0OOOO0O0000O.append(O0OO000000OOO0OOO)
        return O000O0OOOO0O0000O