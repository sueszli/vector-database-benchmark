import os, public, psutil, json, time, re
from projectModel.base import projectBase

class main(projectBase):
    __O00O0OOOO000OO0O0 = '{}/config/quota.json'.format(public.get_panel_path())
    __O00O0OOOO00O00OO0 = '{}/config/mysql_quota.json'.format(public.get_panel_path())
    __OOO00O000OO00O0OO = public.to_string([27492, 21151, 33021, 20026, 20225, 19994, 29256, 19987, 20139, 21151, 33021, 65292, 35831, 20808, 36141, 20080, 20225, 19994, 29256])

    def __init__(OOOOOO00O0OOOOO00):
        if False:
            i = 10
            return i + 15
        _OO0OOO0OOO00000OO = '{}/data/quota_install.pl'.format(public.get_panel_path())
        if not os.path.exists(_OO0OOO0OOO00000OO):
            O00000OO0000O0O0O = '/usr/sbin/xfs_quota'
            if not os.path.exists(O00000OO0000O0O0O):
                if os.path.exists('/usr/bin/apt-get'):
                    public.ExecShell('nohup apt-get install xfsprogs -y > /dev/null &')
                else:
                    public.ExecShell('nohup yum install xfsprogs -y > /dev/null &')
            public.writeFile(_OO0OOO0OOO00000OO, 'True')

    def __O00000O0O0O0000O0(O0OOO0O0OO0OO0O00, args=None):
        if False:
            i = 10
            return i + 15
        ''
        O0O0OOOOO0OO00000 = []
        for O0O0OOOO0OO00O000 in psutil.disk_partitions():
            if O0O0OOOO0OO00O000.fstype == 'xfs':
                O0O0OOOOO0OO00000.append((O0O0OOOO0OO00O000.mountpoint, O0O0OOOO0OO00O000.device, psutil.disk_usage(O0O0OOOO0OO00O000.mountpoint).free, O0O0OOOO0OO00O000.opts.split(',')))
        return O0O0OOOOO0OO00000

    def __O000O0O0000000O00(O0O000O0O0000OOO0, args=None):
        if False:
            for i in range(10):
                print('nop')
        ''
        return O0O000O0O0000OOO0.__OO0000OO00O00O0OO(args.path)

    def __O0000O00O0OO00OOO(OOO0O0O00OOOO0O0O, OOOOOOO00O000OOOO):
        if False:
            while True:
                i = 10
        ''
        O0O0OOOO0O0OO0O00 = OOO0O0O00OOOO0O0O.__O00000O0O0O0000O0()
        for OOOOO000O0O00OOO0 in O0O0OOOO0O0OO0O00:
            if OOOOOOO00O000OOOO.find(OOOOO000O0O00OOO0[0] + '/') == 0:
                if not 'prjquota' in OOOOO000O0O00OOO0[3]:
                    return OOOOO000O0O00OOO0
                return OOOOO000O0O00OOO0[1]
        return ''

    def __OO0000OO00O00O0OO(OOO00O00OO00OO00O, OOOO0OO0O00O00OOO):
        if False:
            while True:
                i = 10
        ''
        if not os.path.exists(OOOO0OO0O00O00OOO):
            return -1
        if not os.path.isdir(OOOO0OO0O00O00OOO):
            return -2
        OOOO0OOO0O0OO000O = OOO00O00OO00OO00O.__O00000O0O0O0000O0()
        for OO0OOO0O00O0O0OO0 in OOOO0OOO0O0OO000O:
            if OOOO0OO0O00O00OOO.find(OO0OOO0O00O0O0OO0[0] + '/') == 0:
                return OO0OOO0O00O0O0OO0[2] / 1024 / 1024
        return -3

    def get_quota_path_list(O0O00O0O000000O00, args=None, get_path=None):
        if False:
            for i in range(10):
                print('nop')
        ''
        if not os.path.exists(O0O00O0O000000O00.__O00O0OOOO000OO0O0):
            public.writeFile(O0O00O0O000000O00.__O00O0OOOO000OO0O0, '[]')
        OOOOO0O00O0O0OOO0 = json.loads(public.readFile(O0O00O0O000000O00.__O00O0OOOO000OO0O0))
        OOO000O0O0O00OOO0 = []
        for OO0O0O0O000OO0OOO in OOOOO0O00O0O0OOO0:
            if not os.path.exists(OO0O0O0O000OO0OOO['path']) or not os.path.isdir(OO0O0O0O000OO0OOO['path']) or os.path.islink(OO0O0O0O000OO0OOO['path']):
                continue
            if get_path:
                if OO0O0O0O000OO0OOO['path'] == get_path:
                    O0OOO0OO00O0O0O00 = psutil.disk_usage(OO0O0O0O000OO0OOO['path'])
                    OO0O0O0O000OO0OOO['used'] = O0OOO0OO00O0O0O00.used
                    OO0O0O0O000OO0OOO['free'] = O0OOO0OO00O0O0O00.free
                    return OO0O0O0O000OO0OOO
                else:
                    continue
            O0OOO0OO00O0O0O00 = psutil.disk_usage(OO0O0O0O000OO0OOO['path'])
            OO0O0O0O000OO0OOO['used'] = O0OOO0OO00O0O0O00.used
            OO0O0O0O000OO0OOO['free'] = O0OOO0OO00O0O0O00.free
            OOO000O0O0O00OOO0.append(OO0O0O0O000OO0OOO)
        if get_path:
            return {'size': 0, 'used': 0, 'free': 0}
        if len(OOO000O0O0O00OOO0) != len(OOOOO0O00O0O0OOO0):
            public.writeFile(O0O00O0O000000O00.__O00O0OOOO000OO0O0, json.dumps(OOO000O0O0O00OOO0))
        return OOOOO0O00O0O0OOO0

    def get_quota_mysql_list(OOO000OO00OO00OO0, args=None, get_name=None):
        if False:
            while True:
                i = 10
        ''
        if not os.path.exists(OOO000OO00OO00OO0.__O00O0OOOO00O00OO0):
            public.writeFile(OOO000OO00OO00OO0.__O00O0OOOO00O00OO0, '[]')
        OO0000OO0O0OOOO00 = json.loads(public.readFile(OOO000OO00OO00OO0.__O00O0OOOO00O00OO0))
        O00O00O00O00O00OO = []
        OOO00O0O00O0OOO00 = public.M('databases')
        for OO000OOOOO0000OO0 in OO0000OO0O0OOOO00:
            if get_name:
                if OO000OOOOO0000OO0['db_name'] == get_name:
                    OO000OOOOO0000OO0['used'] = OO000OOOOO0000OO0['used'] = int(public.get_database_size_by_name(OO000OOOOO0000OO0['db_name']))
                    _O000OO0000O000OO0 = OO000OOOOO0000OO0['size'] * 1024 * 1024
                    if OO000OOOOO0000OO0['used'] > _O000OO0000O000OO0 and OO000OOOOO0000OO0['insert_accept'] or (OO000OOOOO0000OO0['used'] < _O000OO0000O000OO0 and (not OO000OOOOO0000OO0['insert_accept'])):
                        OOO000OO00OO00OO0.mysql_quota_check()
                    return OO000OOOOO0000OO0
            elif OOO00O0O00O0OOO00.where('name=?', OO000OOOOO0000OO0['db_name']).count():
                if args:
                    OO000OOOOO0000OO0['used'] = int(public.get_database_size_by_name(OO000OOOOO0000OO0['db_name']))
                O00O00O00O00O00OO.append(OO000OOOOO0000OO0)
        OOO00O0O00O0OOO00.close()
        if get_name:
            return {'size': 0, 'used': 0}
        if len(O00O00O00O00O00OO) != len(OO0000OO0O0OOOO00):
            public.writeFile(OOO000OO00OO00OO0.__O00O0OOOO00O00OO0, json.dumps(O00O00O00O00O00OO))
        return O00O00O00O00O00OO

    def __O00000O00OO00OOO0(OOOOOO0O0000OO000, OOO00OOO000O000OO, O000OO0O00OO00OOO, O0000OO000O0OO00O, OOO000OO00OOOO00O):
        if False:
            return 10
        ''
        O00O0OO00OO000O00 = OOO00OOO000O000OO.execute("REVOKE ALL PRIVILEGES ON `{}`.* FROM '{}'@'{}';".format(O0000OO000O0OO00O, O000OO0O00OO00OOO, OOO000OO00OOOO00O))
        if O00O0OO00OO000O00:
            raise public.PanelError('移除数据库用户的插入权限失败: {}'.format(O00O0OO00OO000O00))
        O00O0OO00OO000O00 = OOO00OOO000O000OO.execute("GRANT SELECT, DELETE, CREATE, DROP, REFERENCES, INDEX, CREATE TEMPORARY TABLES, LOCK TABLES, CREATE VIEW, EVENT, TRIGGER, SHOW VIEW, CREATE ROUTINE, ALTER ROUTINE, EXECUTE ON `{}`.* TO '{}'@'{}';".format(O0000OO000O0OO00O, O000OO0O00OO00OOO, OOO000OO00OOOO00O))
        if O00O0OO00OO000O00:
            raise public.PanelError('移除数据库用户的插入权限失败: {}'.format(O00O0OO00OO000O00))
        OOO00OOO000O000OO.execute('FLUSH PRIVILEGES;')
        return True

    def __O00OOOOOOOOOO0000(O000O0O00O0OOOOO0, OOO000OOOO00O000O, O0OOO0OO00O00OOO0, OOO00000O0O0OO0O0, O0O0OOOOO0OOOO000):
        if False:
            i = 10
            return i + 15
        ''
        O0000O00O000O0OOO = OOO000OOOO00O000O.execute("REVOKE ALL PRIVILEGES ON `{}`.* FROM '{}'@'{}';".format(OOO00000O0O0OO0O0, O0OOO0OO00O00OOO0, O0O0OOOOO0OOOO000))
        if O0000O00O000O0OOO:
            raise public.PanelError('恢复数据库用户的插入权限失败: {}'.format(O0000O00O000O0OOO))
        O0000O00O000O0OOO = OOO000OOOO00O000O.execute("GRANT ALL PRIVILEGES ON `{}`.* TO '{}'@'{}';".format(OOO00000O0O0OO0O0, O0OOO0OO00O00OOO0, O0O0OOOOO0OOOO000))
        if O0000O00O000O0OOO:
            raise public.PanelError('恢复数据库用户的插入权限失败: {}'.format(O0000O00O000O0OOO))
        OOO000OOOO00O000O.execute('FLUSH PRIVILEGES;')
        return True

    def mysql_quota_service(O0000OOO00O000000):
        if False:
            return 10
        ''
        while 1:
            time.sleep(600)
            O0000OOO00O000000.mysql_quota_check()

    def __OOO0O000OO0O0OOO0(O0OOOO000O0OOO0OO, O0OO0O00OOO0O000O):
        if False:
            while True:
                i = 10
        try:
            if type(O0OO0O00OOO0O000O) != list and type(O0OO0O00OOO0O000O) != str:
                O0OO0O00OOO0O000O = list(O0OO0O00OOO0O000O)
            return O0OO0O00OOO0O000O
        except:
            return []

    def mysql_quota_check(O00000O00O0O00O00):
        if False:
            i = 10
            return i + 15
        ''
        if not O00000O00O0O00O00.__O0OO00O00OOOO0O00():
            return public.returnMsg(False, O00000O00O0O00O00.__OOO00O000OO00O0OO)
        O0000OO0O0O0OOOO0 = O00000O00O0O00O00.get_quota_mysql_list()
        for O0OOO000O0OOO00OO in O0000OO0O0O0OOOO0:
            try:
                if O0OOO000O0OOO00OO['size'] < 1:
                    if not O0OOO000O0OOO00OO['insert_accept']:
                        O00000O00O0O00O00.__O00OOOOOOOOOO0000(OO0000O0000000O0O, OOOOOOOO00O00O00O, O0OOO000O0OOO00OO['db_name'], OOO00000O000OO0OO[0])
                        O0OOO000O0OOO00OO['insert_accept'] = True
                        public.WriteLog('磁盘配额', '已关闭数据库[{}]配额,恢复插入权限'.format(O0OOO000O0OOO00OO['db_name']))
                        continue
                OO00O0OO00000OO00 = public.get_database_size_by_name(O0OOO000O0OOO00OO['db_name']) / 1024 / 1024
                OOOOOOOO00O00O00O = public.M('databases').where('name=?', (O0OOO000O0OOO00OO['db_name'],)).getField('username')
                OO0000O0000000O0O = public.get_mysql_obj(O0OOO000O0OOO00OO['db_name'])
                OO00OOO00OOOOO0O0 = O00000O00O0O00O00.__OOO0O000OO0O0OOO0(OO0000O0000000O0O.query("select Host from mysql.user where User='" + OOOOOOOO00O00O00O + "'"))
                if OO00O0OO00000OO00 < O0OOO000O0OOO00OO['size']:
                    if not O0OOO000O0OOO00OO['insert_accept']:
                        for OOO00000O000OO0OO in OO00OOO00OOOOO0O0:
                            O00000O00O0O00O00.__O00OOOOOOOOOO0000(OO0000O0000000O0O, OOOOOOOO00O00O00O, O0OOO000O0OOO00OO['db_name'], OOO00000O000OO0OO[0])
                        O0OOO000O0OOO00OO['insert_accept'] = True
                        public.WriteLog('磁盘配额', '数据库[{}]因低于配额[{}MB],恢复插入权限'.format(O0OOO000O0OOO00OO['db_name'], O0OOO000O0OOO00OO['size']))
                    if hasattr(OO0000O0000000O0O, 'close'):
                        OO0000O0000000O0O.close()
                    continue
                if O0OOO000O0OOO00OO['insert_accept']:
                    for OOO00000O000OO0OO in OO00OOO00OOOOO0O0:
                        O00000O00O0O00O00.__O00000O00OO00OOO0(OO0000O0000000O0O, OOOOOOOO00O00O00O, O0OOO000O0OOO00OO['db_name'], OOO00000O000OO0OO[0])
                    O0OOO000O0OOO00OO['insert_accept'] = False
                    public.WriteLog('磁盘配额', '数据库[{}]因超出配额[{}MB],移除插入权限'.format(O0OOO000O0OOO00OO['db_name'], O0OOO000O0OOO00OO['size']))
                if hasattr(OO0000O0000000O0O, 'close'):
                    OO0000O0000000O0O.close()
            except:
                public.print_log(public.get_error_info())
        public.writeFile(O00000O00O0O00O00.__O00O0OOOO00O00OO0, json.dumps(O0000OO0O0O0OOOO0))

    def __O0OOOO0OO000OO0OO(O000OOO0OOO0OO0O0, OO0OO0OOOOOO0OOOO):
        if False:
            for i in range(10):
                print('nop')
        ''
        if not O000OOO0OOO0OO0O0.__O0OO00O00OOOO0O00():
            return public.returnMsg(False, O000OOO0OOO0OO0O0.__OOO00O000OO00O0OO)
        if not os.path.exists(O000OOO0OOO0OO0O0.__O00O0OOOO00O00OO0):
            public.writeFile(O000OOO0OOO0OO0O0.__O00O0OOOO00O00OO0, '[]')
        O0OO00O0OO0O0OO00 = int(OO0OO0OOOOOO0OOOO['size'])
        OOO0O00OOO00O0O00 = OO0OO0OOOOOO0OOOO.db_name.strip()
        O0O0O0000OOOOO00O = json.loads(public.readFile(O000OOO0OOO0OO0O0.__O00O0OOOO00O00OO0))
        for O0000O0OO0000O00O in O0O0O0000OOOOO00O:
            if O0000O0OO0000O00O['db_name'] == OOO0O00OOO00O0O00:
                return public.returnMsg(False, '数据库配额已存在')
        O0O0O0000OOOOO00O.append({'db_name': OOO0O00OOO00O0O00, 'size': O0OO00O0OO0O0OO00, 'insert_accept': True})
        public.writeFile(O000OOO0OOO0OO0O0.__O00O0OOOO00O00OO0, json.dumps(O0O0O0000OOOOO00O))
        public.WriteLog('磁盘配额', '创建数据库[{db_name}]的配额限制为: {size}MB'.format(db_name=OOO0O00OOO00O0O00, size=O0OO00O0OO0O0OO00))
        O000OOO0OOO0OO0O0.mysql_quota_check()
        return public.returnMsg(True, '添加成功')

    def __O0OO00O00OOOO0O00(OO0000OO0OOO0OO0O):
        if False:
            return 10
        from pluginAuth import Plugin
        OOO0OO0O000O0O00O = Plugin(False)
        O000OOO0O0OOOO0OO = OOO0OO0O000O0O00O.get_plugin_list()
        return int(O000OOO0O0OOOO0OO['ltd']) > time.time()

    def modify_mysql_quota(OO000OO0O0000O00O, OOO0OOOOO00OOOOOO):
        if False:
            i = 10
            return i + 15
        ''
        if not OO000OO0O0000O00O.__O0OO00O00OOOO0O00():
            return public.returnMsg(False, OO000OO0O0000O00O.__OOO00O000OO00O0OO)
        if not os.path.exists(OO000OO0O0000O00O.__O00O0OOOO00O00OO0):
            public.writeFile(OO000OO0O0000O00O.__O00O0OOOO00O00OO0, '[]')
        if not re.match('^\\d+$', OOO0OOOOO00OOOOOO.size):
            return public.returnMsg(False, '配额大小必须是整数!')
        O0O0OO0OOO0O0OOO0 = int(OOO0OOOOO00OOOOOO['size'])
        O0O0OOO0O00O00O0O = OOO0OOOOO00OOOOOO.db_name.strip()
        OOO0O0OOOO0O000OO = json.loads(public.readFile(OO000OO0O0000O00O.__O00O0OOOO00O00OO0))
        OO0O00O00O000O0OO = False
        for O00O0000000000000 in OOO0O0OOOO0O000OO:
            if O00O0000000000000['db_name'] == O0O0OOO0O00O00O0O:
                O00O0000000000000['size'] = O0O0OO0OOO0O0OOO0
                OO0O00O00O000O0OO = True
                break
        if OO0O00O00O000O0OO:
            public.writeFile(OO000OO0O0000O00O.__O00O0OOOO00O00OO0, json.dumps(OOO0O0OOOO0O000OO))
            public.WriteLog('磁盘配额', '修改数据库[{db_name}]的配额限制为: {size}MB'.format(db_name=O0O0OOO0O00O00O0O, size=O0O0OO0OOO0O0OOO0))
            OO000OO0O0000O00O.mysql_quota_check()
            return public.returnMsg(True, '修改成功')
        return OO000OO0O0000O00O.__O0OOOO0OO000OO0OO(OOO0OOOOO00OOOOOO)

    def __OOOO00OOO0O00OO00(O000O0000OO00O000, O0O00O00O0OO0O00O):
        if False:
            for i in range(10):
                print('nop')
        ''
        O00OOOO00OO0O0000 = []
        OO000O0000O0OOOOO = public.ExecShell("xfs_quota -x -c report {mountpoint}|awk '{{print $1}}'|grep '#'".format(mountpoint=O0O00O00O0OO0O00O))[0]
        if not OO000O0000O0OOOOO:
            return O00OOOO00OO0O0000
        for O0O0O0O0OO0O000O0 in OO000O0000O0OOOOO.split('\n'):
            if O0O0O0O0OO0O000O0:
                O00OOOO00OO0O0000.append(int(O0O0O0O0OO0O000O0.split('#')[-1]))
        return O00OOOO00OO0O0000

    def __O0O00OOOOO0O0OO0O(O00OOO00OOOO00000, O000OO00OOOOO0000, O0O00OO0000O000OO):
        if False:
            print('Hello World!')
        ''
        O00O00O00O000OOOO = 1001
        if not O000OO00OOOOO0000:
            return O00O00O00O000OOOO
        O00O00O00O000OOOO = O000OO00OOOOO0000[-1]['id'] + 1
        OO0OO0O00OOOOOOOO = sorted(O00OOO00OOOO00000.__OOOO00OOO0O00OO00(O0O00OO0000O000OO))
        if OO0OO0O00OOOOOOOO:
            if OO0OO0O00OOOOOOOO[-1] > O00O00O00O000OOOO:
                O00O00O00O000OOOO = OO0OO0O00OOOOOOOO[-1] + 1
        return O00O00O00O000OOOO

    def __O0O0OOO000OOO0000(OOO0OO00OO0OO00OO, OO0OOO0OO00000O00):
        if False:
            return 10
        ''
        if not OOO0OO00OO0OO00OO.__O0OO00O00OOOO0O00():
            return public.returnMsg(False, OOO0OO00OO0OO00OO.__OOO00O000OO00O0OO)
        O00OO00OOO00OO00O = OO0OOO0OO00000O00.path.strip()
        O0O0O00OO0000O000 = int(OO0OOO0OO00000O00.size)
        if not os.path.exists(O00OO00OOO00OO00O):
            return public.returnMsg(False, '指定目录不存在')
        if os.path.isfile(O00OO00OOO00OO00O):
            return public.returnMsg(False, '指定目录不是目录!')
        if os.path.islink(O00OO00OOO00OO00O):
            return public.returnMsg(False, '指定目录是软链接!')
        O0O0O0O0OO0OO0OOO = OOO0OO00OO0OO00OO.get_quota_path_list()
        for OOO0O0OOOOOO0O0OO in O0O0O0O0OO0OO0OOO:
            if OOO0O0OOOOOO0O0OO['path'] == O00OO00OOO00OO00O:
                return public.returnMsg(False, '指定目录已经设置过配额!')
        OO00OO00000OO0O00 = OOO0OO00OO0OO00OO.__OO0000OO00O00O0OO(O00OO00OOO00OO00O)
        if OO00OO00000OO0O00 == -3:
            return public.returnMsg(False, '指定目录所在分区不是XFS分区,不支持目录配额!')
        if OO00OO00000OO0O00 == -2:
            return public.returnMsg(False, '这不是一个有效的目录!')
        if OO00OO00000OO0O00 == -1:
            return public.returnMsg(False, '指定目录不存在!')
        if O0O0O00OO0000O000 > OO00OO00000OO0O00:
            return public.returnMsg(False, '指定磁盘可用的配额容量不足!')
        OO000000O0000OOO0 = OOO0OO00OO0OO00OO.__O0000O00O0OO00OOO(O00OO00OOO00OO00O)
        if not OO000000O0000OOO0:
            return public.returnMsg(False, '指定目录不在xfs磁盘分区中!')
        if isinstance(OO000000O0000OOO0, tuple):
            return public.returnMsg(False, '指定xfs分区未开启目录配额功能,请在挂载该分区时增加prjquota参数<p>/etc/fstab文件配置示例：<pre>{mountpoint}       {path}           xfs             defaults,prjquota       0 0</pre></p><p>注意：配置好后需重新挂载分区或重启服务器才能生效</p>'.format(mountpoint=OO000000O0000OOO0[1], path=OO000000O0000OOO0[0]))
        O00O0OO0O000O0OO0 = OOO0OO00OO0OO00OO.__O0O00OOOOO0O0OO0O(O0O0O0O0OO0OO0OOO, OO000000O0000OOO0)
        OOO0000OO0000O000 = public.ExecShell("xfs_quota -x -c 'project -s -p {path} {quota_id}'".format(path=O00OO00OOO00OO00O, quota_id=O00O0OO0O000O0OO0))
        if OOO0000OO0000O000[1]:
            return public.returnMsg(False, OOO0000OO0000O000[1])
        OOO0000OO0000O000 = public.ExecShell("xfs_quota -x -c 'limit -p bhard={size}m {quota_id}' {mountpoint}".format(quota_id=O00O0OO0O000O0OO0, size=O0O0O00OO0000O000, mountpoint=OO000000O0000OOO0))
        if OOO0000OO0000O000[1]:
            return public.returnMsg(False, OOO0000OO0000O000[1])
        O0O0O0O0OO0OO0OOO.append({'path': OO0OOO0OO00000O00.path, 'size': O0O0O00OO0000O000, 'id': O00O0OO0O000O0OO0})
        public.writeFile(OOO0OO00OO0OO00OO.__O00O0OOOO000OO0O0, json.dumps(O0O0O0O0OO0OO0OOO))
        public.WriteLog('磁盘配额', '创建目录[{path}]的配额限制为: {size}MB'.format(path=O00OO00OOO00OO00O, size=O0O0O00OO0000O000))
        return public.returnMsg(True, '添加成功')

    def modify_path_quota(OOOO000O0OOO000OO, O0O0OOOOO0O0O0000):
        if False:
            i = 10
            return i + 15
        ''
        if not OOOO000O0OOO000OO.__O0OO00O00OOOO0O00():
            return public.returnMsg(False, OOOO000O0OOO000OO.__OOO00O000OO00O0OO)
        O0000000O0OOOOO0O = O0O0OOOOO0O0O0000.path.strip()
        if not re.match('^\\d+$', O0O0OOOOO0O0O0000.size):
            return public.returnMsg(False, '配额大小必须是整数!')
        O0OOO00OOO0OO0000 = int(O0O0OOOOO0O0O0000.size)
        if not os.path.exists(O0000000O0OOOOO0O):
            return public.returnMsg(False, '指定目录不存在')
        if os.path.isfile(O0000000O0OOOOO0O):
            return public.returnMsg(False, '指定目录不是目录!')
        if os.path.islink(O0000000O0OOOOO0O):
            return public.returnMsg(False, '指定目录是软链接!')
        O0OOO000OO0OO0O0O = OOOO000O0OOO000OO.get_quota_path_list()
        O0OO00OOO00O0O00O = 0
        for O00OOO00OOO0000OO in O0OOO000OO0OO0O0O:
            if O00OOO00OOO0000OO['path'] == O0000000O0OOOOO0O:
                O0OO00OOO00O0O00O = O00OOO00OOO0000OO['id']
                break
        if not O0OO00OOO00O0O00O:
            return OOOO000O0OOO000OO.__O0O0OOO000OOO0000(O0O0OOOOO0O0O0000)
        O0O0O0OOOO000O00O = OOOO000O0OOO000OO.__OO0000OO00O00O0OO(O0000000O0OOOOO0O)
        if O0O0O0OOOO000O00O == -3:
            return public.returnMsg(False, '指定目录所在分区不是XFS分区,不支持目录配额!')
        if O0O0O0OOOO000O00O == -2:
            return public.returnMsg(False, '这不是一个有效的目录!')
        if O0O0O0OOOO000O00O == -1:
            return public.returnMsg(False, '指定目录不存在!')
        if O0OOO00OOO0OO0000 > O0O0O0OOOO000O00O:
            return public.returnMsg(False, '指定磁盘可用的配额容量不足!')
        O0000OOOO00O00O0O = OOOO000O0OOO000OO.__O0000O00O0OO00OOO(O0000000O0OOOOO0O)
        if not O0000OOOO00O00O0O:
            return public.returnMsg(False, '指定目录不在xfs磁盘分区中!')
        if isinstance(O0000OOOO00O00O0O, tuple):
            return public.returnMsg(False, '指定xfs分区未开启目录配额功能,请在挂载该分区时增加prjquota参数<p>/etc/fstab文件配置示例：<pre>{mountpoint}       {path}           xfs             defaults,prjquota       0 0</pre></p><p>注意：配置好后需重新挂载分区或重启服务器才能生效</p>'.format(mountpoint=O0000OOOO00O00O0O[1], path=O0000OOOO00O00O0O[0]))
        O0OOO00O0OO00000O = public.ExecShell("xfs_quota -x -c 'project -s -p {path} {quota_id}'".format(path=O0000000O0OOOOO0O, quota_id=O0OO00OOO00O0O00O))
        if O0OOO00O0OO00000O[1]:
            return public.returnMsg(False, O0OOO00O0OO00000O[1])
        O0OOO00O0OO00000O = public.ExecShell("xfs_quota -x -c 'limit -p bhard={size}m {quota_id}' {mountpoint}".format(quota_id=O0OO00OOO00O0O00O, size=O0OOO00OOO0OO0000, mountpoint=O0000OOOO00O00O0O))
        if O0OOO00O0OO00000O[1]:
            return public.returnMsg(False, O0OOO00O0OO00000O[1])
        for O00OOO00OOO0000OO in O0OOO000OO0OO0O0O:
            if O00OOO00OOO0000OO['path'] == O0000000O0OOOOO0O:
                O00OOO00OOO0000OO['size'] = O0OOO00OOO0OO0000
                break
        public.writeFile(OOOO000O0OOO000OO.__O00O0OOOO000OO0O0, json.dumps(O0OOO000OO0OO0O0O))
        public.WriteLog('磁盘配额', '修改目录[{path}]的配额限制为: {size}MB'.format(path=O0000000O0OOOOO0O, size=O0OOO00OOO0OO0000))
        return public.returnMsg(True, '修改成功')