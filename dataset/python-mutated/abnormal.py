import sys, os
import time, hashlib, sys, os, json, requests, re, public, random, string, requests, system, panelPlugin, ajax
import psutil
import gevent
import gevent.pool

class abnormal:
    __system = None
    __plugin = None
    __ajax = None

    def __init__(self):
        if False:
            while True:
                i = 10
        if not self.__system:
            self.__system = system.system()
        if not self.__plugin:
            self.__plugin = panelPlugin.panelPlugin()
        if not self.__ajax:
            self.__ajax = ajax.ajax()

    def mysql_server(self, get):
        if False:
            for i in range(10):
                print('nop')
        if os.path.exists('/www/server/mysql'):
            get.sName = 'mysql'
            mysql = self.__plugin.get_soft_find(get)
            if mysql['status']:
                return True
            else:
                return False
        else:
            return True

    def test_mysql_cpu(self, i):
        if False:
            for i in range(10):
                print('nop')
        self.count = 0
        try:
            pp = psutil.Process(i)
            if pp.name() == 'mysqld':
                self.count += float(pp.cpu_percent(interval=0.1))
        except:
            pass
        return self.count

    def mysql_cpu(self, get):
        if False:
            print('Hello World!')
        pool = gevent.pool.Pool(50)
        threads = []
        for i in psutil.pids():
            threads.append(pool.spawn(self.test_mysql_cpu, i))
        gevent.joinall(threads)
        cpu_count = psutil.cpu_count()
        return int(self.count) / cpu_count

    def mysql_count(self, get):
        if False:
            print('Hello World!')
        ret = public.M('config').field('mysql_root').select()
        password = ret[0]['mysql_root']
        sql = ' mysql -uroot -p' + password + ' -e "select User,Host from mysql.user where host=\'%\'" '
        result = public.ExecShell(sql)
        if re.search('Too many connections', result[1]):
            return True
        else:
            return False

    def return_php(self):
        if False:
            while True:
                i = 10
        ret = []
        if not os.path.exists('/www/server/php'):
            return ret
        for i in os.listdir('/www/server/php'):
            php_list = public.get_php_versions()
            if not i in php_list:
                continue
            if os.path.isdir('/www/server/php/' + i):
                ret.append(i)
        return ret

    def php_server(self, get):
        if False:
            i = 10
            return i + 15
        ret = []
        php_count = self.return_php()
        if len(php_count) >= 1:
            for i in php_count:
                get.sName = 'php-%s.%s' % (i[0], i[1])
                mysql = self.__plugin.get_soft_find(get)
                if not mysql:
                    php = {}
                    php['version'] = i
                    php['status'] = False
                    ret.append(php)
                else:
                    php = {}
                    php['version'] = i
                    php['status'] = mysql['status']
                    ret.append(php)
            else:
                return ret
        else:
            return ret

    def php_conn_max(self, get):
        if False:
            i = 10
            return i + 15
        ret = []
        php_count = self.return_php()
        if len(php_count) >= 1:
            for i in php_count:
                get.sName = i
                try:
                    result = public.HttpGet('http://127.0.0.1/phpfpm_' + i + '_status?json')
                    tmp = json.loads(result)
                    php = {}
                    php['version'] = i
                    php['children'] = tmp['max children reached']
                    php['start_time'] = tmp['start time']
                    ret.append(php)
                except:
                    php = {}
                    php['version'] = i
                    php['children'] = 'ERROR'
                    ret.append(php)
            else:
                return ret
        else:
            return ret

    def test_php_cpu(self, i):
        if False:
            print('Hello World!')
        self.count2 = 0
        try:
            pp = psutil.Process(i)
            if pp.name() == 'php-fpm':
                self.count2 += float(pp.cpu_percent(interval=0.1))
        except:
            pass
        return self.count2

    def php_cpu(self, get):
        if False:
            i = 10
            return i + 15
        pool = gevent.pool.Pool(50)
        threads = []
        for i in psutil.pids():
            threads.append(pool.spawn(self.test_php_cpu, i))
        gevent.joinall(threads)
        cpu_count = psutil.cpu_count()
        return int(self.count2) / cpu_count

    def CPU(self, get):
        if False:
            return 10
        cpu = self.__system.GetCpuInfo()
        return cpu

    def Memory(self, get):
        if False:
            for i in range(10):
                print('nop')
        meory = self.__system.GetMemInfo()
        meory = '%.2f' % (float(meory['memRealUsed']) / float(meory['memTotal']) * 100)
        return meory

    def disk(self, get):
        if False:
            return 10
        disk = cpu = self.__system.GetDiskInfo()
        ret = []
        for i in disk:
            if int(i['size'][-1].replace('%', '')) >= 1:
                ret.append(i)
        return ret

    def not_root_user(self, get):
        if False:
            for i in range(10):
                print('nop')
        cmd = "cat /etc/passwd | awk -F: '($3 == 0) { print $1 }'|grep -v '^root$' "
        ret = public.ExecShell(cmd)
        if len(ret[0]):
            return True
        else:
            return False

    def start(self, get):
        if False:
            i = 10
            return i + 15
        ret = {}
        mysql_server = self.mysql_server(get)
        ret['mysql_server'] = mysql_server
        mysql_cpu = self.mysql_cpu(get)
        ret['mysql_cpu'] = mysql_cpu
        mysql_count = self.mysql_count(get)
        ret['mysql_count'] = mysql_count
        php_server = self.php_server(get)
        ret['php_server'] = php_server
        php_conn_max = self.php_conn_max(get)
        ret['php_conn_max'] = php_conn_max
        php_cpu = self.php_cpu(get)
        ret['php_cpu'] = php_cpu
        CPU = self.CPU(get)
        ret['CPU'] = CPU
        Memory = self.Memory(get)
        ret['Memory'] = Memory
        disk = self.disk(get)
        ret['disk'] = disk
        not_root_user = self.not_root_user(get)
        ret['not_root_user'] = not_root_user
        return ret