import io, re, public, os, sys, shutil, json, hashlib, socket, time
try:
    from BTPanel import session
except:
    pass
from panelRedirect import panelRedirect
import site_dir_auth

class panelSite(panelRedirect):
    siteName = None
    sitePath = None
    sitePort = None
    phpVersion = None
    setupPath = None
    isWriteLogs = None
    nginx_conf_bak = '/tmp/backup_nginx.conf'
    apache_conf_bak = '/tmp/backup_apache.conf'
    is_ipv6 = False

    def __init__(self):
        if False:
            print('Hello World!')
        self.setupPath = public.get_setup_path()
        path = self.setupPath + '/panel/vhost/nginx'
        if not os.path.exists(path):
            public.ExecShell('mkdir -p ' + path + ' && chmod -R 644 ' + path)
        path = self.setupPath + '/panel/vhost/apache'
        if not os.path.exists(path):
            public.ExecShell('mkdir -p ' + path + ' && chmod -R 644 ' + path)
        path = self.setupPath + '/panel/vhost/rewrite'
        if not os.path.exists(path):
            public.ExecShell('mkdir -p ' + path + ' && chmod -R 644 ' + path)
        path = self.setupPath + '/stop'
        if not os.path.exists(path + '/index.html'):
            public.ExecShell('mkdir -p ' + path)
            public.ExecShell('wget -O ' + path + '/index.html ' + public.get_url() + '/stop.html &')
        self.__proxyfile = '{}/data/proxyfile.json'.format(public.get_panel_path())
        self.OldConfigFile()
        if os.path.exists(self.nginx_conf_bak):
            os.remove(self.nginx_conf_bak)
        if os.path.exists(self.apache_conf_bak):
            os.remove(self.apache_conf_bak)
        self.is_ipv6 = os.path.exists(self.setupPath + '/panel/data/ipv6.pl')
        sys.setrecursionlimit(1000000)

    def check_default(self):
        if False:
            i = 10
            return i + 15
        nginx = self.setupPath + '/panel/vhost/nginx'
        httpd = self.setupPath + '/panel/vhost/apache'
        httpd_default = '<VirtualHost *:80>\n    ServerAdmin webmaster@example.com\n    DocumentRoot "/www/server/apache/htdocs"\n    ServerName bt.default.com\n    <Directory "/www/server/apache/htdocs">\n        SetOutputFilter DEFLATE\n        Options FollowSymLinks\n        AllowOverride All\n        Order allow,deny\n        Allow from all\n        DirectoryIndex index.html\n    </Directory>\n</VirtualHost>'
        listen_ipv6 = ''
        if self.is_ipv6:
            listen_ipv6 = '\n    listen [::]:80;'
        nginx_default = 'server\n{\n    listen 80;%s\n    server_name _;\n    index index.html;\n    root /www/server/nginx/html;\n}' % listen_ipv6
        if not os.path.exists(httpd + '/0.default.conf') and (not os.path.exists(httpd + '/default.conf')):
            public.writeFile(httpd + '/0.default.conf', httpd_default)
        if not os.path.exists(nginx + '/0.default.conf') and (not os.path.exists(nginx + '/default.conf')):
            public.writeFile(nginx + '/0.default.conf', nginx_default)

    def apacheAddPort(self, port):
        if False:
            return 10
        port = str(port)
        filename = self.setupPath + '/apache/conf/extra/httpd-ssl.conf'
        if os.path.exists(filename):
            ssl_conf = public.readFile(filename)
            if ssl_conf:
                if ssl_conf.find('Listen 443') != -1:
                    ssl_conf = ssl_conf.replace('Listen 443', '')
                    public.writeFile(filename, ssl_conf)
        filename = self.setupPath + '/apache/conf/httpd.conf'
        if not os.path.exists(filename):
            return
        allConf = public.readFile(filename)
        rep = 'Listen\\s+([0-9]+)\\n'
        tmp = re.findall(rep, allConf)
        if not tmp:
            return False
        for key in tmp:
            if key == port:
                return False
        listen = '\nListen ' + tmp[0] + '\n'
        listen_ipv6 = ''
        allConf = allConf.replace(listen, listen + 'Listen ' + port + listen_ipv6 + '\n')
        public.writeFile(filename, allConf)
        return True

    def apacheAdd(self):
        if False:
            return 10
        import time
        listen = ''
        if self.sitePort != '80':
            self.apacheAddPort(self.sitePort)
        acc = public.md5(str(time.time()))[0:8]
        try:
            httpdVersion = public.readFile(self.setupPath + '/apache/version.pl').strip()
        except:
            httpdVersion = ''
        if httpdVersion == '2.2':
            vName = ''
            if self.sitePort != '80' and self.sitePort != '443':
                vName = 'NameVirtualHost  *:' + self.sitePort + '\n'
            phpConfig = ''
            apaOpt = 'Order allow,deny\n\t\tAllow from all'
        else:
            vName = ''
            phpConfig = '\n    #PHP\n    <FilesMatch \\.php$>\n            SetHandler "proxy:%s"\n    </FilesMatch>\n    ' % (public.get_php_proxy(self.phpVersion, 'apache'),)
            apaOpt = 'Require all granted'
        conf = '%s<VirtualHost *:%s>\n    ServerAdmin webmaster@example.com\n    DocumentRoot "%s"\n    ServerName %s.%s\n    ServerAlias %s\n    #errorDocument 404 /404.html\n    ErrorLog "%s-error_log"\n    CustomLog "%s-access_log" combined\n    \n    #DENY FILES\n     <Files ~ (\\.user.ini|\\.htaccess|\\.git|\\.svn|\\.project|LICENSE|README.md)$>\n       Order allow,deny\n       Deny from all\n    </Files>\n    %s\n    #PATH\n    <Directory "%s">\n        SetOutputFilter DEFLATE\n        Options FollowSymLinks\n        AllowOverride All\n        %s\n        DirectoryIndex index.php index.html index.htm default.php default.html default.htm\n    </Directory>\n</VirtualHost>' % (vName, self.sitePort, self.sitePath, acc, self.siteName, self.siteName, public.GetConfigValue('logs_path') + '/' + self.siteName, public.GetConfigValue('logs_path') + '/' + self.siteName, phpConfig, self.sitePath, apaOpt)
        htaccess = self.sitePath + '/.htaccess'
        if not os.path.exists(htaccess):
            public.writeFile(htaccess, ' ')
        public.ExecShell('chmod -R 755 ' + htaccess)
        public.ExecShell('chown -R www:www ' + htaccess)
        filename = self.setupPath + '/panel/vhost/apache/' + self.siteName + '.conf'
        public.writeFile(filename, conf)
        return True

    def nginxAdd(self):
        if False:
            print('Hello World!')
        listen_ipv6 = ''
        if self.is_ipv6:
            listen_ipv6 = '\n    listen [::]:%s;' % self.sitePort
        conf = 'server\n{{\n    listen {listen_port};{listen_ipv6}\n    server_name {site_name};\n    index index.php index.html index.htm default.php default.htm default.html;\n    root {site_path};\n    \n    #SSL-START {ssl_start_msg}\n    #error_page 404/404.html;\n    #SSL-END\n    \n    #ERROR-PAGE-START  {err_page_msg}\n    #error_page 404 /404.html;\n    #error_page 502 /502.html;\n    #ERROR-PAGE-END\n    \n    #PHP-INFO-START  {php_info_start}\n    include enable-php-{php_version}.conf;\n    #PHP-INFO-END\n    \n    #REWRITE-START {rewrite_start_msg}\n    include {setup_path}/panel/vhost/rewrite/{site_name}.conf;\n    #REWRITE-END\n    \n    #禁止访问的文件或目录\n    location ~ ^/(\\.user.ini|\\.htaccess|\\.git|\\.svn|\\.project|LICENSE|README.md)\n    {{\n        return 404;\n    }}\n    \n    #一键申请SSL证书验证目录相关设置\n    location ~ \\.well-known{{\n        allow all;\n    }}\n    \n    location ~ .*\\.(gif|jpg|jpeg|png|bmp|swf)$\n    {{\n        expires      30d;\n        error_log /dev/null;\n        access_log /dev/null;\n    }}\n    \n    location ~ .*\\.(js|css)?$\n    {{\n        expires      12h;\n        error_log /dev/null;\n        access_log /dev/null; \n    }}\n    access_log  {log_path}/{site_name}.log;\n    error_log  {log_path}/{site_name}.error.log;\n}}'.format(listen_port=self.sitePort, listen_ipv6=listen_ipv6, site_path=self.sitePath, ssl_start_msg=public.getMsg('NGINX_CONF_MSG1'), err_page_msg=public.getMsg('NGINX_CONF_MSG2'), php_info_start=public.getMsg('NGINX_CONF_MSG3'), php_version=self.phpVersion, setup_path=self.setupPath, rewrite_start_msg=public.getMsg('NGINX_CONF_MSG4'), log_path=public.GetConfigValue('logs_path'), site_name=self.siteName)
        filename = self.setupPath + '/panel/vhost/nginx/' + self.siteName + '.conf'
        public.writeFile(filename, conf)
        urlrewritePath = self.setupPath + '/panel/vhost/rewrite'
        urlrewriteFile = urlrewritePath + '/' + self.siteName + '.conf'
        if not os.path.exists(urlrewritePath):
            os.makedirs(urlrewritePath)
        open(urlrewriteFile, 'w+').close()
        if not os.path.exists(urlrewritePath):
            public.writeFile(urlrewritePath, '')
        return True

    def rep_site_config(self, get):
        if False:
            return 10
        self.siteName = get.siteName
        siteInfo = public.M('sites').where('name=?', (self.siteName,)).field('id,path,port').find()
        siteInfo['domains'] = public.M('domains').where('pid=?', (siteInfo['id'],)).field('name,port').select()
        siteInfo['binding'] = public.M('binding').where('pid=?', (siteInfo['id'],)).field('domain,path').select()

    def openlitespeed_add_site(self, get, init_args=None):
        if False:
            i = 10
            return i + 15
        if not self.sitePath:
            return public.returnMsg(False, 'Not specify parameter [sitePath]')
        if init_args:
            self.siteName = init_args['sitename']
            self.phpVersion = init_args['phpv']
            self.sitePath = init_args['rundir']
        conf_dir = self.setupPath + '/panel/vhost/openlitespeed/'
        if not os.path.exists(conf_dir):
            os.makedirs(conf_dir)
        file = conf_dir + self.siteName + '.conf'
        v_h = '\n#VHOST_TYPE BT_SITENAME START\nvirtualhost BT_SITENAME {\nvhRoot BT_RUN_PATH\nconfigFile /www/server/panel/vhost/openlitespeed/detail/BT_SITENAME.conf\nallowSymbolLink 1\nenableScript 1\nrestrained 1\nsetUIDMode 0\n}\n#VHOST_TYPE BT_SITENAME END\n'
        self.old_name = self.siteName
        if hasattr(get, 'dirName'):
            self.siteName = self.siteName + '_' + get.dirName
            v_h = v_h.replace('VHOST_TYPE', 'SUBDIR')
            v_h = v_h.replace('BT_SITENAME', self.siteName)
            v_h = v_h.replace('BT_RUN_PATH', self.sitePath)
        else:
            self.openlitespeed_domain(get)
            v_h = v_h.replace('VHOST_TYPE', 'VHOST')
            v_h = v_h.replace('BT_SITENAME', self.siteName)
            v_h = v_h.replace('BT_RUN_PATH', self.sitePath)
        public.writeFile(file, v_h, 'a+')
        conf = 'docRoot                   $VH_ROOT\nvhDomain                  $VH_NAME\nadminEmails               example@example.com\nenableGzip                1\nenableIpGeo               1\n\nindex  {\n  useServer               0\n  indexFiles index.php,index.html\n}\n\nerrorlog /www/wwwlogs/$VH_NAME_ols.error_log {\n  useServer               0\n  logLevel                ERROR\n  rollingSize             10M\n}\n\naccesslog /www/wwwlogs/$VH_NAME_ols.access_log {\n  useServer               0\n  logFormat               \'%{X-Forwarded-For}i %h %l %u %t "%r" %>s %b "%{Referer}i" "%{User-Agent}i"\'\n  logHeaders              5\n  rollingSize             10M\n  keepDays                10  compressArchive         1\n}\n\nscripthandler  {\n  add                     lsapi:BT_EXTP_NAME php\n}\n\nextprocessor BTSITENAME {\n  type                    lsapi\n  address                 UDS://tmp/lshttpd/BT_EXTP_NAME.sock\n  maxConns                20\n  env                     LSAPI_CHILDREN=20\n  initTimeout             600\n  retryTimeout            0\n  persistConn             1\n  pcKeepAliveTimeout      1\n  respBuffer              0\n  autoStart               1\n  path                    /usr/local/lsws/lsphpBTPHPV/bin/lsphp\n  extUser                 www\n  extGroup                www\n  memSoftLimit            2047M\n  memHardLimit            2047M\n  procSoftLimit           400\n  procHardLimit           500\n}\n\nphpIniOverride  {\nphp_admin_value open_basedir "/tmp/:BT_RUN_PATH"\n}\n\nexpires {\n    enableExpires           1\n    expiresByType           image/*=A43200,text/css=A43200,application/x-javascript=A43200,application/javascript=A43200,font/*=A43200,application/x-font-ttf=A43200\n}\n\nrewrite  {\n  enable                  1\n  autoLoadHtaccess        1\n  include /www/server/panel/vhost/openlitespeed/proxy/BTSITENAME/urlrewrite/*.conf\n  include /www/server/panel/vhost/apache/redirect/BTSITENAME/*.conf\n  include /www/server/panel/vhost/openlitespeed/redirect/BTSITENAME/*.conf\n}\ninclude /www/server/panel/vhost/openlitespeed/proxy/BTSITENAME/*.conf\n'
        open_base_path = self.sitePath
        if self.sitePath[-1] != '/':
            open_base_path = self.sitePath + '/'
        conf = conf.replace('BT_RUN_PATH', open_base_path)
        conf = conf.replace('BT_EXTP_NAME', self.siteName)
        conf = conf.replace('BTPHPV', self.phpVersion)
        conf = conf.replace('BTSITENAME', self.siteName)
        conf_dir = self.setupPath + '/panel/vhost/openlitespeed/detail/'
        if not os.path.exists(conf_dir):
            os.makedirs(conf_dir)
        file = conf_dir + self.siteName + '.conf'
        public.writeFile(file, conf)
        return True

    def __process_cvs(self, key):
        if False:
            i = 10
            return i + 15
        import csv
        with open('/tmp/multiple_website.csv') as f:
            f_csv = csv.reader(f)
            return [dict(zip(key, i)) for i in [i for i in f_csv if 'FTP' not in i]]

    def __create_website_mulitiple(self, websites_info, site_path, get):
        if False:
            i = 10
            return i + 15
        create_successfully = {}
        create_failed = {}
        for data in websites_info:
            if not data:
                continue
            try:
                domains = data['website'].split(',')
                website_name = domains[0].split(':')[0]
                data['port'] = '80' if len(domains[0].split(':')) < 2 else domains[0].split(':')[1]
                get.webname = json.dumps({'domain': website_name, 'domainlist': domains[1:], 'count': 0})
                get.path = data['path'] if 'path' in data and data['path'] != '0' and (data['path'] != '1') else site_path + '/' + website_name
                get.version = data['version'] if 'version' in data and data['version'] != '0' else '00'
                get.ftp = 'true' if 'ftp' in data and data['ftp'] == '1' else False
                get.sql = 'true' if 'sql' in data and data['sql'] == '1' else False
                get.port = data['port'] if 'port' in data else '80'
                get.codeing = 'utf8'
                get.type = 'PHP'
                get.type_id = '0'
                get.ps = ''
                create_other = {}
                create_other['db_status'] = False
                create_other['ftp_status'] = False
                if get.sql == 'true':
                    create_other['db_pass'] = get.datapassword = public.gen_password(16)
                    create_other['db_user'] = get.datauser = website_name.replace('.', '_')
                    create_other['db_status'] = True
                if get.ftp == 'true':
                    create_other['ftp_pass'] = get.ftp_password = public.gen_password(16)
                    create_other['ftp_user'] = get.ftp_username = website_name.replace('.', '_')
                    create_other['ftp_status'] = True
                result = self.AddSite(get, multiple=1)
                if 'status' in result:
                    create_failed[domains[0]] = result['msg']
                    continue
                create_successfully[domains[0]] = create_other
            except:
                create_failed[domains[0]] = '创建出错了，请再试一次'
        return {'status': True, 'msg': '创建网站 [ {} ] 成功'.format(','.join(create_successfully)), 'error': create_failed, 'success': create_successfully}

    def create_website_multiple(self, get):
        if False:
            return 10
        '\n            @name 批量创建网站\n            @author zhwen<2020-11-26>\n            @param create_type txt/csv  txt格式为 “网站名|网站路径|是否创建FTP|是否创建数据库|PHP版本” 每个网站一行\n                                                 "aaa.com:88,bbb.com|/www/wwwserver/aaa.com/或1|1/0|1/0|0/73"\n                                        csv格式为 “网站名|网站端口|网站路径|PHP版本|是否创建数据库|是否创建FTP”\n            @param websites_content     "[[aaa.com|80|/www/wwwserver/aaa.com/|1|1|73]...."\n        '
        key = ['website', 'path', 'ftp', 'sql', 'version']
        site_path = public.M('config').getField('sites_path')
        if get.create_type == 'txt':
            websites_info = [dict(zip(key, i)) for i in [i.strip().split('|') for i in json.loads(get.websites_content)]]
        else:
            websites_info = self.__process_cvs(key)
        res = self.__create_website_mulitiple(websites_info, site_path, get)
        public.serviceReload()
        return res

    def AddSite(self, get, multiple=None):
        if False:
            i = 10
            return i + 15
        self.check_default()
        isError = public.checkWebConfig()
        if isError != True:
            return public.returnMsg(False, 'ERROR: 检测到配置文件有错误,请先排除后再操作<br><br><a style="color:red;">' + isError.replace('\n', '<br>') + '</a>')
        import json, files
        get.path = self.__get_site_format_path(get.path)
        if not public.check_site_path(get.path):
            (a, c) = public.get_sys_path()
            return public.returnMsg(False, '请不要将网站根目录设置到以下关键目录中: <br>{}'.format('<br>'.join(a + c)))
        try:
            siteMenu = json.loads(get.webname)
        except:
            return public.returnMsg(False, 'webname参数格式不正确，应该是可被解析的JSON字符串')
        self.siteName = self.ToPunycode(siteMenu['domain'].strip().split(':')[0]).strip().lower()
        self.sitePath = self.ToPunycodePath(self.GetPath(get.path.replace(' ', ''))).strip()
        self.sitePort = get.port.strip().replace(' ', '')
        if self.sitePort == '':
            get.port = '80'
        if not public.checkPort(self.sitePort):
            return public.returnMsg(False, 'SITE_ADD_ERR_PORT')
        for domain in siteMenu['domainlist']:
            if not len(domain.split(':')) == 2:
                continue
            if not public.checkPort(domain.split(':')[1]):
                return public.returnMsg(False, 'SITE_ADD_ERR_PORT')
        if hasattr(get, 'version'):
            self.phpVersion = get.version.replace(' ', '')
        else:
            self.phpVersion = '00'
        if not self.phpVersion:
            self.phpVersion = '00'
        php_version = self.GetPHPVersion(get)
        is_phpv = False
        for php_v in php_version:
            if self.phpVersion == php_v['version']:
                is_phpv = True
                break
        if not is_phpv:
            return public.returnMsg(False, '指定PHP版本不存在!')
        domain = None
        if not self.__check_site_path(self.sitePath):
            return public.returnMsg(False, 'PATH_ERROR')
        if len(self.phpVersion) < 2:
            return public.returnMsg(False, 'SITE_ADD_ERR_PHPEMPTY')
        reg = '^([\\w\\-\\*]{1,100}\\.){1,24}([\\w\\-]{1,24}|[\\w\\-]{1,24}\\.[\\w\\-]{1,24})$'
        if not re.match(reg, self.siteName):
            return public.returnMsg(False, 'SITE_ADD_ERR_DOMAIN')
        if self.siteName.find('*') != -1:
            return public.returnMsg(False, 'SITE_ADD_ERR_DOMAIN_TOW')
        if self.sitePath[-1] == '.':
            return public.returnMsg(False, '网站目录结尾不可以是 "."')
        if not domain:
            domain = self.siteName
        sql = public.M('sites')
        if sql.where('name=?', (self.siteName,)).count():
            return public.returnMsg(False, 'SITE_ADD_ERR_EXISTS')
        opid = public.M('domain').where('name=?', (self.siteName,)).getField('pid')
        if opid:
            if public.M('sites').where('id=?', (opid,)).count():
                return public.returnMsg(False, 'SITE_ADD_ERR_DOMAIN_EXISTS')
            public.M('domain').where('pid=?', (opid,)).delete()
        if public.M('binding').where('domain=?', (self.siteName,)).count():
            return public.returnMsg(False, 'SITE_ADD_ERR_DOMAIN_EXISTS')
        if not os.path.exists(self.sitePath):
            try:
                os.makedirs(self.sitePath)
            except Exception as ex:
                return public.returnMsg(False, '创建根目录失败, %s' % ex)
            public.ExecShell('chmod -R 755 ' + self.sitePath)
            public.ExecShell('chown -R www:www ' + self.sitePath)
        self.DelUserInI(self.sitePath)
        userIni = self.sitePath + '/.user.ini'
        if not os.path.exists(userIni):
            public.writeFile(userIni, 'open_basedir=' + self.sitePath + '/:/tmp/')
            public.ExecShell('chmod 644 ' + userIni)
            public.ExecShell('chown root:root ' + userIni)
            public.ExecShell('chattr +i ' + userIni)
        ngx_open_basedir_path = self.setupPath + '/panel/vhost/open_basedir/nginx'
        if not os.path.exists(ngx_open_basedir_path):
            os.makedirs(ngx_open_basedir_path, 384)
        ngx_open_basedir_file = ngx_open_basedir_path + '/{}.conf'.format(self.siteName)
        ngx_open_basedir_body = 'set $bt_safe_dir "open_basedir";\nset $bt_safe_open "{}/:/tmp/";'.format(self.sitePath)
        public.writeFile(ngx_open_basedir_file, ngx_open_basedir_body)
        index = self.sitePath + '/index.html'
        if not os.path.exists(index):
            public.writeFile(index, public.readFile('data/defaultDoc.html'))
            public.ExecShell('chmod -R 755 ' + index)
            public.ExecShell('chown -R www:www ' + index)
        doc404 = self.sitePath + '/404.html'
        if not os.path.exists(doc404):
            public.writeFile(doc404, public.readFile('data/404.html'))
            public.ExecShell('chmod -R 755 ' + doc404)
            public.ExecShell('chown -R www:www ' + doc404)
        result = self.nginxAdd()
        result = self.apacheAdd()
        result = self.openlitespeed_add_site(get)
        if not result:
            return public.returnMsg(False, 'SITE_ADD_ERR_WRITE')
        ps = public.xssencode2(get.ps)
        if self.sitePort != '80':
            import firewalls
            get.port = self.sitePort
            get.ps = self.siteName
            firewalls.firewalls().AddAcceptPort(get)
        if not hasattr(get, 'type_id'):
            get.type_id = 0
        public.check_domain_cloud(self.siteName)
        get.pid = sql.table('sites').add('name,path,status,ps,type_id,addtime', (self.siteName, self.sitePath, '1', ps, get.type_id, public.getDate()))
        for domain in siteMenu['domainlist']:
            get.domain = domain
            get.webname = self.siteName
            get.id = str(get.pid)
            self.AddDomain(get, multiple)
        sql.table('domain').add('pid,name,port,addtime', (get.pid, self.siteName, self.sitePort, public.getDate()))
        data = {}
        data['siteStatus'] = True
        data['siteId'] = get.pid
        data['ftpStatus'] = False
        if get.ftp == 'true':
            import ftp
            get.ps = self.siteName
            result = ftp.ftp().AddUser(get)
            if result['status']:
                data['ftpStatus'] = True
                data['ftpUser'] = get.ftp_username
                data['ftpPass'] = get.ftp_password
        data['databaseStatus'] = False
        if get.sql == 'true' or get.sql == 'MySQL':
            import database
            if len(get.datauser) > 16:
                get.datauser = get.datauser[:16]
            get.name = get.datauser
            get.db_user = get.datauser
            get.password = get.datapassword
            get.address = '127.0.0.1'
            get.ps = self.siteName
            result = database.database().AddDatabase(get)
            if result['status']:
                data['databaseStatus'] = True
                data['databaseUser'] = get.datauser
                data['databasePass'] = get.datapassword
        if not multiple:
            public.serviceReload()
        public.WriteLog('TYPE_SITE', 'SITE_ADD_SUCCESS', (self.siteName,))
        return data

    def __get_site_format_path(self, path):
        if False:
            print('Hello World!')
        path = path.replace('//', '/')
        if path[-1:] == '/':
            path = path[:-1]
        return path

    def __check_site_path(self, path):
        if False:
            for i in range(10):
                print('nop')
        path = self.__get_site_format_path(path)
        other_path = public.M('config').where('id=?', ('1',)).field('sites_path,backup_path').find()
        if path == other_path['sites_path'] or path == other_path['backup_path']:
            return False
        return True

    def delete_website_multiple(self, get):
        if False:
            i = 10
            return i + 15
        '\n            @name 批量删除网站\n            @author zhwen<2020-11-17>\n            @param sites_id "1,2"\n            @param ftp 0/1\n            @param database 0/1\n            @param  path 0/1\n        '
        sites_id = get.sites_id.split(',')
        del_successfully = []
        del_failed = {}
        for site_id in sites_id:
            get.id = site_id
            get.webname = public.M('sites').where('id=?', (site_id,)).getField('name')
            if not get.webname:
                continue
            try:
                self.DeleteSite(get, multiple=1)
                del_successfully.append(get.webname)
            except:
                del_failed[get.webname] = '删除时出错了，请再试一次'
                pass
        public.serviceReload()
        return {'status': True, 'msg': '删除网站 [ {} ] 成功'.format(','.join(del_successfully)), 'error': del_failed, 'success': del_successfully}

    def DeleteSite(self, get, multiple=None):
        if False:
            return 10
        proxyconf = self.__read_config(self.__proxyfile)
        id = get.id
        if public.M('sites').where('id=?', (id,)).count() < 1:
            return public.returnMsg(False, '指定站点不存在!')
        siteName = get.webname
        get.siteName = siteName
        self.CloseTomcat(get)
        for i in range(len(proxyconf) - 1, -1, -1):
            if proxyconf[i]['sitename'] == siteName:
                del proxyconf[i]
        self.__write_config(self.__proxyfile, proxyconf)
        m_path = self.setupPath + '/panel/vhost/nginx/proxy/' + siteName
        if os.path.exists(m_path):
            public.ExecShell('rm -rf %s' % m_path)
        m_path = self.setupPath + '/panel/vhost/apache/proxy/' + siteName
        if os.path.exists(m_path):
            public.ExecShell('rm -rf %s' % m_path)
        _dir_aith_file = '%s/panel/data/site_dir_auth.json' % self.setupPath
        _dir_aith_conf = public.readFile(_dir_aith_file)
        if _dir_aith_conf:
            try:
                _dir_aith_conf = json.loads(_dir_aith_conf)
                if siteName in _dir_aith_conf:
                    del _dir_aith_conf[siteName]
            except:
                pass
        self.__write_config(_dir_aith_file, _dir_aith_conf)
        dir_aith_path = self.setupPath + '/panel/vhost/nginx/dir_auth/' + siteName
        if os.path.exists(dir_aith_path):
            public.ExecShell('rm -rf %s' % dir_aith_path)
        dir_aith_path = self.setupPath + '/panel/vhost/apache/dir_auth/' + siteName
        if os.path.exists(dir_aith_path):
            public.ExecShell('rm -rf %s' % dir_aith_path)
        __redirectfile = '%s/panel/data/redirect.conf' % self.setupPath
        redirectconf = self.__read_config(__redirectfile)
        for i in range(len(redirectconf) - 1, -1, -1):
            if redirectconf[i]['sitename'] == siteName:
                del redirectconf[i]
        self.__write_config(__redirectfile, redirectconf)
        m_path = self.setupPath + '/panel/vhost/nginx/redirect/' + siteName
        if os.path.exists(m_path):
            public.ExecShell('rm -rf %s' % m_path)
        m_path = self.setupPath + '/panel/vhost/apache/redirect/' + siteName
        if os.path.exists(m_path):
            public.ExecShell('rm -rf %s' % m_path)
        confPath = self.setupPath + '/panel/vhost/nginx/' + siteName + '.conf'
        if os.path.exists(confPath):
            os.remove(confPath)
        confPath = self.setupPath + '/panel/vhost/apache/' + siteName + '.conf'
        if os.path.exists(confPath):
            os.remove(confPath)
        open_basedir_file = self.setupPath + '/panel/vhost/open_basedir/nginx/' + siteName + '.conf'
        if os.path.exists(open_basedir_file):
            os.remove(open_basedir_file)
        vhost_file = '/www/server/panel/vhost/openlitespeed/{}.conf'.format(siteName)
        if os.path.exists(vhost_file):
            public.ExecShell('rm -f {}*'.format(vhost_file))
        vhost_detail_file = '/www/server/panel/vhost/openlitespeed/detail/{}.conf'.format(siteName)
        if os.path.exists(vhost_detail_file):
            public.ExecShell('rm -f {}*'.format(vhost_detail_file))
        vhost_ssl_file = '/www/server/panel/vhost/openlitespeed/detail/ssl/{}.conf'.format(siteName)
        if os.path.exists(vhost_ssl_file):
            public.ExecShell('rm -f {}*'.format(vhost_ssl_file))
        vhost_sub_file = '/www/server/panel/vhost/openlitespeed/detail/{}_sub.conf'.format(siteName)
        if os.path.exists(vhost_sub_file):
            public.ExecShell('rm -f {}*'.format(vhost_sub_file))
        vhost_redirect_file = '/www/server/panel/vhost/openlitespeed/redirect/{}'.format(siteName)
        if os.path.exists(vhost_redirect_file):
            public.ExecShell('rm -rf {}*'.format(vhost_redirect_file))
        vhost_proxy_file = '/www/server/panel/vhost/openlitespeed/proxy/{}'.format(siteName)
        if os.path.exists(vhost_proxy_file):
            public.ExecShell('rm -rf {}*'.format(vhost_proxy_file))
        self._del_ols_listen_conf(siteName)
        filename = '/www/server/panel/vhost/rewrite/' + siteName + '.conf'
        if os.path.exists(filename):
            os.remove(filename)
            public.ExecShell('rm -f ' + confPath + '/rewrite/' + siteName + '_*')
        filename = public.GetConfigValue('logs_path') + '/' + siteName + '*'
        public.ExecShell('rm -f ' + filename)
        public.ExecShell('rm -f ' + public.GetConfigValue('logs_path') + '/' + siteName + '-*')
        if 'path' in get:
            if get.path == '1':
                import files
                get.path = self.__get_site_format_path(public.M('sites').where('id=?', (id,)).getField('path'))
                if self.__check_site_path(get.path):
                    files.files().DeleteDir(get)
                get.path = '1'
        if not multiple:
            public.serviceReload()
        public.M('sites').where('id=?', (id,)).delete()
        public.M('binding').where('pid=?', (id,)).delete()
        public.M('domain').where('pid=?', (id,)).delete()
        public.WriteLog('TYPE_SITE', 'SITE_DEL_SUCCESS', (siteName,))
        if hasattr(get, 'database'):
            if get.database == '1':
                find = public.M('databases').where('pid=?', (id,)).field('id,name').find()
                if find:
                    import database
                    get.name = find['name']
                    get.id = find['id']
                    database.database().DeleteDatabase(get)
        if hasattr(get, 'ftp'):
            if get.ftp == '1':
                find = public.M('ftps').where('pid=?', (id,)).field('id,name').find()
                if find:
                    import ftp
                    get.username = find['name']
                    get.id = find['id']
                    ftp.ftp().DeleteUser(get)
        return public.returnMsg(True, 'SITE_DEL_SUCCESS')

    def _del_ols_listen_conf(self, sitename):
        if False:
            print('Hello World!')
        conf_dir = '/www/server/panel/vhost/openlitespeed/listen/'
        if not os.path.exists(conf_dir):
            return False
        for i in os.listdir(conf_dir):
            file_name = conf_dir + i
            if os.path.isdir(file_name):
                continue
            conf = public.readFile(file_name)
            if not conf:
                continue
            map_rep = 'map\\s+{}.*'.format(sitename)
            conf = re.sub(map_rep, '', conf)
            if 'map' not in conf:
                public.ExecShell('rm -f {}*'.format(file_name))
                continue
            public.writeFile(file_name, conf)

    def ToPunycode(self, domain):
        if False:
            while True:
                i = 10
        import re
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

    def ToPunycodePath(self, path):
        if False:
            print('Hello World!')
        if sys.version_info[0] == 2:
            path = path.encode('utf-8')
        if os.path.exists(path):
            return path
        import re
        match = re.search(u'[\x80-ÿ]+', path)
        if not match:
            match = re.search(u'[一-龥]+', path)
        if not match:
            return path
        npath = ''
        for ph in path.split('/'):
            npath += '/' + self.ToPunycode(ph)
        return npath.replace('//', '/')

    def export_domains(self, args):
        if False:
            print('Hello World!')
        '\n            @name 导出域名列表\n            @author hwliang<2020-10-27>\n            @param args<dict_obj>{\n                siteName: string<网站名称>\n            }\n            @return string\n        '
        pid = public.M('sites').where('name=?', args.siteName).getField('id')
        domains = public.M('domain').where('pid=?', pid).field('name,port').select()
        text_data = []
        for domain in domains:
            text_data.append('{}:{}'.format(domain['name'], domain['port']))
        data = '\n'.join(text_data)
        return public.send_file(data, '{}_domains'.format(args.siteName))

    def import_domains(self, args):
        if False:
            return 10
        '\n            @name 导入域名\n            @author hwliang<2020-10-27>\n            @param args<dict_obj>{\n                siteName: string<网站名称>\n                domains: string<域名列表> 每行一个 格式： 域名:端口\n            }\n            @return string\n        '
        domains_tmp = args.domains.split('\n')
        get = public.dict_obj()
        get.webname = args.siteName
        get.id = public.M('sites').where('name=?', args.siteName).getField('id')
        domains = []
        for domain in domains_tmp:
            if public.M('domain').where('name=?', domain.split(':')[0]).count():
                continue
            domains.append(domain)
        get.domain = ','.join(domains)
        return self.AddDomain(get)

    def AddDomain(self, get, multiple=None):
        if False:
            for i in range(10):
                print('nop')
        isError = public.checkWebConfig()
        if isError != True:
            return public.returnMsg(False, 'ERROR: 检测到配置文件有错误,请先排除后再操作<br><br><a style="color:red;">' + isError.replace('\n', '<br>') + '</a>')
        if not 'domain' in get:
            return public.returnMsg(False, '请填写域名!')
        if len(get.domain) < 3:
            return public.returnMsg(False, 'SITE_ADD_DOMAIN_ERR_EMPTY')
        domains = get.domain.replace(' ', '').split(',')
        for domain in domains:
            if domain == '':
                continue
            domain = domain.strip().split(':')
            get.domain = self.ToPunycode(domain[0]).lower()
            get.port = '80'
            reg = '^([\\w\\-\\*]{1,100}\\.){1,24}([\\w\\-]{1,24}|[\\w\\-]{1,24}\\.[\\w\\-]{1,24})$'
            if not re.match(reg, get.domain):
                return public.returnMsg(False, 'SITE_ADD_DOMAIN_ERR_FORMAT')
            if len(domain) == 2:
                get.port = domain[1]
            if get.port == '':
                get.port = '80'
            if not public.checkPort(get.port):
                return public.returnMsg(False, 'SITE_ADD_DOMAIN_ERR_POER')
            sql = public.M('domain')
            opid = sql.where('name=? AND (port=? OR pid=?)', (get.domain, get.port, get.id)).getField('pid')
            if opid:
                if public.M('sites').where('id=?', (opid,)).count():
                    return public.returnMsg(False, 'SITE_ADD_DOMAIN_ERR_EXISTS')
                sql.where('pid=?', (opid,)).delete()
            if public.M('binding').where('domain=?', (get.domain,)).count():
                return public.returnMsg(False, 'SITE_ADD_ERR_DOMAIN_EXISTS')
            self.NginxDomain(get)
            try:
                self.ApacheDomain(get)
                self.openlitespeed_domain(get)
                if self._check_ols_ssl(get.webname):
                    get.port = '443'
                    self.openlitespeed_domain(get)
                    get.port = '80'
            except:
                pass
            if len(domain) == 2:
                get.port = domain[1]
            if get.port != '80':
                import firewalls
                get.ps = get.domain
                firewalls.firewalls().AddAcceptPort(get)
            if not multiple:
                public.serviceReload()
            public.check_domain_cloud(get.domain)
            public.WriteLog('TYPE_SITE', 'DOMAIN_ADD_SUCCESS', (get.webname, get.domain))
            sql.table('domain').add('pid,name,port,addtime', (get.id, get.domain, get.port, public.getDate()))
        return public.returnMsg(True, 'SITE_ADD_DOMAIN')

    def _check_ols_ssl(self, webname):
        if False:
            print('Hello World!')
        conf = public.readFile('/www/server/panel/vhost/openlitespeed/listen/443.conf')
        if conf and webname in conf:
            return True
        return False

    def openlitespeed_set_80_domain(self, get, conf):
        if False:
            print('Hello World!')
        rep = 'map\\s+{}.*'.format(get.webname)
        domains = get.webname.strip().split(',')
        if conf:
            map_tmp = re.search(rep, conf)
            if map_tmp:
                map_tmp = map_tmp.group()
                domains = map_tmp.strip().split(',')
                if not public.inArray(domains, get.domain):
                    new_map = '{},{}'.format(conf, get.domain)
                    conf = re.sub(rep, new_map, conf)
            else:
                map_tmp = '\tmap\t{d} {d}\n'.format(d=domains[0])
                listen_rep = 'secure\\s*0'
                conf = re.sub(listen_rep, 'secure 0\n' + map_tmp, conf)
            return conf
        else:
            rep_default = 'listener\\s+Default\\{(\n|[\\s\\w\\*\\:\\#\\.\\,])*'
            tmp = re.search(rep_default, conf)
            if tmp:
                tmp = tmp.group()
                new_map = '\tmap\t{d} {d}\n'.format(d=domains[0])
                tmp += new_map
                conf = re.sub(rep_default, tmp, conf)
        return conf

    def openlitespeed_domain(self, get):
        if False:
            i = 10
            return i + 15
        listen_dir = '/www/server/panel/vhost/openlitespeed/listen/'
        if not os.path.exists(listen_dir):
            os.makedirs(listen_dir)
        listen_file = listen_dir + get.port + '.conf'
        listen_conf = public.readFile(listen_file)
        try:
            get.webname = json.loads(get.webname)
            get.domain = get.webname['domain'].replace('\r', '')
            get.webname = get.domain + ',' + ','.join(get.webname['domainlist'])
            if get.webname[-1] == ',':
                get.webname = get.webname[:-1]
        except:
            pass
        if listen_conf:
            rep = 'map\\s+{}.*'.format(get.webname)
            map_tmp = re.search(rep, listen_conf)
            if map_tmp:
                map_tmp = map_tmp.group()
                domains = map_tmp.strip().split(',')
                if not public.inArray(domains, get.domain):
                    new_map = '{},{}'.format(map_tmp, get.domain)
                    listen_conf = re.sub(rep, new_map, listen_conf)
            else:
                domains = get.webname.strip().split(',')
                map_tmp = '\tmap\t{d} {d}'.format(d=domains[0])
                listen_rep = 'secure\\s*0'
                listen_conf = re.sub(listen_rep, 'secure 0\n' + map_tmp, listen_conf)
        else:
            listen_conf = '\nlistener Default%s{\n    address *:%s\n    secure 0\n    map %s %s\n}\n' % (get.port, get.port, get.webname, get.domain)
        public.writeFile(listen_file, listen_conf)
        return True

    def NginxDomain(self, get):
        if False:
            i = 10
            return i + 15
        file = self.setupPath + '/panel/vhost/nginx/' + get.webname + '.conf'
        conf = public.readFile(file)
        if not conf:
            return
        rep = 'server_name\\s*(.*);'
        tmp = re.search(rep, conf).group()
        domains = tmp.replace(';', '').strip().split(' ')
        if not public.inArray(domains, get.domain):
            newServerName = tmp.replace(';', ' ' + get.domain + ';')
            conf = conf.replace(tmp, newServerName)
        rep = 'listen\\s+[\\[\\]\\:]*([0-9]+).*;'
        tmp = re.findall(rep, conf)
        if not public.inArray(tmp, get.port):
            listen = re.search(rep, conf).group()
            listen_ipv6 = ''
            if self.is_ipv6:
                listen_ipv6 = '\n\t\tlisten [::]:' + get.port + ';'
            conf = conf.replace(listen, listen + '\n\t\tlisten ' + get.port + ';' + listen_ipv6)
        public.writeFile(file, conf)
        return True

    def ApacheDomain(self, get):
        if False:
            print('Hello World!')
        file = self.setupPath + '/panel/vhost/apache/' + get.webname + '.conf'
        conf = public.readFile(file)
        if not conf:
            return
        port = get.port
        siteName = get.webname
        newDomain = get.domain
        find = public.M('sites').where('id=?', (get.id,)).field('id,name,path').find()
        sitePath = find['path']
        siteIndex = 'index.php index.html index.htm default.php default.html default.htm'
        if conf.find('<VirtualHost *:' + port + '>') != -1:
            repV = '<VirtualHost\\s+\\*\\:' + port + '>(.|\n)*</VirtualHost>'
            domainV = re.search(repV, conf).group()
            rep = 'ServerAlias\\s*(.*)\\n'
            tmp = re.search(rep, domainV).group(0)
            domains = tmp.strip().split(' ')
            if not public.inArray(domains, newDomain):
                rs = tmp.replace('\n', '')
                newServerName = rs + ' ' + newDomain + '\n'
                myconf = domainV.replace(tmp, newServerName)
                conf = re.sub(repV, myconf, conf)
            if conf.find('<VirtualHost *:443>') != -1:
                repV = '<VirtualHost\\s+\\*\\:443>(.|\\n)*</VirtualHost>'
                domainV = re.search(repV, conf).group()
                rep = 'ServerAlias\\s*(.*)\\n'
                tmp = re.search(rep, domainV).group(0)
                domains = tmp.strip().split(' ')
                if not public.inArray(domains, newDomain):
                    rs = tmp.replace('\n', '')
                    newServerName = rs + ' ' + newDomain + '\n'
                    myconf = domainV.replace(tmp, newServerName)
                    conf = re.sub(repV, myconf, conf)
        else:
            try:
                httpdVersion = public.readFile(self.setupPath + '/apache/version.pl').strip()
            except:
                httpdVersion = ''
            if httpdVersion == '2.2':
                vName = ''
                if self.sitePort != '80' and self.sitePort != '443':
                    vName = 'NameVirtualHost  *:' + port + '\n'
                phpConfig = ''
                apaOpt = 'Order allow,deny\n\t\tAllow from all'
            else:
                vName = ''
                version = public.get_php_version_conf(conf)
                if len(version) < 2:
                    return public.returnMsg(False, 'PHP_GET_ERR')
                phpConfig = '\n    #PHP\n    <FilesMatch \\.php$>\n            SetHandler "proxy:%s"\n    </FilesMatch>\n    ' % (public.get_php_proxy(version, 'apache'),)
                apaOpt = 'Require all granted'
            newconf = '<VirtualHost *:%s>\n    ServerAdmin webmaster@example.com\n    DocumentRoot "%s"\n    ServerName %s.%s\n    ServerAlias %s\n    #errorDocument 404 /404.html\n    ErrorLog "%s-error_log"\n    CustomLog "%s-access_log" combined\n    %s\n    \n    #DENY FILES\n     <Files ~ (\\.user.ini|\\.htaccess|\\.git|\\.svn|\\.project|LICENSE|README.md)$>\n       Order allow,deny\n       Deny from all\n    </Files>\n    \n    #PATH\n    <Directory "%s">\n        SetOutputFilter DEFLATE\n        Options FollowSymLinks\n        AllowOverride All\n        %s\n        DirectoryIndex %s\n    </Directory>\n</VirtualHost>' % (port, sitePath, siteName, port, newDomain, public.GetConfigValue('logs_path') + '/' + siteName, public.GetConfigValue('logs_path') + '/' + siteName, phpConfig, sitePath, apaOpt, siteIndex)
            conf += '\n\n' + newconf
        if port != '80' and port != '888':
            self.apacheAddPort(port)
        public.writeFile(file, conf)
        return True

    def delete_domain_multiple(self, get):
        if False:
            print('Hello World!')
        '\n            @name 批量删除网站\n            @author zhwen<2020-11-17>\n            @param id "1"\n            @param domains_id 1,2,3\n        '
        domains_id = get.domains_id.split(',')
        get.webname = public.M('sites').where('id=?', (get.id,)).getField('name')
        del_successfully = []
        del_failed = {}
        for domain_id in domains_id:
            get.domain = public.M('domain').where('id=? and pid=?', (domain_id, get.id)).getField('name')
            get.port = str(public.M('domain').where('id=? and pid=?', (domain_id, get.id)).getField('port'))
            if not get.webname:
                continue
            try:
                result = self.DelDomain(get, multiple=1)
                tmp = get.domain + ':' + get.port
                if not result['status']:
                    del_failed[tmp] = result['msg']
                    continue
                del_successfully.append(tmp)
            except:
                tmp = get.domain + ':' + get.port
                del_failed[tmp] = '删除时错误了，请再试一次'
                pass
        public.serviceReload()
        return {'status': True, 'msg': '删除域名 [ {} ] 成功'.format(','.join(del_successfully)), 'error': del_failed, 'success': del_successfully}

    def DelDomain(self, get, multiple=None):
        if False:
            print('Hello World!')
        if not 'id' in get:
            return public.returnMsg(False, '请选择域名')
        if not 'port' in get:
            return public.returnMsg(False, '请选择端口')
        sql = public.M('domain')
        id = get['id']
        port = get.port
        find = sql.where('pid=? AND name=?', (get.id, get.domain)).field('id,name').find()
        domain_count = sql.table('domain').where('pid=?', (id,)).count()
        if domain_count == 1:
            return public.returnMsg(False, 'SITE_DEL_DOMAIN_ERR_ONLY')
        file = self.setupPath + '/panel/vhost/nginx/' + get['webname'] + '.conf'
        conf = public.readFile(file)
        if conf:
            rep = 'server_name\\s+(.+);'
            tmp = re.search(rep, conf).group()
            newServerName = tmp.replace(' ' + get['domain'] + ';', ';')
            newServerName = newServerName.replace(' ' + get['domain'] + ' ', ' ')
            conf = conf.replace(tmp, newServerName)
            rep = 'listen.*[\\s:]+(\\d+).*;'
            tmp = re.findall(rep, conf)
            port_count = sql.table('domain').where('pid=? AND port=?', (get.id, get.port)).count()
            if public.inArray(tmp, port) == True and port_count < 2:
                rep = '\\n*\\s+listen.*[\\s:]+' + port + '\\s*;'
                conf = re.sub(rep, '', conf)
            public.writeFile(file, conf.strip())
        file = self.setupPath + '/panel/vhost/apache/' + get['webname'] + '.conf'
        conf = public.readFile(file)
        if conf:
            try:
                rep = '\\n*<VirtualHost \\*\\:' + port + '>(.|\n){500,1500}</VirtualHost>'
                tmp = re.search(rep, conf).group()
                rep1 = 'ServerAlias\\s+(.+)\n'
                tmp1 = re.findall(rep1, tmp)
                tmp2 = tmp1[0].split(' ')
                if len(tmp2) < 2:
                    conf = re.sub(rep, '', conf)
                    rep = 'NameVirtualHost.+\\:' + port + '\n'
                    conf = re.sub(rep, '', conf)
                else:
                    newServerName = tmp.replace(' ' + get['domain'] + '\n', '\n')
                    newServerName = newServerName.replace(' ' + get['domain'] + ' ', ' ')
                    conf = conf.replace(tmp, newServerName)
                public.writeFile(file, conf.strip())
            except:
                pass
        self._del_ols_domain(get)
        sql.table('domain').where('id=?', (find['id'],)).delete()
        public.WriteLog('TYPE_SITE', 'DOMAIN_DEL_SUCCESS', (get.webname, get.domain))
        if not multiple:
            public.serviceReload()
        return public.returnMsg(True, 'DEL_SUCCESS')

    def _del_ols_domain(self, get):
        if False:
            return 10
        conf_dir = '/www/server/panel/vhost/openlitespeed/listen/'
        if not os.path.exists(conf_dir):
            return False
        for i in os.listdir(conf_dir):
            file_name = conf_dir + i
            if os.path.isdir(file_name):
                continue
            conf = public.readFile(file_name)
            map_rep = 'map\\s+{}\\s+(.*)'.format(get.webname)
            domains = re.search(map_rep, conf)
            if domains:
                domains = domains.group(1).split(',')
                if get.domain in domains:
                    domains.remove(get.domain)
                if len(domains) == 0:
                    os.remove(file_name)
                    continue
                else:
                    domains = ','.join(domains)
                    map_c = 'map\t{} '.format(get.webname) + domains
                    conf = re.sub(map_rep, map_c, conf)
            public.writeFile(file_name, conf)

    def CheckDomainPing(self, get):
        if False:
            print('Hello World!')
        try:
            epass = public.GetRandomString(32)
            spath = get.path + '/.well-known/pki-validation'
            if not os.path.exists(spath):
                public.ExecShell("mkdir -p '" + spath + "'")
            public.writeFile(spath + '/fileauth.txt', epass)
            result = public.httpGet('http://' + get.domain.replace('*.', '') + '/.well-known/pki-validation/fileauth.txt')
            if result == epass:
                return True
            return False
        except:
            return False

    def SetSSL(self, get):
        if False:
            print('Hello World!')
        siteName = get.siteName
        path = '/www/server/panel/vhost/cert/' + siteName
        csrpath = path + '/fullchain.pem'
        keypath = path + '/privkey.pem'
        if get.key.find('KEY') == -1:
            return public.returnMsg(False, 'SITE_SSL_ERR_PRIVATE')
        if get.csr.find('CERTIFICATE') == -1:
            return public.returnMsg(False, 'SITE_SSL_ERR_CERT')
        public.writeFile('/tmp/cert.pl', get.csr)
        if not public.CheckCert('/tmp/cert.pl'):
            return public.returnMsg(False, '证书错误,请粘贴正确的PEM格式证书!')
        backup_cert = '/tmp/backup_cert_' + siteName
        import shutil
        if os.path.exists(backup_cert):
            shutil.rmtree(backup_cert)
        if os.path.exists(path):
            shutil.move(path, backup_cert)
        if os.path.exists(path):
            shutil.rmtree(path)
        public.ExecShell('mkdir -p ' + path)
        public.writeFile(keypath, get.key)
        public.writeFile(csrpath, get.csr)
        result = self.SetSSLConf(get)
        if not result['status']:
            return result
        isError = public.checkWebConfig()
        if type(isError) == str:
            if os.path.exists(path):
                shutil.rmtree(backup_cert)
            shutil.move(backup_cert, path)
            return public.returnMsg(False, 'ERROR: <br><a style="color:red;">' + isError.replace('\n', '<br>') + '</a>')
        public.serviceReload()
        if os.path.exists(path + '/partnerOrderId'):
            os.remove(path + '/partnerOrderId')
        if os.path.exists(path + '/certOrderId'):
            os.remove(path + '/certOrderId')
        p_file = '/etc/letsencrypt/live/' + get.siteName
        if os.path.exists(p_file):
            shutil.rmtree(p_file)
        public.WriteLog('TYPE_SITE', 'SITE_SSL_SAVE_SUCCESS')
        if os.path.exists(backup_cert):
            shutil.rmtree(backup_cert)
        return public.returnMsg(True, 'SITE_SSL_SUCCESS')

    def GetRunPath(self, get):
        if False:
            i = 10
            return i + 15
        if not hasattr(get, 'id'):
            if hasattr(get, 'siteName'):
                get.id = public.M('sites').where('name=?', (get.siteName,)).getField('id')
            else:
                get.id = public.M('sites').where('path=?', (get.path,)).getField('id')
        if not get.id:
            return False
        if type(get.id) == list:
            get.id = get.id[0]['id']
        result = self.GetSiteRunPath(get)
        if 'runPath' in result:
            return result['runPath']
        return False

    def CreateLet(self, get):
        if False:
            while True:
                i = 10
        domains = json.loads(get.domains)
        if not len(domains):
            return public.returnMsg(False, '请选择域名')
        file_auth = True
        if hasattr(get, 'dnsapi'):
            file_auth = False
        if not hasattr(get, 'dnssleep'):
            get.dnssleep = 10
        email = public.M('users').getField('email')
        if hasattr(get, 'email'):
            if get.email.find('@') == -1:
                get.email = email
            else:
                get.email = get.email.strip()
                public.M('users').where('id=?', (1,)).setField('email', get.email)
        else:
            get.email = email
        for domain in domains:
            if public.checkIp(domain):
                continue
            if domain.find('*.') >= 0 and file_auth:
                return public.returnMsg(False, '泛域名不能使用【文件验证】的方式申请证书!')
        if file_auth:
            get.sitename = get.siteName
            if self.GetRedirectList(get):
                return public.returnMsg(False, 'SITE_SSL_ERR_301')
            if self.GetProxyList(get):
                return public.returnMsg(False, '已开启反向代理的站点无法申请SSL!')
            data = self.get_site_info(get.siteName)
            get.id = data['id']
            runPath = self.GetRunPath(get)
            if runPath != '/':
                if runPath[:1] != '/':
                    runPath = '/' + runPath
            else:
                runPath = ''
            get.site_dir = data['path'] + runPath
        else:
            dns_api_list = self.GetDnsApi(get)
            get.dns_param = None
            for dns in dns_api_list:
                if dns['name'] == get.dnsapi:
                    param = []
                    if not dns['data']:
                        continue
                    for val in dns['data']:
                        param.append(val['value'])
                    get.dns_param = '|'.join(param)
            n_list = ['dns', 'dns_bt']
            if not get.dnsapi in n_list:
                if len(get.dns_param) < 16:
                    return public.returnMsg(False, '请先设置【%s】的API接口参数.' % get.dnsapi)
            if get.dnsapi == 'dns_bt':
                if not os.path.exists('plugin/dns/dns_main.py'):
                    return public.returnMsg(False, '请先到软件商店安装【云解析】，并完成域名NS绑定.')
        self.check_ssl_pack()
        try:
            import panelLets
            public.mod_reload(panelLets)
        except Exception as ex:
            if str(ex).find('No module named requests') != -1:
                public.ExecShell('pip install requests &')
                return public.returnMsg(False, '缺少requests组件，请尝试修复面板!')
            return public.returnMsg(False, str(ex))
        lets = panelLets.panelLets()
        result = lets.apple_lest_cert(get)
        if result['status'] and (not 'code' in result):
            get.onkey = 1
            path = '/www/server/panel/cert/' + get.siteName
            if os.path.exists(path + '/certOrderId'):
                os.remove(path + '/certOrderId')
            result = self.SetSSLConf(get)
        return result

    def get_site_info(self, siteName):
        if False:
            for i in range(10):
                print('nop')
        data = public.M('sites').where('name=?', siteName).field('id,path,name').find()
        return data

    def check_ssl_pack(self):
        if False:
            print('Hello World!')
        try:
            import requests
        except:
            public.ExecShell('pip install requests')
        try:
            import OpenSSL
        except:
            public.ExecShell('pip install pyopenssl')

    def Check_DnsApi(self, dnsapi):
        if False:
            while True:
                i = 10
        dnsapis = self.GetDnsApi(None)
        for dapi in dnsapis:
            if dapi['name'] == dnsapi:
                if not dapi['data']:
                    return True
                for d in dapi['data']:
                    if d['key'] == '':
                        return False
        return True

    def GetDnsApi(self, get):
        if False:
            i = 10
            return i + 15
        api_path = './config/dns_api.json'
        api_init = './config/dns_api_init.json'
        if not os.path.exists(api_path):
            if os.path.exists(api_init):
                import shutil
                shutil.copyfile(api_init, api_path)
        apis = json.loads(public.ReadFile(api_path))
        path = '/root/.acme.sh'
        if not os.path.exists(path + '/account.conf'):
            path = '/.acme.sh'
        account = public.readFile(path + '/account.conf')
        if not account:
            account = ''
        is_write = False
        for i in range(len(apis)):
            if not apis[i]['data']:
                continue
            for j in range(len(apis[i]['data'])):
                if apis[i]['data'][j]['value']:
                    continue
                match = re.search(apis[i]['data'][j]['key'] + "\\s*=\\s*'(.+)'", account)
                if match:
                    apis[i]['data'][j]['value'] = match.groups()[0]
                if apis[i]['data'][j]['value']:
                    is_write = True
        if is_write:
            public.writeFile('./config/dns_api.json', json.dumps(apis))
        result = []
        for i in apis:
            if i['name'] == 'Dns_com':
                continue
            result.insert(0, i)
        return result

    def SetDnsApi(self, get):
        if False:
            return 10
        pdata = json.loads(get.pdata)
        apis = json.loads(public.ReadFile('./config/dns_api.json'))
        is_write = False
        for key in pdata.keys():
            for i in range(len(apis)):
                if not apis[i]['data']:
                    continue
                for j in range(len(apis[i]['data'])):
                    if apis[i]['data'][j]['key'] != key:
                        continue
                    apis[i]['data'][j]['value'] = pdata[key]
                    is_write = True
        if is_write:
            public.writeFile('./config/dns_api.json', json.dumps(apis))
        return public.returnMsg(True, '设置成功!')

    def GetSiteDomains(self, get):
        if False:
            print('Hello World!')
        data = {}
        domains = public.M('domain').where('pid=?', (get.id,)).field('name,id').select()
        binding = public.M('binding').where('pid=?', (get.id,)).field('domain,id').select()
        if type(binding) == str:
            return binding
        for b in binding:
            tmp = {}
            tmp['name'] = b['domain']
            tmp['id'] = b['id']
            tmp['binding'] = True
            domains.append(tmp)
        data['domains'] = domains
        data['email'] = public.M('users').where('id=?', (1,)).getField('email')
        if data['email'] == '287962566@qq.com':
            data['email'] = ''
        return data

    def GetFormatSSLResult(self, result):
        if False:
            i = 10
            return i + 15
        try:
            import re
            rep = '\\s*Domain:.+\n\\s+Type:.+\n\\s+Detail:.+'
            tmps = re.findall(rep, result)
            statusList = []
            for tmp in tmps:
                arr = tmp.strip().split('\n')
                status = {}
                for ar in arr:
                    tmp1 = ar.strip().split(':')
                    status[tmp1[0].strip()] = tmp1[1].strip()
                    if len(tmp1) > 2:
                        status[tmp1[0].strip()] = tmp1[1].strip() + ':' + tmp1[2]
                statusList.append(status)
            return statusList
        except:
            return None

    def get_tls13(self):
        if False:
            print('Hello World!')
        nginx_bin = '/www/server/nginx/sbin/nginx'
        nginx_v = public.ExecShell(nginx_bin + ' -V 2>&1')[0]
        nginx_v_re = re.findall('nginx/(\\d\\.\\d+).+OpenSSL\\s+(\\d\\.\\d+)', nginx_v, re.DOTALL)
        if nginx_v_re:
            if nginx_v_re[0][0] in ['1.8', '1.9', '1.7', '1.6', '1.5', '1.4']:
                return ''
            if float(nginx_v_re[0][0]) >= 1.15 and float(nginx_v_re[0][-1]) >= 1.1:
                return ' TLSv1.3'
        else:
            _v = re.search('nginx/1\\.1(5|6|7|8|9).\\d', nginx_v)
            if not _v:
                _v = re.search('nginx/1\\.2\\d\\.\\d', nginx_v)
            openssl_v = public.ExecShell(nginx_bin + ' -V 2>&1|grep OpenSSL')[0].find('OpenSSL 1.1.') != -1
            if _v and openssl_v:
                return ' TLSv1.3'
        return ''

    def get_apache_proxy(self, conf):
        if False:
            while True:
                i = 10
        rep = '\n*#引用反向代理规则，注释后配置的反向代理将无效\n+\\s+IncludeOptiona.*'
        proxy = re.search(rep, conf)
        if proxy:
            return proxy.group()
        return ''

    def _get_site_domains(self, sitename):
        if False:
            while True:
                i = 10
        site_id = public.M('sites').where('name=?', (sitename,)).field('id').find()
        domains = public.M('domain').where('pid=?', (site_id['id'],)).field('name').select()
        domains = [d['name'] for d in domains]
        return domains

    def set_ols_ssl(self, get, siteName):
        if False:
            while True:
                i = 10
        listen_conf = self.setupPath + '/panel/vhost/openlitespeed/listen/443.conf'
        conf = public.readFile(listen_conf)
        ssl_conf = '\n        vhssl {\n          keyFile                 /www/server/panel/vhost/cert/BTDOMAIN/privkey.pem\n          certFile                /www/server/panel/vhost/cert/BTDOMAIN/fullchain.pem\n          certChain               1\n          sslProtocol             24\n          ciphers                 EECDH+AESGCM:EDH+AESGCM:AES256+EECDH:AES256+EDH:ECDHE-RSA-AES128-GCM-SHA384:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA128:DHE-RSA-AES128-GCM-SHA384:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES128-GCM-SHA128:ECDHE-RSA-AES128-SHA384:ECDHE-RSA-AES128-SHA128:ECDHE-RSA-AES128-SHA:ECDHE-RSA-AES128-SHA:DHE-RSA-AES128-SHA128:DHE-RSA-AES128-SHA128:DHE-RSA-AES128-SHA:DHE-RSA-AES128-SHA:ECDHE-RSA-DES-CBC3-SHA:EDH-RSA-DES-CBC3-SHA:AES128-GCM-SHA384:AES128-GCM-SHA128:AES128-SHA128:AES128-SHA128:AES128-SHA:AES128-SHA:DES-CBC3-SHA:HIGH:!aNULL:!eNULL:!EXPORT:!DES:!MD5:!PSK:!RC4\n          enableECDHE             1\n          renegProtection         1\n          sslSessionCache         1\n          enableSpdy              15\n          enableStapling           1\n          ocspRespMaxAge           86400\n        }\n        '
        ssl_dir = self.setupPath + '/panel/vhost/openlitespeed/detail/ssl/'
        if not os.path.exists(ssl_dir):
            os.makedirs(ssl_dir)
        ssl_file = ssl_dir + '{}.conf'.format(siteName)
        if not os.path.exists(ssl_file):
            ssl_conf = ssl_conf.replace('BTDOMAIN', siteName)
            public.writeFile(ssl_file, ssl_conf, 'a+')
        include_ssl = '\ninclude {}'.format(ssl_file)
        detail_file = self.setupPath + '/panel/vhost/openlitespeed/detail/{}.conf'.format(siteName)
        public.writeFile(detail_file, include_ssl, 'a+')
        if not conf:
            conf = '\nlistener SSL443 {\n  map                     BTSITENAME BTDOMAIN\n  address                 *:443\n  secure                  1\n  keyFile                 /www/server/panel/vhost/cert/BTSITENAME/privkey.pem\n  certFile                /www/server/panel/vhost/cert/BTSITENAME/fullchain.pem\n  certChain               1\n  sslProtocol             24\n  ciphers                 EECDH+AESGCM:EDH+AESGCM:AES256+EECDH:AES256+EDH:ECDHE-RSA-AES128-GCM-SHA384:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA128:DHE-RSA-AES128-GCM-SHA384:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES128-GCM-SHA128:ECDHE-RSA-AES128-SHA384:ECDHE-RSA-AES128-SHA128:ECDHE-RSA-AES128-SHA:ECDHE-RSA-AES128-SHA:DHE-RSA-AES128-SHA128:DHE-RSA-AES128-SHA128:DHE-RSA-AES128-SHA:DHE-RSA-AES128-SHA:ECDHE-RSA-DES-CBC3-SHA:EDH-RSA-DES-CBC3-SHA:AES128-GCM-SHA384:AES128-GCM-SHA128:AES128-SHA128:AES128-SHA128:AES128-SHA:AES128-SHA:DES-CBC3-SHA:HIGH:!aNULL:!eNULL:!EXPORT:!DES:!MD5:!PSK:!RC4\n  enableECDHE             1\n  renegProtection         1\n  sslSessionCache         1\n  enableSpdy              15\n  enableStapling           1\n  ocspRespMaxAge           86400\n}\n'
        else:
            rep = 'listener\\s*SSL443\\s*{'
            map = '\n  map {s} {s}'.format(s=siteName)
            conf = re.sub(rep, 'listener SSL443 {' + map, conf)
        domain = ','.join(self._get_site_domains(siteName))
        conf = conf.replace('BTSITENAME', siteName).replace('BTDOMAIN', domain)
        public.writeFile(listen_conf, conf)

    def _get_ap_static_security(self, ap_conf):
        if False:
            print('Hello World!')
        if not ap_conf:
            return ''
        ap_static_security = re.search('#SECURITY-START(.|\n)*#SECURITY-END', ap_conf)
        if ap_static_security:
            return ap_static_security.group()
        return ''

    def SetSSLConf(self, get):
        if False:
            while True:
                i = 10
        siteName = get.siteName
        if not 'first_domain' in get:
            get.first_domain = siteName
        file = self.setupPath + '/panel/vhost/nginx/' + siteName + '.conf'
        if not os.path.exists(file):
            file = self.setupPath + '/panel/vhost/nginx/node_' + siteName + '.conf'
        if not os.path.exists(file):
            file = self.setupPath + '/panel/vhost/nginx/java_' + siteName + '.conf'
        ng_file = file
        conf = public.readFile(file)
        if conf:
            if conf.find('ssl_certificate') == -1:
                sslStr = '#error_page 404/404.html;\n    ssl_certificate    /www/server/panel/vhost/cert/%s/fullchain.pem;\n    ssl_certificate_key    /www/server/panel/vhost/cert/%s/privkey.pem;\n    ssl_protocols TLSv1.1 TLSv1.2%s;\n    ssl_ciphers EECDH+CHACHA20:EECDH+CHACHA20-draft:EECDH+AES128:RSA+AES128:EECDH+AES256:RSA+AES256:EECDH+3DES:RSA+3DES:!MD5;\n    ssl_prefer_server_ciphers on;\n    ssl_session_cache shared:SSL:10m;\n    ssl_session_timeout 10m;\n    add_header Strict-Transport-Security "max-age=31536000";\n    error_page 497  https://$host$request_uri;\n' % (get.first_domain, get.first_domain, self.get_tls13())
                if conf.find('ssl_certificate') != -1:
                    public.serviceReload()
                    return public.returnMsg(True, 'SITE_SSL_OPEN_SUCCESS')
                conf = conf.replace('#error_page 404/404.html;', sslStr)
                rep = 'listen.*[\\s:]+(\\d+).*;'
                tmp = re.findall(rep, conf)
                if not public.inArray(tmp, '443'):
                    listen = re.search(rep, conf).group()
                    versionStr = public.readFile('/www/server/nginx/version.pl')
                    http2 = ''
                    if versionStr:
                        if versionStr.find('1.8.1') == -1:
                            http2 = ' http2'
                    default_site = ''
                    if conf.find('default_server') != -1:
                        default_site = ' default_server'
                    listen_ipv6 = ';'
                    if self.is_ipv6:
                        listen_ipv6 = ';\n\tlisten [::]:443 ssl' + http2 + default_site + ';'
                    conf = conf.replace(listen, listen + '\n\tlisten 443 ssl' + http2 + default_site + listen_ipv6)
                shutil.copyfile(file, self.nginx_conf_bak)
                public.writeFile(file, conf)
        file = self.setupPath + '/panel/vhost/apache/' + siteName + '.conf'
        is_node_apache = False
        if not os.path.exists(file):
            is_node_apache = True
            file = self.setupPath + '/panel/vhost/apache/node_' + siteName + '.conf'
        is_java_apache = False
        if not os.path.exists(file):
            is_java_apache = True
            is_node_apache = False
            file = self.setupPath + '/panel/vhost/apache/java_' + siteName + '.conf'
        conf = public.readFile(file)
        ap_static_security = self._get_ap_static_security(conf)
        if conf:
            ap_proxy = self.get_apache_proxy(conf)
            if conf.find('SSLCertificateFile') == -1 and conf.find('VirtualHost') != -1:
                find = public.M('sites').where('name=?', (siteName,)).field('id,path').find()
                tmp = public.M('domain').where('pid=?', (find['id'],)).field('name').select()
                domains = ''
                for key in tmp:
                    domains += key['name'] + ' '
                path = (find['path'] + '/' + self.GetRunPath(get)).replace('//', '/')
                index = 'index.php index.html index.htm default.php default.html default.htm'
                try:
                    httpdVersion = public.readFile(self.setupPath + '/apache/version.pl').strip()
                except:
                    httpdVersion = ''
                if httpdVersion == '2.2':
                    vName = ''
                    phpConfig = ''
                    apaOpt = 'Order allow,deny\n\t\tAllow from all'
                else:
                    vName = ''
                    version = public.get_php_version_conf(conf)
                    if len(version) < 2:
                        return public.returnMsg(False, 'PHP_GET_ERR')
                    phpConfig = '\n    #PHP\n    <FilesMatch \\.php$>\n            SetHandler "proxy:%s"\n    </FilesMatch>\n    ' % (public.get_php_proxy(version, 'apache'),)
                    apaOpt = 'Require all granted'
                sslStr = '%s<VirtualHost *:443>\n    ServerAdmin webmaster@example.com\n    DocumentRoot "%s"\n    ServerName SSL.%s\n    ServerAlias %s\n    #errorDocument 404 /404.html\n    ErrorLog "%s-error_log"\n    CustomLog "%s-access_log" combined\n    %s\n    #SSL\n    SSLEngine On\n    SSLCertificateFile /www/server/panel/vhost/cert/%s/fullchain.pem\n    SSLCertificateKeyFile /www/server/panel/vhost/cert/%s/privkey.pem\n    SSLCipherSuite EECDH+CHACHA20:EECDH+CHACHA20-draft:EECDH+AES128:RSA+AES128:EECDH+AES256:RSA+AES256:EECDH+3DES:RSA+3DES:!MD5\n    SSLProtocol All -SSLv2 -SSLv3 -TLSv1\n    SSLHonorCipherOrder On\n    %s\n    %s\n\n    #DENY FILES\n     <Files ~ (\\.user.ini|\\.htaccess|\\.git|\\.svn|\\.project|LICENSE|README.md)$>\n       Order allow,deny\n       Deny from all\n    </Files>\n\n    #PATH\n    <Directory "%s">\n        SetOutputFilter DEFLATE\n        Options FollowSymLinks\n        AllowOverride All\n        %s\n        DirectoryIndex %s\n    </Directory>\n</VirtualHost>' % (vName, path, siteName, domains, public.GetConfigValue('logs_path') + '/' + siteName, public.GetConfigValue('logs_path') + '/' + siteName, ap_proxy, get.first_domain, get.first_domain, ap_static_security, phpConfig, path, apaOpt, index)
                conf = conf + '\n' + sslStr
                self.apacheAddPort('443')
                shutil.copyfile(file, self.apache_conf_bak)
                public.writeFile(file, conf)
                if is_node_apache:
                    from projectModel.nodejsModel import main
                    m = main()
                    project_find = m.get_project_find(siteName)
                    m.set_apache_config(project_find)
                if is_java_apache:
                    from projectModel.javaModel import main
                    m = main()
                    project_find = m.get_project_find(siteName)
                    m.set_apache_config(project_find)
        self.set_ols_ssl(get, siteName)
        isError = public.checkWebConfig()
        if isError != True:
            if os.path.exists(self.nginx_conf_bak):
                shutil.copyfile(self.nginx_conf_bak, ng_file)
            if os.path.exists(self.apache_conf_bak):
                shutil.copyfile(self.apache_conf_bak, file)
            public.ExecShell('rm -f /tmp/backup_*.conf')
            return public.returnMsg(False, '证书错误: <br><a style="color:red;">' + isError.replace('\n', '<br>') + '</a>')
        sql = public.M('firewall')
        import firewalls
        get.port = '443'
        get.ps = 'HTTPS'
        firewalls.firewalls().AddAcceptPort(get)
        public.serviceReload()
        self.save_cert(get)
        public.WriteLog('TYPE_SITE', 'SITE_SSL_OPEN_SUCCESS', (siteName,))
        result = public.returnMsg(True, 'SITE_SSL_OPEN_SUCCESS')
        result['csr'] = public.readFile('/www/server/panel/vhost/cert/' + get.siteName + '/fullchain.pem')
        result['key'] = public.readFile('/www/server/panel/vhost/cert/' + get.siteName + '/privkey.pem')
        return result

    def save_cert(self, get):
        if False:
            while True:
                i = 10
        import panelSSL
        ss = panelSSL.panelSSL()
        get.keyPath = '/www/server/panel/vhost/cert/' + get.siteName + '/privkey.pem'
        get.certPath = '/www/server/panel/vhost/cert/' + get.siteName + '/fullchain.pem'
        return ss.SaveCert(get)
        return True

    def HttpToHttps(self, get):
        if False:
            return 10
        siteName = get.siteName
        file = self.setupPath + '/panel/vhost/nginx/' + siteName + '.conf'
        if not os.path.exists(file):
            file = self.setupPath + '/panel/vhost/nginx/node_' + siteName + '.conf'
        if not os.path.exists(file):
            file = self.setupPath + '/panel/vhost/nginx/java_' + siteName + '.conf'
        conf = public.readFile(file)
        if conf:
            if conf.find('ssl_certificate') == -1:
                return public.returnMsg(False, '当前未开启SSL')
            to = '#error_page 404/404.html;\n    #HTTP_TO_HTTPS_START\n    if ($server_port !~ 443){\n        rewrite ^(/.*)$ https://$host$1 permanent;\n    }\n    #HTTP_TO_HTTPS_END'
            conf = conf.replace('#error_page 404/404.html;', to)
            public.writeFile(file, conf)
        file = self.setupPath + '/panel/vhost/apache/' + siteName + '.conf'
        if not os.path.exists(file):
            file = self.setupPath + '/panel/vhost/apache/node_' + siteName + '.conf'
        conf = public.readFile(file)
        if conf:
            httpTohttos = 'combined\n    #HTTP_TO_HTTPS_START\n    <IfModule mod_rewrite.c>\n        RewriteEngine on\n        RewriteCond %{SERVER_PORT} !^443$\n        RewriteRule (.*) https://%{SERVER_NAME}$1 [L,R=301]\n    </IfModule>\n    #HTTP_TO_HTTPS_END'
            conf = re.sub('combined', httpTohttos, conf, 1)
            public.writeFile(file, conf)
        conf_dir = '{}/panel/vhost/openlitespeed/redirect/{}/'.format(self.setupPath, siteName)
        if not os.path.exists(conf_dir):
            os.makedirs(conf_dir)
        file = conf_dir + 'force_https.conf'
        ols_force_https = '\n#HTTP_TO_HTTPS_START\n<IfModule mod_rewrite.c>\n    RewriteEngine on\n    RewriteCond %{SERVER_PORT} !^443$\n    RewriteRule (.*) https://%{SERVER_NAME}$1 [L,R=301]\n</IfModule>\n#HTTP_TO_HTTPS_END'
        public.writeFile(file, ols_force_https)
        public.serviceReload()
        return public.returnMsg(True, 'SET_SUCCESS')

    def CloseToHttps(self, get):
        if False:
            i = 10
            return i + 15
        siteName = get.siteName
        file = self.setupPath + '/panel/vhost/nginx/' + siteName + '.conf'
        if not os.path.exists(file):
            file = self.setupPath + '/panel/vhost/nginx/node_' + siteName + '.conf'
        if not os.path.exists(file):
            file = self.setupPath + '/panel/vhost/nginx/java_' + siteName + '.conf'
        conf = public.readFile(file)
        if conf:
            rep = '\n\\s*#HTTP_TO_HTTPS_START(.|\n){1,300}#HTTP_TO_HTTPS_END'
            conf = re.sub(rep, '', conf)
            rep = '\\s+if.+server_port.+\n.+\n\\s+\\s*}'
            conf = re.sub(rep, '', conf)
            public.writeFile(file, conf)
        file = self.setupPath + '/panel/vhost/apache/' + siteName + '.conf'
        conf = public.readFile(file)
        if conf:
            rep = '\n\\s*#HTTP_TO_HTTPS_START(.|\n){1,300}#HTTP_TO_HTTPS_END'
            conf = re.sub(rep, '', conf)
            public.writeFile(file, conf)
        file = '{}/panel/vhost/openlitespeed/redirect/{}/force_https.conf'.format(self.setupPath, siteName)
        public.ExecShell('rm -f {}*'.format(file))
        public.serviceReload()
        return public.returnMsg(True, 'SET_SUCCESS')

    def IsToHttps(self, siteName):
        if False:
            return 10
        file = self.setupPath + '/panel/vhost/nginx/' + siteName + '.conf'
        if not os.path.exists(file):
            file = self.setupPath + '/panel/vhost/nginx/node_' + siteName + '.conf'
        if not os.path.exists(file):
            file = self.setupPath + '/panel/vhost/nginx/java_' + siteName + '.conf'
        conf = public.readFile(file)
        if conf:
            if conf.find('HTTP_TO_HTTPS_START') != -1:
                return True
            if conf.find('$server_port !~ 443') != -1:
                return True
        return False

    def CloseSSLConf(self, get):
        if False:
            return 10
        siteName = get.siteName
        file = self.setupPath + '/panel/vhost/nginx/' + siteName + '.conf'
        if not os.path.exists(file):
            file = self.setupPath + '/panel/vhost/nginx/node_' + siteName + '.conf'
        if not os.path.exists(file):
            file = self.setupPath + '/panel/vhost/nginx/java_' + siteName + '.conf'
        conf = public.readFile(file)
        if conf:
            rep = '\n\\s*#HTTP_TO_HTTPS_START(.|\n){1,300}#HTTP_TO_HTTPS_END'
            conf = re.sub(rep, '', conf)
            rep = '\\s+ssl_certificate\\s+.+;\\s+ssl_certificate_key\\s+.+;'
            conf = re.sub(rep, '', conf)
            rep = '\\s+ssl_protocols\\s+.+;\n'
            conf = re.sub(rep, '', conf)
            rep = '\\s+ssl_ciphers\\s+.+;\n'
            conf = re.sub(rep, '', conf)
            rep = '\\s+ssl_prefer_server_ciphers\\s+.+;\n'
            conf = re.sub(rep, '', conf)
            rep = '\\s+ssl_session_cache\\s+.+;\n'
            conf = re.sub(rep, '', conf)
            rep = '\\s+ssl_session_timeout\\s+.+;\n'
            conf = re.sub(rep, '', conf)
            rep = '\\s+ssl_ecdh_curve\\s+.+;\n'
            conf = re.sub(rep, '', conf)
            rep = '\\s+ssl_session_tickets\\s+.+;\n'
            conf = re.sub(rep, '', conf)
            rep = '\\s+ssl_stapling\\s+.+;\n'
            conf = re.sub(rep, '', conf)
            rep = '\\s+ssl_stapling_verify\\s+.+;\n'
            conf = re.sub(rep, '', conf)
            rep = '\\s+add_header\\s+.+;\n'
            conf = re.sub(rep, '', conf)
            rep = '\\s+add_header\\s+.+;\n'
            conf = re.sub(rep, '', conf)
            rep = '\\s+ssl\\s+on;'
            conf = re.sub(rep, '', conf)
            rep = '\\s+error_page\\s497.+;'
            conf = re.sub(rep, '', conf)
            rep = '\\s+if.+server_port.+\n.+\n\\s+\\s*}'
            conf = re.sub(rep, '', conf)
            rep = '\\s+listen\\s+443.*;'
            conf = re.sub(rep, '', conf)
            rep = '\\s+listen\\s+\\[::\\]:443.*;'
            conf = re.sub(rep, '', conf)
            public.writeFile(file, conf)
        file = self.setupPath + '/panel/vhost/apache/' + siteName + '.conf'
        if not os.path.exists(file):
            file = self.setupPath + '/panel/vhost/apache/node_' + siteName + '.conf'
        if not os.path.exists(file):
            file = self.setupPath + '/panel/vhost/apache/java_' + siteName + '.conf'
        conf = public.readFile(file)
        if conf:
            rep = '\n<VirtualHost \\*\\:443>(.|\n)*<\\/VirtualHost>'
            conf = re.sub(rep, '', conf)
            rep = '\n\\s*#HTTP_TO_HTTPS_START(.|\n){1,250}#HTTP_TO_HTTPS_END'
            conf = re.sub(rep, '', conf)
            rep = 'NameVirtualHost  *:443\n'
            conf = conf.replace(rep, '')
            public.writeFile(file, conf)
        ssl_file = self.setupPath + '/panel/vhost/openlitespeed/detail/ssl/{}.conf'.format(siteName)
        detail_file = self.setupPath + '/panel/vhost/openlitespeed/detail/' + siteName + '.conf'
        force_https = self.setupPath + '/panel/vhost/openlitespeed/redirect/' + siteName
        string = 'rm -f {}/force_https.conf*'.format(force_https)
        public.ExecShell(string)
        detail_conf = public.readFile(detail_file)
        if detail_conf:
            detail_conf = detail_conf.replace('\ninclude ' + ssl_file, '')
            public.writeFile(detail_file, detail_conf)
        public.ExecShell('rm -f {}*'.format(ssl_file))
        self._del_ols_443_domain(siteName)
        partnerOrderId = '/www/server/panel/vhost/cert/' + siteName + '/partnerOrderId'
        if os.path.exists(partnerOrderId):
            public.ExecShell('rm -f ' + partnerOrderId)
        p_file = '/etc/letsencrypt/live/' + siteName + '/partnerOrderId'
        if os.path.exists(p_file):
            public.ExecShell('rm -f ' + p_file)
        public.WriteLog('TYPE_SITE', 'SITE_SSL_CLOSE_SUCCESS', (siteName,))
        public.serviceReload()
        return public.returnMsg(True, 'SITE_SSL_CLOSE_SUCCESS')

    def _del_ols_443_domain(self, sitename):
        if False:
            while True:
                i = 10
        file = '/www/server/panel/vhost/openlitespeed/listen/443.conf'
        conf = public.readFile(file)
        if conf:
            rep = '\n\\s*map\\s*{}'.format(sitename)
            conf = re.sub(rep, '', conf)
            if not 'map ' in conf:
                public.ExecShell('rm -f {}*'.format(file))
                return
            public.writeFile(file, conf)

    def GetSSL(self, get):
        if False:
            return 10
        siteName = get.siteName
        path = os.path.join('/www/server/panel/vhost/cert/', siteName)
        if not os.path.isfile(os.path.join(path, 'fullchain.pem')) and (not os.path.isfile(os.path.join(path, 'privkey.pem'))):
            path = os.path.join('/etc/letsencrypt/live/', siteName)
        type = 0
        if os.path.exists(path + '/README'):
            type = 1
        if os.path.exists(path + '/partnerOrderId'):
            type = 2
        if os.path.exists(path + '/certOrderId'):
            type = 3
        csrpath = path + '/fullchain.pem'
        keypath = path + '/privkey.pem'
        key = public.readFile(keypath)
        csr = public.readFile(csrpath)
        file = self.setupPath + '/panel/vhost/' + public.get_webserver() + '/' + siteName + '.conf'
        if not os.path.exists(file):
            file = self.setupPath + '/panel/vhost/' + public.get_webserver() + '/node_' + siteName + '.conf'
        if not os.path.exists(file):
            file = self.setupPath + '/panel/vhost/' + public.get_webserver() + '/java_' + siteName + '.conf'
        if public.get_webserver() == 'openlitespeed':
            file = self.setupPath + '/panel/vhost/' + public.get_webserver() + '/detail/' + siteName + '.conf'
        conf = public.readFile(file)
        if not conf:
            return public.returnMsg(False, '指定网站配置文件不存在!')
        if public.get_webserver() == 'nginx':
            keyText = 'ssl_certificate'
        elif public.get_webserver() == 'apache':
            keyText = 'SSLCertificateFile'
        else:
            keyText = 'openlitespeed/detail/ssl'
        status = True
        if conf.find(keyText) == -1:
            status = False
            type = -1
        toHttps = self.IsToHttps(siteName)
        id = public.M('sites').where('name=?', (siteName,)).getField('id')
        domains = public.M('domain').where('pid=?', (id,)).field('name').select()
        cert_data = {}
        if csr:
            get.certPath = csrpath
            import panelSSL
            cert_data = panelSSL.panelSSL().GetCertName(get)
        email = public.M('users').where('id=?', (1,)).getField('email')
        if email == '287962566@qq.com':
            email = ''
        index = ''
        auth_type = 'http'
        if status == True:
            if type != 1:
                import acme_v2
                acme = acme_v2.acme_v2()
                index = acme.check_order_exists(csrpath)
                if index:
                    if index.find('/') == -1:
                        auth_type = acme._config['orders'][index]['auth_type']
                    type = 1
            else:
                crontab_file = 'vhost/cert/crontab.json'
                tmp = public.readFile(crontab_file)
                if tmp:
                    crontab_config = json.loads(tmp)
                    if siteName in crontab_config:
                        if 'dnsapi' in crontab_config[siteName]:
                            auth_type = 'dns'
            if os.path.exists(path + '/certOrderId'):
                type = 3
        oid = -1
        if type == 3:
            oid = int(public.readFile(path + '/certOrderId'))
        return {'status': status, 'oid': oid, 'domain': domains, 'key': key, 'csr': csr, 'type': type, 'httpTohttps': toHttps, 'cert_data': cert_data, 'email': email, 'index': index, 'auth_type': auth_type}

    def set_site_status_multiple(self, get):
        if False:
            while True:
                i = 10
        '\n            @name 批量设置网站状态\n            @author zhwen<2020-11-17>\n            @param sites_id "1,2"\n            @param status 0/1\n        '
        sites_id = get.sites_id.split(',')
        sites_name = []
        for site_id in sites_id:
            get.id = site_id
            get.name = public.M('sites').where('id=?', (site_id,)).getField('name')
            sites_name.append(get.name)
            if get.status == '1':
                self.SiteStart(get, multiple=1)
            else:
                self.SiteStop(get, multiple=1)
        public.serviceReload()
        if get.status == '1':
            return {'status': True, 'msg': '开启网站 [ {} ] 成功'.format(','.join(sites_name)), 'error': {}, 'success': sites_name}
        else:
            return {'status': True, 'msg': '停止网站 [ {} ] 成功'.format(','.join(sites_name)), 'error': {}, 'success': sites_name}

    def SiteStart(self, get, multiple=None):
        if False:
            print('Hello World!')
        id = get.id
        Path = self.setupPath + '/stop'
        sitePath = public.M('sites').where('id=?', (id,)).getField('path')
        file = self.setupPath + '/panel/vhost/nginx/' + get.name + '.conf'
        conf = public.readFile(file)
        if conf:
            conf = conf.replace(Path, sitePath)
            conf = conf.replace('#include', 'include')
            public.writeFile(file, conf)
        file = self.setupPath + '/panel/vhost/apache/' + get.name + '.conf'
        conf = public.readFile(file)
        if conf:
            conf = conf.replace(Path, sitePath)
            conf = conf.replace('#IncludeOptional', 'IncludeOptional')
            public.writeFile(file, conf)
        file = self.setupPath + '/panel/vhost/openlitespeed/' + get.name + '.conf'
        conf = public.readFile(file)
        if conf:
            rep = 'vhRoot\\s*{}'.format(Path)
            new_content = 'vhRoot {}'.format(sitePath)
            conf = re.sub(rep, new_content, conf)
            public.writeFile(file, conf)
        public.M('sites').where('id=?', (id,)).setField('status', '1')
        if not multiple:
            public.serviceReload()
        public.WriteLog('TYPE_SITE', 'SITE_START_SUCCESS', (get.name,))
        return public.returnMsg(True, 'SITE_START_SUCCESS')

    def _process_has_run_dir(self, website_name, website_path, stop_path):
        if False:
            return 10
        '\n            @name 当网站存在允许目录时停止网站需要做处理\n            @author zhwen<2020-11-17>\n            @param site_id 1\n            @param names test,baohu\n        '
        conf = public.readFile(self.setupPath + '/panel/vhost/nginx/' + website_name + '.conf')
        if not conf:
            return False
        try:
            really_path = re.search('root\\s+(.*);', conf).group(1)
            tmp = stop_path + '/' + really_path.replace(website_path + '/', '')
            public.ExecShell('mkdir {t} && ln -s {s}/index.html {t}/index.html'.format(t=tmp, s=stop_path))
        except:
            pass

    def SiteStop(self, get, multiple=None):
        if False:
            i = 10
            return i + 15
        path = self.setupPath + '/stop'
        id = get.id
        site_status = public.M('sites').where('id=?', (id,)).getField('status')
        if str(site_status) != '1':
            return public.returnMsg(True, 'SITE_STOP_SUCCESS')
        if not os.path.exists(path):
            os.makedirs(path)
            public.downloadFile('http://{}/stop.html'.format(public.get_url()), path + '/index.html')
        binding = public.M('binding').where('pid=?', (id,)).field('id,pid,domain,path,port,addtime').select()
        for b in binding:
            bpath = path + '/' + b['path']
            if not os.path.exists(bpath):
                public.ExecShell('mkdir -p ' + bpath)
                public.ExecShell('ln -sf ' + path + '/index.html ' + bpath + '/index.html')
        sitePath = public.M('sites').where('id=?', (id,)).getField('path')
        self._process_has_run_dir(get.name, sitePath, path)
        file = self.setupPath + '/panel/vhost/nginx/' + get.name + '.conf'
        conf = public.readFile(file)
        if conf:
            src_path = 'root ' + sitePath
            dst_path = 'root ' + path
            if conf.find(src_path) != -1:
                conf = conf.replace(src_path, dst_path)
            else:
                conf = conf.replace(sitePath, path)
            conf = conf.replace('include', '#include')
            public.writeFile(file, conf)
        file = self.setupPath + '/panel/vhost/apache/' + get.name + '.conf'
        conf = public.readFile(file)
        if conf:
            conf = conf.replace(sitePath, path)
            conf = conf.replace('IncludeOptional', '#IncludeOptional')
            public.writeFile(file, conf)
        file = self.setupPath + '/panel/vhost/openlitespeed/' + get.name + '.conf'
        conf = public.readFile(file)
        if conf:
            rep = 'vhRoot\\s*{}'.format(sitePath)
            new_content = 'vhRoot {}'.format(path)
            conf = re.sub(rep, new_content, conf)
            public.writeFile(file, conf)
        public.M('sites').where('id=?', (id,)).setField('status', '0')
        if not multiple:
            public.serviceReload()
        public.WriteLog('TYPE_SITE', 'SITE_STOP_SUCCESS', (get.name,))
        return public.returnMsg(True, 'SITE_STOP_SUCCESS')

    def GetLimitNet(self, get):
        if False:
            i = 10
            return i + 15
        id = get.id
        siteName = public.M('sites').where('id=?', (id,)).getField('name')
        filename = self.setupPath + '/panel/vhost/nginx/' + siteName + '.conf'
        data = {}
        conf = public.readFile(filename)
        try:
            rep = '\\s+limit_conn\\s+perserver\\s+([0-9]+);'
            tmp = re.search(rep, conf).groups()
            data['perserver'] = int(tmp[0])
            rep = '\\s+limit_conn\\s+perip\\s+([0-9]+);'
            tmp = re.search(rep, conf).groups()
            data['perip'] = int(tmp[0])
            rep = '\\s+limit_rate\\s+([0-9]+)\\w+;'
            tmp = re.search(rep, conf).groups()
            data['limit_rate'] = int(tmp[0])
        except:
            data['perserver'] = 0
            data['perip'] = 0
            data['limit_rate'] = 0
        return data

    def SetLimitNet(self, get):
        if False:
            i = 10
            return i + 15
        if public.get_webserver() != 'nginx':
            return public.returnMsg(False, 'SITE_NETLIMIT_ERR')
        id = get.id
        if int(get.perserver) < 1 or int(get.perip) < 1 or int(get.perip) < 1:
            return public.returnMsg(False, '并发限制，IP限制，流量限制必需大于0')
        perserver = 'limit_conn perserver ' + get.perserver + ';'
        perip = 'limit_conn perip ' + get.perip + ';'
        limit_rate = 'limit_rate ' + get.limit_rate + 'k;'
        siteName = public.M('sites').where('id=?', (id,)).getField('name')
        filename = self.setupPath + '/panel/vhost/nginx/' + siteName + '.conf'
        conf = public.readFile(filename)
        oldLimit = self.setupPath + '/panel/vhost/nginx/limit.conf'
        if os.path.exists(oldLimit):
            os.remove(oldLimit)
        limit = self.setupPath + '/nginx/conf/nginx.conf'
        nginxConf = public.readFile(limit)
        limitConf = 'limit_conn_zone $binary_remote_addr zone=perip:10m;\n\t\tlimit_conn_zone $server_name zone=perserver:10m;'
        nginxConf = nginxConf.replace('#limit_conn_zone $binary_remote_addr zone=perip:10m;', limitConf)
        public.writeFile(limit, nginxConf)
        if conf.find('limit_conn perserver') != -1:
            rep = 'limit_conn\\s+perserver\\s+([0-9]+);'
            conf = re.sub(rep, perserver, conf)
            rep = 'limit_conn\\s+perip\\s+([0-9]+);'
            conf = re.sub(rep, perip, conf)
            rep = 'limit_rate\\s+([0-9]+)\\w+;'
            conf = re.sub(rep, limit_rate, conf)
        else:
            conf = conf.replace('#error_page 404/404.html;', '#error_page 404/404.html;\n    ' + perserver + '\n    ' + perip + '\n    ' + limit_rate)
        import shutil
        shutil.copyfile(filename, self.nginx_conf_bak)
        public.writeFile(filename, conf)
        isError = public.checkWebConfig()
        if isError != True:
            if os.path.exists(self.nginx_conf_bak):
                shutil.copyfile(self.nginx_conf_bak, filename)
            return public.returnMsg(False, 'ERROR: <br><a style="color:red;">' + isError.replace('\n', '<br>') + '</a>')
        public.serviceReload()
        public.WriteLog('TYPE_SITE', 'SITE_NETLIMIT_OPEN_SUCCESS', (siteName,))
        return public.returnMsg(True, 'SET_SUCCESS')

    def CloseLimitNet(self, get):
        if False:
            for i in range(10):
                print('nop')
        id = get.id
        siteName = public.M('sites').where('id=?', (id,)).getField('name')
        filename = self.setupPath + '/panel/vhost/nginx/' + siteName + '.conf'
        conf = public.readFile(filename)
        rep = '\\s+limit_conn\\s+perserver\\s+([0-9]+);'
        conf = re.sub(rep, '', conf)
        rep = '\\s+limit_conn\\s+perip\\s+([0-9]+);'
        conf = re.sub(rep, '', conf)
        rep = '\\s+limit_rate\\s+([0-9]+)\\w+;'
        conf = re.sub(rep, '', conf)
        public.writeFile(filename, conf)
        public.serviceReload()
        public.WriteLog('TYPE_SITE', 'SITE_NETLIMIT_CLOSE_SUCCESS', (siteName,))
        return public.returnMsg(True, 'SITE_NETLIMIT_CLOSE_SUCCESS')

    def Get301Status(self, get):
        if False:
            i = 10
            return i + 15
        siteName = get.siteName
        result = {}
        domains = ''
        id = public.M('sites').where('name=?', (siteName,)).getField('id')
        tmp = public.M('domain').where('pid=?', (id,)).field('name').select()
        node = public.M('sites').where('id=? and project_type=?', (id, 'Node')).count()
        if node:
            node = 'node_'
        else:
            node = ''
        for key in tmp:
            domains += key['name'] + ','
        try:
            if public.get_webserver() == 'nginx':
                conf = public.readFile(self.setupPath + '/panel/vhost/nginx/' + node + siteName + '.conf')
                if conf.find('301-START') == -1:
                    result['domain'] = domains[:-1]
                    result['src'] = ''
                    result['status'] = False
                    result['url'] = 'http://'
                    return result
                rep = 'return\\s+301\\s+((http|https)\\://.+);'
                arr = re.search(rep, conf).groups()[0]
                rep = "'\\^(([\\w-]+\\.)+[\\w-]+)'"
                tmp = re.search(rep, conf)
                src = ''
                if tmp:
                    src = tmp.groups()[0]
            elif public.get_webserver() == 'apache':
                conf = public.readFile(self.setupPath + '/panel/vhost/apache/' + node + siteName + '.conf')
                if conf.find('301-START') == -1:
                    result['domain'] = domains[:-1]
                    result['src'] = ''
                    result['status'] = False
                    result['url'] = 'http://'
                    return result
                rep = 'RewriteRule\\s+.+\\s+((http|https)\\://.+)\\s+\\['
                arr = re.search(rep, conf).groups()[0]
                rep = '\\^((\\w+\\.)+\\w+)\\s+\\[NC'
                tmp = re.search(rep, conf)
                src = ''
                if tmp:
                    src = tmp.groups()[0]
            else:
                conf = public.readFile(self.setupPath + '/panel/vhost/openlitespeed/redirect/{s}/{s}.conf'.format(s=siteName))
                if not conf:
                    result['domain'] = domains[:-1]
                    result['src'] = ''
                    result['status'] = False
                    result['url'] = 'http://'
                    return result
                rep = 'RewriteRule\\s+.+\\s+((http|https)\\://.+)\\s+\\['
                arr = re.search(rep, conf).groups()[0]
                rep = '\\^((\\w+\\.)+\\w+)\\s+\\[NC'
                tmp = re.search(rep, conf)
                src = ''
                if tmp:
                    src = tmp.groups()[0]
        except:
            src = ''
            arr = 'http://'
        result['domain'] = domains[:-1]
        result['src'] = src.replace("'", '')
        result['status'] = True
        if len(arr) < 3:
            result['status'] = False
        result['url'] = arr
        return result

    def Set301Status(self, get):
        if False:
            print('Hello World!')
        siteName = get.siteName
        srcDomain = get.srcDomain
        toDomain = get.toDomain
        type = get.type
        rep = '(http|https)\\://.+'
        if not re.match(rep, toDomain):
            return public.returnMsg(False, 'Url地址不正确!')
        filename = self.setupPath + '/panel/vhost/nginx/' + siteName + '.conf'
        mconf = public.readFile(filename)
        if mconf == False:
            return public.returnMsg(False, '指定配置文件不存在!')
        if mconf:
            if srcDomain == 'all':
                conf301 = '\t#301-START\n\t\treturn 301 ' + toDomain + '$request_uri;\n\t#301-END'
            else:
                conf301 = "\t#301-START\n\t\tif ($host ~ '^" + srcDomain + "'){\n\t\t\treturn 301 " + toDomain + '$request_uri;\n\t\t}\n\t#301-END'
            if type == '1':
                mconf = mconf.replace('#error_page 404/404.html;', '#error_page 404/404.html;\n' + conf301)
            else:
                rep = '\\s+#301-START(.|\n){1,300}#301-END'
                mconf = re.sub(rep, '', mconf)
            public.writeFile(filename, mconf)
        filename = self.setupPath + '/panel/vhost/apache/' + siteName + '.conf'
        mconf = public.readFile(filename)
        if mconf:
            if type == '1':
                if srcDomain == 'all':
                    conf301 = '\n\t#301-START\n\t<IfModule mod_rewrite.c>\n\t\tRewriteEngine on\n\t\tRewriteRule ^(.*)$ ' + toDomain + '$1 [L,R=301]\n\t</IfModule>\n\t#301-END\n'
                else:
                    conf301 = '\n\t#301-START\n\t<IfModule mod_rewrite.c>\n\t\tRewriteEngine on\n\t\tRewriteCond %{HTTP_HOST} ^' + srcDomain + ' [NC]\n\t\tRewriteRule ^(.*) ' + toDomain + '$1 [L,R=301]\n\t</IfModule>\n\t#301-END\n'
                rep = 'combined'
                mconf = mconf.replace(rep, rep + '\n\t' + conf301)
            else:
                rep = '\n\\s+#301-START(.|\n){1,300}#301-END\n*'
                mconf = re.sub(rep, '\n\n', mconf, 1)
                mconf = re.sub(rep, '\n\n', mconf, 1)
            public.writeFile(filename, mconf)
        conf_dir = self.setupPath + '/panel/vhost/openlitespeed/redirect/{}/'.format(siteName)
        if not os.path.exists(conf_dir):
            os.makedirs(conf_dir)
        file = conf_dir + siteName + '.conf'
        if type == '1':
            if srcDomain == 'all':
                conf301 = '#301-START\nRewriteEngine on\nRewriteRule ^(.*)$ ' + toDomain + '$1 [L,R=301]#301-END\n'
            else:
                conf301 = '#301-START\nRewriteEngine on\nRewriteCond %{HTTP_HOST} ^' + srcDomain + ' [NC]\nRewriteRule ^(.*) ' + toDomain + '$1 [L,R=301]\n#301-END\n'
            public.writeFile(file, conf301)
        else:
            public.ExecShell('rm -f {}*'.format(file))
        isError = public.checkWebConfig()
        if isError != True:
            return public.returnMsg(False, 'ERROR: <br><a style="color:red;">' + isError.replace('\n', '<br>') + '</a>')
        public.serviceReload()
        return public.returnMsg(True, 'SUCCESS')

    def GetDirBinding(self, get):
        if False:
            print('Hello World!')
        path = public.M('sites').where('id=?', (get.id,)).getField('path')
        if not os.path.exists(path):
            checks = ['/', '/usr', '/etc']
            if path in checks:
                data = {}
                data['dirs'] = []
                data['binding'] = []
                return data
            public.ExecShell('mkdir -p ' + path)
            public.ExecShell('chmod 755 ' + path)
            public.ExecShell('chown www:www ' + path)
            get.path = path
            self.SetDirUserINI(get)
            siteName = public.M('sites').where('id=?', (get.id,)).getField('name')
            public.WriteLog('网站管理', '站点[' + siteName + '],根目录[' + path + ']不存在,已重新创建!')
        dirnames = []
        for filename in os.listdir(path):
            try:
                json.dumps(filename)
                if sys.version_info[0] == 2:
                    filename = filename.encode('utf-8')
                else:
                    filename.encode('utf-8')
                filePath = path + '/' + filename
                if os.path.islink(filePath):
                    continue
                if os.path.isdir(filePath):
                    dirnames.append(filename)
            except:
                pass
        data = {}
        data['dirs'] = dirnames
        data['binding'] = public.M('binding').where('pid=?', (get.id,)).field('id,pid,domain,path,port,addtime').select()
        return data

    def AddDirBinding(self, get):
        if False:
            return 10
        import shutil
        id = get.id
        tmp = get.domain.split(':')
        domain = tmp[0].lower()
        port = '80'
        version = ''
        if len(tmp) > 1:
            port = tmp[1]
        if not hasattr(get, 'dirName'):
            public.returnMsg(False, 'DIR_EMPTY')
        dirName = get.dirName
        reg = '^([\\w\\-\\*]{1,100}\\.){1,4}(\\w{1,10}|\\w{1,10}\\.\\w{1,10})$'
        if not re.match(reg, domain):
            return public.returnMsg(False, 'SITE_ADD_ERR_DOMAIN')
        siteInfo = public.M('sites').where('id=?', (id,)).field('id,path,name').find()
        webdir = siteInfo['path'] + '/' + dirName
        sql = public.M('binding')
        if sql.where('domain=?', (domain,)).count() > 0:
            return public.returnMsg(False, 'SITE_ADD_ERR_DOMAIN_EXISTS')
        if public.M('domain').where('name=?', (domain,)).count() > 0:
            return public.returnMsg(False, 'SITE_ADD_ERR_DOMAIN_EXISTS')
        filename = self.setupPath + '/panel/vhost/nginx/' + siteInfo['name'] + '.conf'
        nginx_conf_file = filename
        conf = public.readFile(filename)
        if conf:
            listen_ipv6 = ''
            if self.is_ipv6:
                listen_ipv6 = '\n    listen [::]:%s;' % port
            rep = 'enable-php-(\\w{2,5})\\.conf'
            tmp = re.search(rep, conf).groups()
            version = tmp[0]
            bindingConf = '\n#BINDING-%s-START\nserver\n{\n    listen %s;%s\n    server_name %s;\n    index index.php index.html index.htm default.php default.htm default.html;\n    root %s;\n    \n    include enable-php-%s.conf;\n    include %s/panel/vhost/rewrite/%s.conf;\n    #禁止访问的文件或目录\n    location ~ ^/(\\.user.ini|\\.htaccess|\\.git|\\.svn|\\.project|LICENSE|README.md)\n    {\n        return 404;\n    }\n    \n    #一键申请SSL证书验证目录相关设置\n    location ~ \\.well-known{\n        allow all;\n    }\n    \n    location ~ .*\\.(gif|jpg|jpeg|png|bmp|swf)$\n    {\n        expires      30d;\n        error_log /dev/null;\n        access_log /dev/null; \n    }\n    location ~ .*\\.(js|css)?$\n    {\n        expires      12h;\n        error_log /dev/null;\n        access_log /dev/null; \n    }\n    access_log %s.log;\n    error_log  %s.error.log;\n}\n#BINDING-%s-END' % (domain, port, listen_ipv6, domain, webdir, version, self.setupPath, siteInfo['name'], public.GetConfigValue('logs_path') + '/' + siteInfo['name'], public.GetConfigValue('logs_path') + '/' + siteInfo['name'], domain)
            conf += bindingConf
            shutil.copyfile(filename, self.nginx_conf_bak)
            public.writeFile(filename, conf)
        filename = self.setupPath + '/panel/vhost/apache/' + siteInfo['name'] + '.conf'
        conf = public.readFile(filename)
        if conf:
            try:
                try:
                    httpdVersion = public.readFile(self.setupPath + '/apache/version.pl').strip()
                except:
                    httpdVersion = ''
                if httpdVersion == '2.2':
                    phpConfig = ''
                    apaOpt = 'Order allow,deny\n\t\tAllow from all'
                else:
                    version = public.get_php_version_conf(conf)
                    phpConfig = '\n    #PHP     \n    <FilesMatch \\.php>\n        SetHandler "proxy:%s"\n    </FilesMatch>\n    ' % (public.get_php_proxy(version, 'apache'),)
                    apaOpt = 'Require all granted'
                bindingConf = '\n\n#BINDING-%s-START\n<VirtualHost *:%s>\n    ServerAdmin webmaster@example.com\n    DocumentRoot "%s"\n    ServerAlias %s\n    #errorDocument 404 /404.html\n    ErrorLog "%s-error_log"\n    CustomLog "%s-access_log" combined\n    %s\n    \n    #DENY FILES\n     <Files ~ (\\.user.ini|\\.htaccess|\\.git|\\.svn|\\.project|LICENSE|README.md)$>\n       Order allow,deny\n       Deny from all\n    </Files>\n    \n    #PATH\n    <Directory "%s">\n        SetOutputFilter DEFLATE\n        Options FollowSymLinks\n        AllowOverride All\n        %s\n        DirectoryIndex index.php index.html index.htm default.php default.html default.htm\n    </Directory>\n</VirtualHost>\n#BINDING-%s-END' % (domain, port, webdir, domain, public.GetConfigValue('logs_path') + '/' + siteInfo['name'], public.GetConfigValue('logs_path') + '/' + siteInfo['name'], phpConfig, webdir, apaOpt, domain)
                conf += bindingConf
                shutil.copyfile(filename, self.apache_conf_bak)
                public.writeFile(filename, conf)
            except:
                pass
        get.webname = siteInfo['name']
        get.port = port
        self.phpVersion = version
        self.siteName = siteInfo['name']
        self.sitePath = webdir
        listen_file = self.setupPath + '/panel/vhost/openlitespeed/listen/80.conf'
        listen_conf = public.readFile(listen_file)
        if listen_conf:
            rep = 'secure\\s*0'
            map = '\tmap {}_{} {}'.format(siteInfo['name'], dirName, domain)
            listen_conf = re.sub(rep, 'secure 0\n' + map, listen_conf)
            public.writeFile(listen_file, listen_conf)
        self.openlitespeed_add_site(get)
        isError = public.checkWebConfig()
        if isError != True:
            if os.path.exists(self.nginx_conf_bak):
                shutil.copyfile(self.nginx_conf_bak, nginx_conf_file)
            if os.path.exists(self.apache_conf_bak):
                shutil.copyfile(self.apache_conf_bak, filename)
            return public.returnMsg(False, 'ERROR: <br><a style="color:red;">' + isError.replace('\n', '<br>') + '</a>')
        public.M('binding').add('pid,domain,port,path,addtime', (id, domain, port, dirName, public.getDate()))
        public.serviceReload()
        public.WriteLog('TYPE_SITE', 'SITE_BINDING_ADD_SUCCESS', (siteInfo['name'], dirName, domain))
        return public.returnMsg(True, 'ADD_SUCCESS')

    def delete_dir_bind_multiple(self, get):
        if False:
            while True:
                i = 10
        '\n            @name 批量删除网站\n            @author zhwen<2020-11-17>\n            @param bind_ids 1,2,3\n        '
        bind_ids = get.bind_ids.split(',')
        del_successfully = []
        del_failed = {}
        for bind_id in bind_ids:
            get.id = bind_id
            domain = public.M('binding').where('id=?', (get.id,)).getField('domain')
            if not domain:
                continue
            try:
                self.DelDirBinding(get, multiple=1)
                del_successfully.append(domain)
            except:
                del_failed[domain] = '删除时错误了，请再试一次'
                pass
        public.serviceReload()
        return {'status': True, 'msg': '删除 [ {} ] 子目录绑定成功'.format(','.join(del_successfully)), 'error': del_failed, 'success': del_successfully}

    def DelDirBinding(self, get, multiple=None):
        if False:
            while True:
                i = 10
        id = get.id
        binding = public.M('binding').where('id=?', (id,)).field('id,pid,domain,path').find()
        siteName = public.M('sites').where('id=?', (binding['pid'],)).getField('name')
        filename = self.setupPath + '/panel/vhost/nginx/' + siteName + '.conf'
        conf = public.readFile(filename)
        if conf:
            rep = '\\s*.+BINDING-' + binding['domain'] + '-START(.|\n)+BINDING-' + binding['domain'] + '-END'
            conf = re.sub(rep, '', conf)
            public.writeFile(filename, conf)
        filename = self.setupPath + '/panel/vhost/apache/' + siteName + '.conf'
        conf = public.readFile(filename)
        if conf:
            rep = '\\s*.+BINDING-' + binding['domain'] + '-START(.|\n)+BINDING-' + binding['domain'] + '-END'
            conf = re.sub(rep, '', conf)
            public.writeFile(filename, conf)
        filename = self.setupPath + '/panel/vhost/openlitespeed/' + siteName + '.conf'
        conf = public.readFile(filename)
        rep = '#SUBDIR\\s*{s}_{d}\\s*START(\n|.)+#SUBDIR\\s*{s}_{d}\\s*END'.format(s=siteName, d=binding['path'])
        if conf:
            conf = re.sub(rep, '', conf)
            public.writeFile(filename, conf)
        get.webname = siteName
        get.domain = binding['domain']
        self._del_ols_domain(get)
        listen_file = self.setupPath + '/panel/vhost/openlitespeed/listen/80.conf'
        listen_conf = public.readFile(listen_file)
        if listen_conf:
            map_reg = '\\s*map\\s*{}_{}.*'.format(siteName, binding['path'])
            listen_conf = re.sub(map_reg, '', listen_conf)
            public.writeFile(listen_file, listen_conf)
        detail_file = '{}/panel/vhost/openlitespeed/detail/{}_{}.conf'.format(self.setupPath, siteName, binding['path'])
        public.ExecShell('rm -f {}*'.format(detail_file))
        public.M('binding').where('id=?', (id,)).delete()
        filename = self.setupPath + '/panel/vhost/rewrite/' + siteName + '_' + binding['path'] + '.conf'
        if os.path.exists(filename):
            public.ExecShell('rm -rf %s' % filename)
        if not multiple:
            public.serviceReload()
        public.WriteLog('TYPE_SITE', 'SITE_BINDING_DEL_SUCCESS', (siteName, binding['path']))
        return public.returnMsg(True, 'DEL_SUCCESS')

    def GetDirRewrite(self, get):
        if False:
            i = 10
            return i + 15
        id = get.id
        find = public.M('binding').where('id=?', (id,)).field('id,pid,domain,path').find()
        site = public.M('sites').where('id=?', (find['pid'],)).field('id,name,path').find()
        if public.get_webserver() != 'nginx':
            filename = site['path'] + '/' + find['path'] + '/.htaccess'
        else:
            filename = self.setupPath + '/panel/vhost/rewrite/' + site['name'] + '_' + find['path'] + '.conf'
        if hasattr(get, 'add'):
            public.writeFile(filename, '')
            if public.get_webserver() == 'nginx':
                file = self.setupPath + '/panel/vhost/nginx/' + site['name'] + '.conf'
                conf = public.readFile(file)
                domain = find['domain']
                rep = '\n#BINDING-' + domain + '-START(.|\n)+BINDING-' + domain + '-END'
                tmp = re.search(rep, conf).group()
                dirConf = tmp.replace('rewrite/' + site['name'] + '.conf;', 'rewrite/' + site['name'] + '_' + find['path'] + '.conf;')
                conf = conf.replace(tmp, dirConf)
                public.writeFile(file, conf)
        data = {}
        data['status'] = False
        if os.path.exists(filename):
            data['status'] = True
            data['data'] = public.readFile(filename)
            data['rlist'] = ['0.当前']
            webserver = public.get_webserver()
            if webserver == 'openlitespeed':
                webserver = 'apache'
            for ds in os.listdir('rewrite/' + webserver):
                if ds == 'list.txt':
                    continue
                data['rlist'].append(ds[0:len(ds) - 5])
            data['filename'] = filename
        return data

    def GetIndex(self, get):
        if False:
            for i in range(10):
                print('nop')
        id = get.id
        Name = public.M('sites').where('id=?', (id,)).getField('name')
        file = self.setupPath + '/panel/vhost/' + public.get_webserver() + '/' + Name + '.conf'
        if public.get_webserver() == 'openlitespeed':
            file = self.setupPath + '/panel/vhost/' + public.get_webserver() + '/detail/' + Name + '.conf'
        conf = public.readFile(file)
        if conf == False:
            return public.returnMsg(False, '指定网站配置文件不存在!')
        if public.get_webserver() == 'nginx':
            rep = '\\s+index\\s+(.+);'
        elif public.get_webserver() == 'apache':
            rep = 'DirectoryIndex\\s+(.+)\n'
        else:
            rep = 'indexFiles\\s+(.+)\n'
        if re.search(rep, conf):
            tmp = re.search(rep, conf).groups()
            if public.get_webserver() == 'openlitespeed':
                return tmp[0]
            return tmp[0].replace(' ', ',')
        return public.returnMsg(False, '获取失败,配置文件中不存在默认文档')

    def SetIndex(self, get):
        if False:
            for i in range(10):
                print('nop')
        id = get.id
        if get.Index.find('.') == -1:
            return public.returnMsg(False, 'SITE_INDEX_ERR_FORMAT')
        Index = get.Index.replace(' ', '')
        Index = get.Index.replace(',,', ',')
        if len(Index) < 3:
            return public.returnMsg(False, 'SITE_INDEX_ERR_EMPTY')
        Name = public.M('sites').where('id=?', (id,)).getField('name')
        Index_L = Index.replace(',', ' ')
        file = self.setupPath + '/panel/vhost/nginx/' + Name + '.conf'
        conf = public.readFile(file)
        if conf:
            rep = '\\s+index\\s+.+;'
            conf = re.sub(rep, '\n\tindex ' + Index_L + ';', conf)
            public.writeFile(file, conf)
        file = self.setupPath + '/panel/vhost/apache/' + Name + '.conf'
        conf = public.readFile(file)
        if conf:
            rep = 'DirectoryIndex\\s+.+\n'
            conf = re.sub(rep, 'DirectoryIndex ' + Index_L + '\n', conf)
            public.writeFile(file, conf)
        file = self.setupPath + '/panel/vhost/openlitespeed/detail/' + Name + '.conf'
        conf = public.readFile(file)
        if conf:
            rep = 'indexFiles\\s+.+\n'
            Index = Index.split(',')
            Index = [i for i in Index if i]
            Index = ','.join(Index)
            conf = re.sub(rep, 'indexFiles ' + Index + '\n', conf)
            public.writeFile(file, conf)
        public.serviceReload()
        public.WriteLog('TYPE_SITE', 'SITE_INDEX_SUCCESS', (Name, Index_L))
        return public.returnMsg(True, 'SET_SUCCESS')

    def SetPath(self, get):
        if False:
            for i in range(10):
                print('nop')
        id = get.id
        Path = self.GetPath(get.path)
        if Path == '' or id == '0':
            return public.returnMsg(False, 'DIR_EMPTY')
        if not self.__check_site_path(Path):
            return public.returnMsg(False, 'PATH_ERROR')
        if not public.check_site_path(Path):
            (a, c) = public.get_sys_path()
            return public.returnMsg(False, '请不要将网站根目录设置到以下关键目录中: <br>{}'.format('<br>'.join(a + c)))
        SiteFind = public.M('sites').where('id=?', (id,)).field('path,name').find()
        if SiteFind['path'] == Path:
            return public.returnMsg(False, 'SITE_PATH_ERR_RE')
        Name = SiteFind['name']
        file = self.setupPath + '/panel/vhost/nginx/' + Name + '.conf'
        conf = public.readFile(file)
        if conf:
            conf = conf.replace(SiteFind['path'], Path)
            public.writeFile(file, conf)
        file = self.setupPath + '/panel/vhost/apache/' + Name + '.conf'
        conf = public.readFile(file)
        if conf:
            rep = 'DocumentRoot\\s+.+\n'
            conf = re.sub(rep, 'DocumentRoot "' + Path + '"\n', conf)
            rep = '<Directory\\s+.+\n'
            conf = re.sub(rep, '<Directory "' + Path + '">\n', conf)
            public.writeFile(file, conf)
        file = self.setupPath + '/panel/vhost/openlitespeed/' + Name + '.conf'
        conf = public.readFile(file)
        if conf:
            reg = 'vhRoot.*'
            conf = re.sub(reg, 'vhRoot ' + Path, conf)
            public.writeFile(file, conf)
        userIni = Path + '/.user.ini'
        if os.path.exists(userIni):
            public.ExecShell('chattr -i ' + userIni)
        public.writeFile(userIni, 'open_basedir=' + Path + '/:/tmp/')
        public.ExecShell('chmod 644 ' + userIni)
        public.ExecShell('chown root:root ' + userIni)
        public.ExecShell('chattr +i ' + userIni)
        public.set_site_open_basedir_nginx(Name)
        public.serviceReload()
        public.M('sites').where('id=?', (id,)).setField('path', Path)
        public.WriteLog('TYPE_SITE', 'SITE_PATH_SUCCESS', (Name,))
        return public.returnMsg(True, 'SET_SUCCESS')

    def GetPHPVersion(self, get):
        if False:
            return 10
        phpVersions = public.get_php_versions()
        phpVersions.insert(0, 'other')
        phpVersions.insert(0, '00')
        httpdVersion = ''
        filename = self.setupPath + '/apache/version.pl'
        if os.path.exists(filename):
            httpdVersion = public.readFile(filename).strip()
        if httpdVersion == '2.2':
            phpVersions = ('00', '52', '53', '54')
        if httpdVersion == '2.4':
            if '52' in phpVersions:
                phpVersions.remove('52')
        if os.path.exists('/www/server/nginx/sbin/nginx'):
            cfile = '/www/server/nginx/conf/enable-php-00.conf'
            if not os.path.exists(cfile):
                public.writeFile(cfile, '')
        s_type = getattr(get, 's_type', 0)
        data = []
        for val in phpVersions:
            tmp = {}
            checkPath = self.setupPath + '/php/' + val + '/bin/php'
            if val in ['00', 'other']:
                checkPath = '/etc/init.d/bt'
            if httpdVersion == '2.2':
                checkPath = self.setupPath + '/php/' + val + '/libphp5.so'
            if os.path.exists(checkPath):
                tmp['version'] = val
                tmp['name'] = 'PHP-' + val
                if val == '00':
                    tmp['name'] = '纯静态'
                if val == 'other':
                    if s_type:
                        tmp['name'] = '自定义'
                    else:
                        continue
                data.append(tmp)
        return data

    def GetSitePHPVersion(self, get):
        if False:
            for i in range(10):
                print('nop')
        try:
            siteName = get.siteName
            data = {}
            data['phpversion'] = public.get_site_php_version(siteName)
            conf = public.readFile(self.setupPath + '/panel/vhost/' + public.get_webserver() + '/' + siteName + '.conf')
            data['tomcat'] = conf.find('#TOMCAT-START')
            data['tomcatversion'] = public.readFile(self.setupPath + '/tomcat/version.pl')
            data['nodejsversion'] = public.readFile(self.setupPath + '/node.js/version.pl')
            data['php_other'] = ''
            if data['phpversion'] == 'other':
                other_file = '/www/server/panel/vhost/other_php/{}/enable-php-other.conf'.format(siteName)
                if os.path.exists(other_file):
                    conf = public.readFile(other_file)
                    data['php_other'] = re.findall('fastcgi_pass\\s+(.+);', conf)[0]
            return data
        except:
            return public.returnMsg(False, 'SITE_PHPVERSION_ERR_A22,{}'.format(public.get_error_info()))

    def set_site_php_version_multiple(self, get):
        if False:
            return 10
        '\n            @name 批量设置PHP版本\n            @author zhwen<2020-11-17>\n            @param sites_id "1,2"\n            @param version 52...74\n        '
        sites_id = get.sites_id.split(',')
        set_phpv_successfully = []
        set_phpv_failed = {}
        for site_id in sites_id:
            get.id = site_id
            get.siteName = public.M('sites').where('id=?', (site_id,)).getField('name')
            if not get.siteName:
                continue
            try:
                result = self.SetPHPVersion(get, multiple=1)
                if not result['status']:
                    set_phpv_failed[get.siteName] = result['msg']
                    continue
                set_phpv_successfully.append(get.siteName)
            except:
                set_phpv_failed[get.siteName] = '设置时错误了，请再试一次'
                pass
        public.serviceReload()
        return {'status': True, 'msg': '设置网站 [ {} ] PHP版本成功'.format(','.join(set_phpv_successfully)), 'error': set_phpv_failed, 'success': set_phpv_successfully}

    def SetPHPVersion(self, get, multiple=None):
        if False:
            while True:
                i = 10
        siteName = get.siteName
        version = get.version
        if version == 'other' and (not public.get_webserver() in ['nginx', 'tengine']):
            return public.returnMsg(False, '自定义PHP配置只支持Nginx')
        try:
            file = self.setupPath + '/panel/vhost/nginx/' + siteName + '.conf'
            conf = public.readFile(file)
            if conf:
                other_path = '/www/server/panel/vhost/other_php/{}'.format(siteName)
                if not os.path.exists(other_path):
                    os.makedirs(other_path)
                other_rep = '{}/enable-php-other.conf'.format(other_path)
                if version == 'other':
                    dst = other_rep
                    get.other = get.other.strip()
                    if not get.other:
                        return public.returnMsg(False, '自定义版本时PHP连接配置不能为空!')
                    if not re.match('^(\\d+\\.\\d+\\.\\d+\\.\\d+:\\d+|unix:[\\w/\\.-]+)$', get.other):
                        return public.returnMsg(False, 'PHP连接配置格式不正确，请参考示例!')
                    other_tmp = get.other.split(':')
                    if other_tmp[0] == 'unix':
                        if not os.path.exists(other_tmp[1]):
                            return public.returnMsg(False, '指定unix套接字[{}]不存在，请核实!'.format(other_tmp[1]))
                    elif not public.check_tcp(other_tmp[0], int(other_tmp[1])):
                        return public.returnMsg(False, '无法连接[{}],请排查本机是否可连接目标服务器'.format(get.other))
                    other_conf = 'location ~ [^/]\\.php(/|$)\n{{\n    try_files $uri =404;\n    fastcgi_pass  {};\n    fastcgi_index index.php;\n    include fastcgi.conf;\n    include pathinfo.conf;\n}}'.format(get.other)
                    public.writeFile(other_rep, other_conf)
                    conf = conf.replace(other_rep, dst)
                    rep = 'include\\s+enable-php-(\\w{2,5})\\.conf'
                    tmp = re.search(rep, conf)
                    if tmp:
                        conf = conf.replace(tmp.group(), 'include ' + dst)
                else:
                    dst = 'enable-php-' + version + '.conf'
                    conf = conf.replace(other_rep, dst)
                    rep = 'enable-php-(\\w{2,5})\\.conf'
                    tmp = re.search(rep, conf)
                    if tmp:
                        conf = conf.replace(tmp.group(), dst)
                public.writeFile(file, conf)
                try:
                    import site_dir_auth
                    site_dir_auth_module = site_dir_auth.SiteDirAuth()
                    auth_list = site_dir_auth_module.get_dir_auth(get)
                    if auth_list:
                        for i in auth_list[siteName]:
                            auth_name = i['name']
                            auth_file = '{setup_path}/panel/vhost/nginx/dir_auth/{site_name}/{auth_name}.conf'.format(setup_path=self.setupPath, site_name=siteName, auth_name=auth_name)
                            if os.path.exists(auth_file):
                                site_dir_auth_module.change_dir_auth_file_nginx_phpver(siteName, version, auth_name)
                except:
                    pass
            file = self.setupPath + '/panel/vhost/apache/' + siteName + '.conf'
            conf = public.readFile(file)
            if conf and version != 'other':
                rep = '(unix:/tmp/php-cgi-(\\w{2,5})\\.sock\\|fcgi://localhost|fcgi://127.0.0.1:\\d+)'
                tmp = re.search(rep, conf).group()
                conf = conf.replace(tmp, public.get_php_proxy(version, 'apache'))
                public.writeFile(file, conf)
            if version != 'other':
                file = self.setupPath + '/panel/vhost/openlitespeed/detail/' + siteName + '.conf'
                conf = public.readFile(file)
                if conf:
                    rep = 'lsphp\\d+'
                    tmp = re.search(rep, conf)
                    if tmp:
                        conf = conf.replace(tmp.group(), 'lsphp' + version)
                        public.writeFile(file, conf)
            if not multiple:
                public.serviceReload()
            public.WriteLog('TYPE_SITE', 'SITE_PHPVERSION_SUCCESS', (siteName, version))
            return public.returnMsg(True, 'SITE_PHPVERSION_SUCCESS', (siteName, version))
        except:
            return public.get_error_info()
            return public.returnMsg(False, '设置失败，没有在网站配置文件中找到enable-php-xx相关配置项!')

    def GetDirUserINI(self, get):
        if False:
            i = 10
            return i + 15
        path = get.path + self.GetRunPath(get)
        if not path:
            return public.returnMsg(False, '获取目录失败')
        id = get.id
        get.name = public.M('sites').where('id=?', (id,)).getField('name')
        data = {}
        data['logs'] = self.GetLogsStatus(get)
        data['userini'] = False
        user_ini_file = path + '/.user.ini'
        user_ini_conf = public.readFile(user_ini_file)
        if user_ini_conf and 'open_basedir' in user_ini_conf:
            data['userini'] = True
        data['runPath'] = self.GetSiteRunPath(get)
        data['pass'] = self.GetHasPwd(get)
        return data

    def DelUserInI(self, path, up=0):
        if False:
            i = 10
            return i + 15
        useriniPath = path + '/.user.ini'
        if os.path.exists(useriniPath):
            public.ExecShell('chattr -i ' + useriniPath)
            try:
                os.remove(useriniPath)
            except:
                pass
        for p1 in os.listdir(path):
            try:
                npath = path + '/' + p1
                if not os.path.isdir(npath):
                    continue
                useriniPath = npath + '/.user.ini'
                if os.path.exists(useriniPath):
                    public.ExecShell('chattr -i ' + useriniPath)
                    os.remove(useriniPath)
                if up < 3:
                    self.DelUserInI(npath, up + 1)
            except:
                continue
        return True

    def SetDirUserINI(self, get):
        if False:
            i = 10
            return i + 15
        path = get.path
        runPath = self.GetRunPath(get)
        filename = path + runPath + '/.user.ini'
        siteName = public.M('sites').where('path=?', (get.path,)).getField('name')
        conf = public.readFile(filename)
        try:
            self._set_ols_open_basedir(get)
            public.ExecShell('chattr -i ' + filename)
            if conf and 'open_basedir' in conf:
                rep = '\n*open_basedir.*'
                conf = re.sub(rep, '', conf)
                if not conf:
                    os.remove(filename)
                else:
                    public.writeFile(filename, conf)
                    public.ExecShell('chattr +i ' + filename)
                public.set_site_open_basedir_nginx(siteName)
                return public.returnMsg(True, 'SITE_BASEDIR_CLOSE_SUCCESS')
            if conf and 'session.save_path' in conf:
                rep = 'session.save_path\\s*=\\s*(.*)'
                s_path = re.search(rep, conf).groups(1)[0]
                public.writeFile(filename, conf + '\nopen_basedir={}/:/tmp/:{}'.format(path, s_path))
            else:
                public.writeFile(filename, 'open_basedir={}/:/tmp/'.format(path))
            public.ExecShell('chattr +i ' + filename)
            public.set_site_open_basedir_nginx(siteName)
            public.serviceReload()
            return public.returnMsg(True, 'SITE_BASEDIR_OPEN_SUCCESS')
        except Exception as e:
            public.ExecShell('chattr +i ' + filename)
            return str(e)

    def _set_ols_open_basedir(self, get):
        if False:
            print('Hello World!')
        try:
            sitename = public.M('sites').where('id=?', (get.id,)).getField('name')
            f = '/www/server/panel/vhost/openlitespeed/detail/{}.conf'.format(sitename)
            c = public.readFile(f)
            if not c:
                return False
            if f:
                rep = '\nphp_admin_value\\s*open_basedir.*'
                result = re.search(rep, c)
                s = 'on'
                if not result:
                    s = 'off'
                    rep = '\n#php_admin_value\\s*open_basedir.*'
                    result = re.search(rep, c)
                result = result.group()
                if s == 'on':
                    c = re.sub(rep, '\n#' + result[1:], c)
                else:
                    result = result.replace('#', '')
                    c = re.sub(rep, result, c)
                public.writeFile(f, c)
        except:
            pass

    def __read_config(self, path):
        if False:
            while True:
                i = 10
        if not os.path.exists(path):
            public.writeFile(path, '[]')
        upBody = public.readFile(path)
        if not upBody:
            upBody = '[]'
        return json.loads(upBody)

    def __write_config(self, path, data):
        if False:
            for i in range(10):
                print('nop')
        return public.writeFile(path, json.dumps(data))

    def GetProxyDetals(self, get):
        if False:
            i = 10
            return i + 15
        proxyUrl = self.__read_config(self.__proxyfile)
        sitename = get.sitename
        proxyname = get.proxyname
        for i in proxyUrl:
            if i['proxyname'] == proxyname and i['sitename'] == sitename:
                return i

    def GetProxyList(self, get):
        if False:
            i = 10
            return i + 15
        n = 0
        for w in ['nginx', 'apache']:
            conf_path = '%s/panel/vhost/%s/%s.conf' % (self.setupPath, w, get.sitename)
            old_conf = ''
            if os.path.exists(conf_path):
                old_conf = public.readFile(conf_path)
            rep = '(#PROXY-START(\n|.)+#PROXY-END)'
            url_rep = 'proxy_pass (.*);|ProxyPass\\s/\\s(.*)|Host\\s(.*);'
            host_rep = 'Host\\s(.*);'
            if re.search(rep, old_conf):
                if w == 'nginx':
                    get.todomain = str(re.search(host_rep, old_conf).group(1))
                    get.proxysite = str(re.search(url_rep, old_conf).group(1))
                else:
                    get.todomain = ''
                    get.proxysite = str(re.search(url_rep, old_conf).group(2))
                get.proxyname = '旧代理'
                get.type = 1
                get.proxydir = '/'
                get.advanced = 0
                get.cachetime = 1
                get.cache = 0
                get.subfilter = '[{"sub1":"","sub2":""},{"sub1":"","sub2":""},{"sub1":"","sub2":""}]'
                public.ExecShell('cp %s %s_bak' % (conf_path, conf_path))
                conf = re.sub(rep, '', old_conf)
                public.writeFile(conf_path, conf)
                if n == 0:
                    self.CreateProxy(get)
                n += 1
            if n == '1':
                public.serviceReload()
        proxyUrl = self.__read_config(self.__proxyfile)
        sitename = get.sitename
        proxylist = []
        for i in proxyUrl:
            if i['sitename'] == sitename:
                proxylist.append(i)
        return proxylist

    def del_proxy_multiple(self, get):
        if False:
            while True:
                i = 10
        '\n            @name 批量网站到期时间\n            @author zhwen<2020-11-20>\n            @param site_id 1\n            @param proxynames ces,aaa\n        '
        proxynames = get.proxynames.split(',')
        del_successfully = []
        del_failed = {}
        get.sitename = public.M('sites').where('id=?', (get.site_id,)).getField('name')
        for proxyname in proxynames:
            if not proxyname:
                continue
            get.proxyname = proxyname
            try:
                resule = self.RemoveProxy(get, multiple=1)
                if not resule['status']:
                    del_failed[proxyname] = resule['msg']
                del_successfully.append(proxyname)
            except:
                del_failed[proxyname] = '删除时错误，请再试一次'
                pass
        return {'status': True, 'msg': '删除反向代理 [ {} ] 成功'.format(','.join(del_failed)), 'error': del_failed, 'success': del_successfully}

    def RemoveProxy(self, get, multiple=None):
        if False:
            for i in range(10):
                print('nop')
        conf = self.__read_config(self.__proxyfile)
        sitename = get.sitename
        proxyname = get.proxyname
        for i in range(len(conf)):
            c_sitename = conf[i]['sitename']
            c_proxyname = conf[i]['proxyname']
            if c_sitename == sitename and c_proxyname == proxyname:
                proxyname_md5 = self.__calc_md5(c_proxyname)
                for w in ['apache', 'nginx', 'openlitespeed']:
                    p = '{sp}/panel/vhost/{w}/proxy/{s}/{m}_{s}.conf*'.format(sp=self.setupPath, w=w, s=c_sitename, m=proxyname_md5)
                    public.ExecShell('rm -f {}'.format(p))
                p = '{sp}/panel/vhost/openlitespeed/proxy/{s}/urlrewrite/{m}_{s}.conf*'.format(sp=self.setupPath, m=proxyname_md5, s=get.sitename)
                public.ExecShell('rm -f {}'.format(p))
                del conf[i]
                self.__write_config(self.__proxyfile, conf)
                self.SetNginx(get)
                self.SetApache(get.sitename)
                if not multiple:
                    public.serviceReload()
                return public.returnMsg(True, '删除成功')

    def __check_even(self, get, action=''):
        if False:
            while True:
                i = 10
        conf_data = self.__read_config(self.__proxyfile)
        for i in conf_data:
            if i['sitename'] == get.sitename:
                if action == 'create':
                    if i['proxydir'] == get.proxydir or i['proxyname'] == get.proxyname:
                        return i
                elif i['proxyname'] != get.proxyname and i['proxydir'] == get.proxydir:
                    return i

    def __check_proxy_even(self, get, action=''):
        if False:
            for i in range(10):
                print('nop')
        conf_data = self.__read_config(self.__proxyfile)
        n = 0
        if action == '':
            for i in conf_data:
                if i['sitename'] == get.sitename:
                    n += 1
            if n == 1:
                return
        for i in conf_data:
            if i['sitename'] == get.sitename:
                if i['advanced'] != int(get.advanced):
                    return i

    def __calc_md5(self, proxyname):
        if False:
            print('Hello World!')
        md5 = hashlib.md5()
        md5.update(proxyname.encode('utf-8'))
        return md5.hexdigest()

    def __CheckUrl(self, get):
        if False:
            for i in range(10):
                print('nop')
        sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sk.settimeout(5)
        rep = '(https?)://([\\w\\.\\-]+):?([\\d]+)?'
        h = re.search(rep, get.proxysite).group(1)
        d = re.search(rep, get.proxysite).group(2)
        try:
            p = re.search(rep, get.proxysite).group(3)
        except:
            p = ''
        try:
            if p:
                sk.connect((d, int(p)))
            elif h == 'http':
                sk.connect((d, 80))
            else:
                sk.connect((d, 443))
        except:
            return public.returnMsg(False, '目标URL无法访问')

    def __CheckStart(self, get, action=''):
        if False:
            return 10
        isError = public.checkWebConfig()
        if isError != True:
            return public.returnMsg(False, '配置文件出错请先排查配置')
        if action == 'create':
            if sys.version_info.major < 3:
                if len(get.proxyname) < 3 or len(get.proxyname) > 40:
                    return public.returnMsg(False, '名称必须大于3小于40个字符串')
            elif len(get.proxyname.encode('utf-8')) < 3 or len(get.proxyname.encode('utf-8')) > 40:
                return public.returnMsg(False, '名称必须大于3小于40个字符串')
        if self.__check_even(get, action):
            return public.returnMsg(False, '指定反向代理名称或代理文件夹已存在')
        if self.__check_proxy_even(get, action):
            return public.returnMsg(False, '不能同时设置目录代理和全局代理')
        if get.cachetime:
            try:
                int(get.cachetime)
            except:
                return public.returnMsg(False, '请输入数字')
        rep = 'http(s)?\\:\\/\\/'
        tod = '[a-zA-Z]+$'
        repte = '[\\?\\=\\[\\]\\)\\(\\*\\&\\^\\%\\$\\#\\@\\!\\~\\`{\\}\\>\\<\\,\',"]+'
        if re.search(repte, get.proxydir):
            return public.returnMsg(False, '代理目录不能有以下特殊符号 ?,=,[,],),(,*,&,^,%,$,#,@,!,~,`,{,},>,<,\\,\',"]')
        if get.todomain:
            if re.search('[\\}\\{\\#\\;"\']+', get.todomain):
                return public.returnMsg(False, '发送域名格式错误:' + get.todomain + '<br>不能存在以下特殊字符【 }  { # ; " \' 】 ')
        if public.get_webserver() != 'openlitespeed' and (not get.todomain):
            get.todomain = '$host'
        if not re.match(rep, get.proxysite):
            return public.returnMsg(False, '域名格式错误 ' + get.proxysite)
        if re.search(repte, get.proxysite):
            return public.returnMsg(False, '目标URL不能有以下特殊符号 ?,=,[,],),(,*,&,^,%,$,#,@,!,~,`,{,},>,<,\\,\',"]')
        subfilter = json.loads(get.subfilter)
        if subfilter:
            for s in subfilter:
                if not s['sub1']:
                    if s['sub2']:
                        return public.returnMsg(False, '请输入被替换的内容')
                elif s['sub1'] == s['sub2']:
                    return public.returnMsg(False, '替换内容与被替换内容不能一致')

    def SetNginx(self, get):
        if False:
            while True:
                i = 10
        ng_proxyfile = '%s/panel/vhost/nginx/proxy/%s/*.conf' % (self.setupPath, get.sitename)
        ng_file = self.setupPath + '/panel/vhost/nginx/' + get.sitename + '.conf'
        p_conf = self.__read_config(self.__proxyfile)
        cureCache = ''
        if public.get_webserver() == 'nginx':
            shutil.copyfile(ng_file, '/tmp/ng_file_bk.conf')
        cureCache += '\n    location ~ /purge(/.*) {\n        proxy_cache_purge cache_one $host$1$is_args$args;\n        #access_log  /www/wwwlogs/%s_purge_cache.log;\n    }' % get.sitename
        if os.path.exists(ng_file):
            self.CheckProxy(get)
            ng_conf = public.readFile(ng_file)
            if not p_conf:
                rep = '#清理缓存规则[\\w\\s\\~\\/\\(\\)\\.\\*\\{\\}\\;\\$\n\\#]+.*\n.*'
                ng_conf = re.sub(rep, '', ng_conf)
                oldconf = 'location ~ .*\\.(gif|jpg|jpeg|png|bmp|swf)$\n    {\n        expires      30d;\n        error_log /dev/null;\n        access_log /dev/null;\n    }\n    location ~ .*\\.(js|css)?$\n    {\n        expires      12h;\n        error_log /dev/null;\n        access_log /dev/null;\n    }'
                if '(gif|jpg|jpeg|png|bmp|swf)$' not in ng_conf:
                    ng_conf = re.sub('access_log\\s*/www', oldconf + '\n\taccess_log  /www', ng_conf)
                public.writeFile(ng_file, ng_conf)
                return
            sitenamelist = []
            for i in p_conf:
                sitenamelist.append(i['sitename'])
            if get.sitename in sitenamelist:
                rep = 'include.*\\/proxy\\/.*\\*.conf;'
                if not re.search(rep, ng_conf):
                    rep = 'location.+\\(gif[\\w\\|\\$\\(\\)\n\\{\\}\\s\\;\\/\\~\\.\\*\\\\\\?]+access_log\\s+/'
                    ng_conf = re.sub(rep, 'access_log  /', ng_conf)
                    ng_conf = ng_conf.replace('include enable-php-', '#清理缓存规则\n' + cureCache + '\n\t#引用反向代理规则，注释后配置的反向代理将无效\n\t' + 'include ' + ng_proxyfile + ';\n\n\tinclude enable-php-')
                    public.writeFile(ng_file, ng_conf)
            else:
                rep = '#清理缓存规则[\\w\\s\\~\\/\\(\\)\\.\\*\\{\\}\\;\\$\n\\#]+.*\n.*'
                ng_conf = re.sub(rep, '', ng_conf)
                oldconf = 'location ~ .*\\.(gif|jpg|jpeg|png|bmp|swf)$\n    {\n        expires      30d;\n        error_log /dev/null;\n        access_log /dev/null;\n    }\n    location ~ .*\\.(js|css)?$\n    {\n        expires      12h;\n        error_log /dev/null;\n        access_log /dev/null;\n    }'
                if '(gif|jpg|jpeg|png|bmp|swf)$' not in ng_conf:
                    ng_conf = re.sub('access_log\\s*/www', oldconf + '\n\taccess_log  /www', ng_conf)
                public.writeFile(ng_file, ng_conf)

    def SetApache(self, sitename):
        if False:
            i = 10
            return i + 15
        ap_proxyfile = '%s/panel/vhost/apache/proxy/%s/*.conf' % (self.setupPath, sitename)
        ap_file = self.setupPath + '/panel/vhost/apache/' + sitename + '.conf'
        p_conf = public.readFile(self.__proxyfile)
        if public.get_webserver() == 'apache':
            shutil.copyfile(ap_file, '/tmp/ap_file_bk.conf')
        if os.path.exists(ap_file):
            ap_conf = public.readFile(ap_file)
            if p_conf == '[]':
                rep = '\n*#引用反向代理规则，注释后配置的反向代理将无效\n+\\s+IncludeOptiona[\\s\\w\\/\\.\\*]+'
                ap_conf = re.sub(rep, '', ap_conf)
                public.writeFile(ap_file, ap_conf)
                return
            if sitename in p_conf:
                rep = 'combined(\n|.)+IncludeOptional.*\\/proxy\\/.*conf'
                rep1 = 'combined'
                if not re.search(rep, ap_conf):
                    ap_conf = ap_conf.replace(rep1, rep1 + '\n\t#引用反向代理规则，注释后配置的反向代理将无效\n\t' + '\n\tIncludeOptional ' + ap_proxyfile)
                    public.writeFile(ap_file, ap_conf)
            else:
                rep = '\n*#引用反向代理规则，注释后配置的反向代理将无效\n+\\s+IncludeOptiona[\\s\\w\\/\\.\\*]+'
                ap_conf = re.sub(rep, '', ap_conf)
                public.writeFile(ap_file, ap_conf)

    def _set_ols_proxy(self, get):
        if False:
            return 10
        proxyname_md5 = self.__calc_md5(get.proxyname)
        dir_path = '%s/panel/vhost/openlitespeed/proxy/%s/' % (self.setupPath, get.sitename)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_path = '{}{}_{}.conf'.format(dir_path, proxyname_md5, get.sitename)
        reverse_proxy_conf = '\nextprocessor %s {\n  type                    proxy\n  address                 %s\n  maxConns                1000\n  pcKeepAliveTimeout      600\n  initTimeout             600\n  retryTimeout            0\n  respBuffer              0\n}\n' % (get.proxyname, get.proxysite)
        public.writeFile(file_path, reverse_proxy_conf)
        dir_path = '%s/panel/vhost/openlitespeed/proxy/%s/urlrewrite/' % (self.setupPath, get.sitename)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_path = '{}{}_{}.conf'.format(dir_path, proxyname_md5, get.sitename)
        reverse_urlrewrite_conf = '\nRewriteRule ^%s(.*)$ http://%s/$1 [P,E=Proxy-Host:%s]\n' % (get.proxydir, get.proxyname, get.todomain)
        public.writeFile(file_path, reverse_urlrewrite_conf)

    def CheckLocation(self, get):
        if False:
            print('Hello World!')
        rewriteconfpath = '%s/panel/vhost/rewrite/%s.conf' % (self.setupPath, get.sitename)
        nginxconfpath = '%s/nginx/conf/nginx.conf' % self.setupPath
        vhostpath = '%s/panel/vhost/nginx/%s.conf' % (self.setupPath, get.sitename)
        rep = 'location\\s+/[\n\\s]+{'
        for i in [rewriteconfpath, nginxconfpath, vhostpath]:
            conf = public.readFile(i)
            if re.findall(rep, conf):
                return public.returnMsg(False, '伪静态/nginx主配置/vhost/文件已经存在全局反向代理')

    def CreateProxy(self, get):
        if False:
            i = 10
            return i + 15
        try:
            nocheck = get.nocheck
        except:
            nocheck = ''
        if not nocheck:
            if self.__CheckStart(get, 'create'):
                return self.__CheckStart(get, 'create')
        if public.get_webserver() == 'nginx':
            if self.CheckLocation(get):
                return self.CheckLocation(get)
        if not get.proxysite.split('//')[-1]:
            return public.returnMsg(False, '目标URL不能为[http://或https://],请填写完整URL，如：https://www.bt.cn')
        proxyUrl = self.__read_config(self.__proxyfile)
        proxyUrl.append({'proxyname': get.proxyname, 'sitename': get.sitename, 'proxydir': get.proxydir, 'proxysite': get.proxysite, 'todomain': get.todomain, 'type': int(get.type), 'cache': int(get.cache), 'subfilter': json.loads(get.subfilter), 'advanced': int(get.advanced), 'cachetime': int(get.cachetime)})
        self.__write_config(self.__proxyfile, proxyUrl)
        self.SetNginx(get)
        self.SetApache(get.sitename)
        self._set_ols_proxy(get)
        status = self.SetProxy(get)
        if not status['status']:
            return status
        if get.proxydir == '/':
            get.version = '00'
            get.siteName = get.sitename
            self.SetPHPVersion(get)
        public.serviceReload()
        return public.returnMsg(True, '添加成功')

    def GetProxyFile(self, get):
        if False:
            i = 10
            return i + 15
        import files
        conf = self.__read_config(self.__proxyfile)
        sitename = get.sitename
        proxyname = get.proxyname
        proxyname_md5 = self.__calc_md5(proxyname)
        get.path = '%s/panel/vhost/%s/proxy/%s/%s_%s.conf' % (self.setupPath, get.webserver, sitename, proxyname_md5, sitename)
        for i in conf:
            if proxyname == i['proxyname'] and sitename == i['sitename'] and (i['type'] != 1):
                return public.returnMsg(False, '代理已暂停')
        f = files.files()
        return (f.GetFileBody(get), get.path)

    def SaveProxyFile(self, get):
        if False:
            i = 10
            return i + 15
        import files
        f = files.files()
        return f.SaveFileBody(get)

    def check_annotate(self, data):
        if False:
            print('Hello World!')
        rep = '\n\\s*#Set\\s*Nginx\\s*Cache'
        if re.search(rep, data):
            return True

    def old_proxy_conf(self, conf, ng_conf_file, get):
        if False:
            print('Hello World!')
        rep = 'location\\s*\\~\\*.*gif\\|png\\|jpg\\|css\\|js\\|woff\\|woff2\\)\\$'
        if not re.search(rep, conf):
            return conf
        self.RemoveProxy(get)
        self.CreateProxy(get)
        return public.readFile(ng_conf_file)

    def ModifyProxy(self, get):
        if False:
            while True:
                i = 10
        proxyname_md5 = self.__calc_md5(get.proxyname)
        ap_conf_file = '{p}/panel/vhost/apache/proxy/{s}/{n}_{s}.conf'.format(p=self.setupPath, s=get.sitename, n=proxyname_md5)
        ng_conf_file = '{p}/panel/vhost/nginx/proxy/{s}/{n}_{s}.conf'.format(p=self.setupPath, s=get.sitename, n=proxyname_md5)
        ols_conf_file = '{p}/panel/vhost/openlitespeed/proxy/{s}/urlrewrite/{n}_{s}.conf'.format(p=self.setupPath, s=get.sitename, n=proxyname_md5)
        if self.__CheckStart(get):
            return self.__CheckStart(get)
        conf = self.__read_config(self.__proxyfile)
        random_string = public.GetRandomString(8)
        for i in range(len(conf)):
            if conf[i]['proxyname'] == get.proxyname and conf[i]['sitename'] == get.sitename:
                if int(get.type) != 1:
                    public.ExecShell('mv {f} {f}_bak'.format(f=ap_conf_file))
                    public.ExecShell('mv {f} {f}_bak'.format(f=ng_conf_file))
                    public.ExecShell('mv {f} {f}_bak'.format(f=ols_conf_file))
                    conf[i]['type'] = int(get.type)
                    self.__write_config(self.__proxyfile, conf)
                    public.serviceReload()
                    return public.returnMsg(True, '修改成功')
                else:
                    if os.path.exists(ap_conf_file + '_bak'):
                        public.ExecShell('mv {f}_bak {f}'.format(f=ap_conf_file))
                        public.ExecShell('mv {f}_bak {f}'.format(f=ng_conf_file))
                        public.ExecShell('mv {f}_bak {f}'.format(f=ols_conf_file))
                    ng_conf = public.readFile(ng_conf_file)
                    ng_conf = self.old_proxy_conf(ng_conf, ng_conf_file, get)
                    php_pass_proxy = get.proxysite
                    if get.proxysite[-1] == '/' or get.proxysite.count('/') > 2 or '?' in get.proxysite:
                        php_pass_proxy = re.search('(https?\\:\\/\\/[\\w\\.]+)', get.proxysite).group(0)
                    ng_conf = re.sub('location\\s+[\\^\\~]*\\s?%s' % conf[i]['proxydir'], 'location ^~ ' + get.proxydir, ng_conf)
                    ng_conf = re.sub('proxy_pass\\s+%s' % conf[i]['proxysite'], 'proxy_pass ' + get.proxysite, ng_conf)
                    ng_conf = re.sub('location\\s+\\~\\*\\s+\\\\.\\(php.*\n\\{\\s*proxy_pass\\s+%s.*' % php_pass_proxy, 'location ~* \\.(php|jsp|cgi|asp|aspx)$\n{\n\tproxy_pass %s;' % php_pass_proxy, ng_conf)
                    ng_conf = re.sub('location\\s+\\~\\*\\s+\\\\.\\(gif.*\n\\{\\s*proxy_pass\\s+%s.*' % php_pass_proxy, 'location ~* \\.(gif|png|jpg|css|js|woff|woff2)$\n{\n\tproxy_pass %s;' % php_pass_proxy, ng_conf)
                    backslash = ''
                    if 'Host $host' in ng_conf:
                        backslash = '\\'
                    ng_conf = re.sub('\\sHost\\s+%s' % backslash + conf[i]['todomain'], ' Host ' + get.todomain, ng_conf)
                    cache_rep = 'proxy_cache_valid\\s+200\\s+304\\s+301\\s+302\\s+\\d+m;((\\n|.)+expires\\s+\\d+m;)*'
                    if int(get.cache) == 1:
                        if re.search(cache_rep, ng_conf):
                            expires_rep = '\\{\n\\s+expires\\s+12h;'
                            ng_conf = re.sub(expires_rep, '{', ng_conf)
                            ng_conf = re.sub(cache_rep, 'proxy_cache_valid 200 304 301 302 {0}m;'.format(get.cachetime), ng_conf)
                        else:
                            ng_cache = '\n    if ( $uri ~* "\\.(gif|png|jpg|css|js|woff|woff2)$" )\n    {\n        expires 12h;\n    }\n    proxy_ignore_headers Set-Cookie Cache-Control expires;\n    proxy_cache cache_one;\n    proxy_cache_key $host$uri$is_args$args;\n    proxy_cache_valid 200 304 301 302 %sm;' % get.cachetime
                            if self.check_annotate(ng_conf):
                                cache_rep = '\n\\s*#Set\\s*Nginx\\s*Cache(.|\n)*no-cache;\\s*\n*\\s*\\}'
                                ng_conf = re.sub(cache_rep, '\n\t#Set Nginx Cache\n' + ng_cache, ng_conf)
                            else:
                                cache_rep = 'proxy_set_header\\s+REMOTE-HOST\\s+\\$remote_addr;'
                                ng_conf = re.sub(cache_rep, '\\n\\tproxy_set_header\\s+REMOTE-HOST\\s+\\$remote_addr;\\n\\t#Set Nginx Cache' + ng_cache, ng_conf)
                    else:
                        no_cache = '\n    #Set Nginx Cache\n    set $static_file%s 0;\n    if ( $uri ~* "\\.(gif|png|jpg|css|js|woff|woff2)$" )\n    {\n        set $static_file%s 1;\n        expires 12h;\n        }\n    if ( $static_file%s = 0 )\n    {\n    add_header Cache-Control no-cache;\n    }' % (random_string, random_string, random_string)
                        if self.check_annotate(ng_conf):
                            rep = '\\n\\s*#Set\\s*Nginx\\s*Cache(.|\\n)*\\d+m;'
                            ng_conf = re.sub(rep, no_cache, ng_conf)
                        else:
                            rep = '\\s+proxy_cache\\s+cache_one.*[\\n\\s\\w\\_\\";\\$]+m;'
                            ng_conf = re.sub(rep, no_cache, ng_conf)
                    sub_rep = 'sub_filter'
                    subfilter = json.loads(get.subfilter)
                    if str(conf[i]['subfilter']) != str(subfilter):
                        if re.search(sub_rep, ng_conf):
                            sub_rep = '\\s+proxy_set_header\\s+Accept-Encoding(.|\n)+off;'
                            ng_conf = re.sub(sub_rep, '', ng_conf)
                        ng_subdata = ''
                        ng_sub_filter = '\n    proxy_set_header Accept-Encoding "";%s\n    sub_filter_once off;'
                        if subfilter:
                            for s in subfilter:
                                if not s['sub1']:
                                    continue
                                if '"' in s['sub1']:
                                    s['sub1'] = s['sub1'].replace('"', '\\"')
                                if '"' in s['sub2']:
                                    s['sub2'] = s['sub2'].replace('"', '\\"')
                                ng_subdata += '\n\tsub_filter "%s" "%s";' % (s['sub1'], s['sub2'])
                        if ng_subdata:
                            ng_sub_filter = ng_sub_filter % ng_subdata
                        else:
                            ng_sub_filter = ''
                        sub_rep = '#Set\\s+Nginx\\s+Cache'
                        ng_conf = re.sub(sub_rep, '#Set Nginx Cache\n' + ng_sub_filter, ng_conf)
                    ap_conf = public.readFile(ap_conf_file)
                    ap_conf = re.sub('ProxyPass\\s+%s\\s+%s' % (conf[i]['proxydir'], conf[i]['proxysite']), 'ProxyPass %s %s' % (get.proxydir, get.proxysite), ap_conf)
                    ap_conf = re.sub('ProxyPassReverse\\s+%s\\s+%s' % (conf[i]['proxydir'], conf[i]['proxysite']), 'ProxyPassReverse %s %s' % (get.proxydir, get.proxysite), ap_conf)
                    p = '{p}/panel/vhost/openlitespeed/proxy/{s}/{n}_{s}.conf'.format(p=self.setupPath, n=proxyname_md5, s=get.sitename)
                    c = public.readFile(p)
                    if c:
                        rep = 'address\\s+(.*)'
                        new_proxysite = 'address\t{}'.format(get.proxysite)
                        c = re.sub(rep, new_proxysite, c)
                        public.writeFile(p, c)
                    c = public.readFile(ols_conf_file)
                    if c:
                        rep = 'RewriteRule\\s*\\^{}\\(\\.\\*\\)\\$\\s+http://{}/\\$1\\s*\\[P,E=Proxy-Host:{}\\]'.format(conf[i]['proxydir'], get.proxyname, conf[i]['todomain'])
                        new_content = 'RewriteRule ^{}(.*)$ http://{}/$1 [P,E=Proxy-Host:{}]'.format(get.proxydir, get.proxyname, get.todomain)
                        c = re.sub(rep, new_content, c)
                        public.writeFile(ols_conf_file, c)
                    conf[i]['proxydir'] = get.proxydir
                    conf[i]['proxysite'] = get.proxysite
                    conf[i]['todomain'] = get.todomain
                    conf[i]['type'] = int(get.type)
                    conf[i]['cache'] = int(get.cache)
                    conf[i]['subfilter'] = json.loads(get.subfilter)
                    conf[i]['advanced'] = int(get.advanced)
                    conf[i]['cachetime'] = int(get.cachetime)
                    public.writeFile(ng_conf_file, ng_conf)
                    public.writeFile(ap_conf_file, ap_conf)
                    self.__write_config(self.__proxyfile, conf)
                    self.SetNginx(get)
                    self.SetApache(get.sitename)
                    public.serviceReload()
                    return public.returnMsg(True, '修改成功')

    def SetProxy(self, get):
        if False:
            i = 10
            return i + 15
        sitename = get.sitename
        advanced = int(get.advanced)
        type = int(get.type)
        cache = int(get.cache)
        cachetime = int(get.cachetime)
        proxysite = get.proxysite
        proxydir = get.proxydir
        ng_file = self.setupPath + '/panel/vhost/nginx/' + sitename + '.conf'
        ap_file = self.setupPath + '/panel/vhost/apache/' + sitename + '.conf'
        p_conf = self.__read_config(self.__proxyfile)
        random_string = public.GetRandomString(8)
        ng_cache = '\n    if ( $uri ~* "\\.(gif|png|jpg|css|js|woff|woff2)$" )\n    {\n    \texpires 12h;\n    }\n    proxy_ignore_headers Set-Cookie Cache-Control expires;\n    proxy_cache cache_one;\n    proxy_cache_key $host$uri$is_args$args;\n    proxy_cache_valid 200 304 301 302 %sm;' % cachetime
        no_cache = '\n    set $static_file%s 0;\n    if ( $uri ~* "\\.(gif|png|jpg|css|js|woff|woff2)$" )\n    {\n    \tset $static_file%s 1;\n    \texpires 12h;\n        }\n    if ( $static_file%s = 0 )\n    {\n    add_header Cache-Control no-cache;\n    }' % (random_string, random_string, random_string)
        ng_proxy = '\n#PROXY-START%s\n\nlocation ^~ %s\n{\n    proxy_pass %s;\n    proxy_set_header Host %s;\n    proxy_set_header X-Real-IP $remote_addr;\n    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n    proxy_set_header REMOTE-HOST $remote_addr;\n    \n    add_header X-Cache $upstream_cache_status;\n    \n    #Set Nginx Cache\n    %s\n    %s\n}\n\n#PROXY-END%s'
        ng_proxy_cache = ''
        proxyname_md5 = self.__calc_md5(get.proxyname)
        ng_proxyfile = '%s/panel/vhost/nginx/proxy/%s/%s_%s.conf' % (self.setupPath, sitename, proxyname_md5, sitename)
        ng_proxydir = '%s/panel/vhost/nginx/proxy/%s' % (self.setupPath, sitename)
        if not os.path.exists(ng_proxydir):
            public.ExecShell('mkdir -p %s' % ng_proxydir)
        ng_subdata = ''
        ng_sub_filter = '\n    proxy_set_header Accept-Encoding "";%s\n    sub_filter_once off;'
        if get.subfilter:
            for s in json.loads(get.subfilter):
                if not s['sub1']:
                    continue
                if '"' in s['sub1']:
                    s['sub1'] = s['sub1'].replace('"', '\\"')
                if '"' in s['sub2']:
                    s['sub2'] = s['sub2'].replace('"', '\\"')
                ng_subdata += '\n\tsub_filter "%s" "%s";' % (s['sub1'], s['sub2'])
        if ng_subdata:
            ng_sub_filter = ng_sub_filter % ng_subdata
        else:
            ng_sub_filter = ''
        if advanced == 1:
            if proxydir[-1] != '/':
                proxydir = '{}/'.format(proxydir)
            if proxysite[-1] != '/':
                proxysite = '{}/'.format(proxysite)
            if type == 1 and cache == 1:
                ng_proxy_cache += ng_proxy % (proxydir, proxydir, proxysite, get.todomain, ng_sub_filter, ng_cache, get.proxydir)
            if type == 1 and cache == 0:
                ng_proxy_cache += ng_proxy % (get.proxydir, get.proxydir, proxysite, get.todomain, ng_sub_filter, no_cache, get.proxydir)
        else:
            if type == 1 and cache == 1:
                ng_proxy_cache += ng_proxy % (get.proxydir, get.proxydir, get.proxysite, get.todomain, ng_sub_filter, ng_cache, get.proxydir)
            if type == 1 and cache == 0:
                ng_proxy_cache += ng_proxy % (get.proxydir, get.proxydir, get.proxysite, get.todomain, ng_sub_filter, no_cache, get.proxydir)
        public.writeFile(ng_proxyfile, ng_proxy_cache)
        ap_proxyfile = '%s/panel/vhost/apache/proxy/%s/%s_%s.conf' % (self.setupPath, get.sitename, proxyname_md5, get.sitename)
        ap_proxydir = '%s/panel/vhost/apache/proxy/%s' % (self.setupPath, get.sitename)
        if not os.path.exists(ap_proxydir):
            public.ExecShell('mkdir -p %s' % ap_proxydir)
        ap_proxy = ''
        if type == 1:
            ap_proxy += '#PROXY-START%s\n<IfModule mod_proxy.c>\n    ProxyRequests Off\n    SSLProxyEngine on\n    ProxyPass %s %s/\n    ProxyPassReverse %s %s/\n    </IfModule>\n#PROXY-END%s' % (get.proxydir, get.proxydir, get.proxysite, get.proxydir, get.proxysite, get.proxydir)
        public.writeFile(ap_proxyfile, ap_proxy)
        isError = public.checkWebConfig()
        if isError != True:
            if public.get_webserver() == 'nginx':
                shutil.copyfile('/tmp/ng_file_bk.conf', ng_file)
            else:
                shutil.copyfile('/tmp/ap_file_bk.conf', ap_file)
            for i in range(len(p_conf) - 1, -1, -1):
                if get.sitename == p_conf[i]['sitename'] and p_conf[i]['proxyname']:
                    del p_conf[i]
            self.RemoveProxy(get)
            return public.returnMsg(False, 'ERROR: %s<br><a style="color:red;">' % public.GetMsg('CONFIG_ERROR') + isError.replace('\n', '<br>') + '</a>')
        return public.returnMsg(True, 'SUCCESS')

    def ProxyCache(self, get):
        if False:
            while True:
                i = 10
        if public.get_webserver() != 'nginx':
            return public.returnMsg(False, 'WAF_NOT_NGINX')
        file = self.setupPath + '/panel/vhost/nginx/' + get.siteName + '.conf'
        conf = public.readFile(file)
        if conf.find('proxy_pass') == -1:
            return public.returnMsg(False, 'SET_ERROR')
        if conf.find('#proxy_cache') != -1:
            conf = conf.replace('#proxy_cache', 'proxy_cache')
            conf = conf.replace('#expires 12h', 'expires 12h')
        else:
            conf = conf.replace('proxy_cache', '#proxy_cache')
            conf = conf.replace('expires 12h', '#expires 12h')
        public.writeFile(file, conf)
        public.serviceReload()
        return public.returnMsg(True, 'SET_SUCCESS')

    def CheckProxy(self, get):
        if False:
            return 10
        if public.get_webserver() != 'nginx':
            return True
        file = self.setupPath + '/nginx/conf/proxy.conf'
        if not os.path.exists(file):
            conf = 'proxy_temp_path %s/nginx/proxy_temp_dir;\n    proxy_cache_path %s/nginx/proxy_cache_dir levels=1:2 keys_zone=cache_one:10m inactive=1d max_size=5g;\n    client_body_buffer_size 512k;\n    proxy_connect_timeout 60;\n    proxy_read_timeout 60;\n    proxy_send_timeout 60;\n    proxy_buffer_size 32k;\n    proxy_buffers 4 64k;\n    proxy_busy_buffers_size 128k;\n    proxy_temp_file_write_size 128k;\n    proxy_next_upstream error timeout invalid_header http_500 http_503 http_404;\n    proxy_cache cache_one;' % (self.setupPath, self.setupPath)
            public.writeFile(file, conf)
        file = self.setupPath + '/nginx/conf/nginx.conf'
        conf = public.readFile(file)
        if conf.find('include proxy.conf;') == -1:
            rep = 'include\\s+mime.types;'
            conf = re.sub(rep, 'include mime.types;\n\tinclude proxy.conf;', conf)
            public.writeFile(file, conf)

    def get_project_find(self, project_name):
        if False:
            return 10
        '\n            @name 获取指定项目配置\n            @author hwliang<2021-08-09>\n            @param project_name<string> 项目名称\n            @return dict\n        '
        project_info = public.M('sites').where('project_type=? AND name=?', ('Java', project_name)).find()
        if not project_info:
            return False
        project_info['project_config'] = json.loads(project_info['project_config'])
        return project_info

    def GetRewriteList(self, get):
        if False:
            for i in range(10):
                print('nop')
        if get.siteName.find('node_') == 0:
            get.siteName = get.siteName.replace('node_', '')
        if get.siteName.find('java_') == 0:
            get.siteName = get.siteName.replace('java_', '')
        rewriteList = {}
        ws = public.get_webserver()
        if ws == 'openlitespeed':
            ws = 'apache'
        if ws == 'apache':
            Java_data = self.get_project_find(get.siteName)
            if not Java_data:
                get.id = public.M('sites').where('name=?', (get.siteName,)).getField('id')
                runPath = self.GetSiteRunPath(get)
                if runPath['runPath'].find('/www/server/stop') != -1:
                    runPath['runPath'] = runPath['runPath'].replace('/www/server/stop', '')
                rewriteList['sitePath'] = public.M('sites').where('name=?', (get.siteName,)).getField('path') + runPath['runPath']
            if Java_data:
                if Java_data['project_config']['java_type'] == 'springboot':
                    if 'static_path' in Java_data['project_config']:
                        rewriteList['sitePath'] = Java_data['project_config']['static_path']
                    else:
                        rewriteList['sitePath'] = Java_data['project_config']['jar_path']
                else:
                    get.id = public.M('sites').where('name=?', (get.siteName,)).getField('id')
                    runPath = self.GetSiteRunPath(get)
                    if runPath['runPath'].find('/www/server/stop') != -1:
                        runPath['runPath'] = runPath['runPath'].replace('/www/server/stop', '')
                    rewriteList['sitePath'] = public.M('sites').where('name=?', (get.siteName,)).getField('path') + runPath['runPath']
        rewriteList['rewrite'] = []
        rewriteList['rewrite'].append('0.' + public.getMsg('SITE_REWRITE_NOW'))
        for ds in os.listdir('rewrite/' + ws):
            if ds == 'list.txt':
                continue
            rewriteList['rewrite'].append(ds[0:len(ds) - 5])
        rewriteList['rewrite'] = sorted(rewriteList['rewrite'])
        return rewriteList

    def SetRewriteTel(self, get):
        if False:
            return 10
        ws = public.get_webserver()
        if ws == 'openlitespeed':
            ws = 'apache'
        if sys.version_info[0] == 2:
            get.name = get.name.encode('utf-8')
        filename = 'rewrite/' + ws + '/' + get.name + '.conf'
        public.writeFile(filename, get.data)
        return public.returnMsg(True, 'SITE_REWRITE_SAVE')

    def ToBackup(self, get):
        if False:
            while True:
                i = 10
        id = get.id
        find = public.M('sites').where('id=?', (id,)).field('name,path,id').find()
        import time
        fileName = find['name'] + '_' + time.strftime('%Y%m%d_%H%M%S', time.localtime()) + '.zip'
        backupPath = session['config']['backup_path'] + '/site'
        zipName = backupPath + '/' + fileName
        if not os.path.exists(backupPath):
            os.makedirs(backupPath)
        tmps = '/tmp/panelExec.log'
        execStr = "cd '" + find['path'] + "' && zip '" + zipName + "' -x .user.ini -r ./ > " + tmps + ' 2>&1'
        public.ExecShell(execStr)
        sql = public.M('backup').add('type,name,pid,filename,size,addtime', (0, fileName, find['id'], zipName, 0, public.getDate()))
        public.WriteLog('TYPE_SITE', 'SITE_BACKUP_SUCCESS', (find['name'],))
        return public.returnMsg(True, 'BACKUP_SUCCESS')

    def DelBackup(self, get):
        if False:
            for i in range(10):
                print('nop')
        id = get.id
        where = 'id=?'
        filename = public.M('backup').where(where, (id,)).getField('filename')
        if os.path.exists(filename):
            os.remove(filename)
        name = ''
        if filename == 'qiniu':
            name = public.M('backup').where(where, (id,)).getField('name')
            public.ExecShell(public.get_python_bin() + ' ' + self.setupPath + '/panel/script/backup_qiniu.py delete_file ' + name)
        public.WriteLog('TYPE_SITE', 'SITE_BACKUP_DEL_SUCCESS', (name, filename))
        public.M('backup').where(where, (id,)).delete()
        return public.returnMsg(True, 'DEL_SUCCESS')

    def OldConfigFile(self):
        if False:
            return 10
        moveTo = 'data/moveTo.pl'
        if os.path.exists(moveTo):
            return
        filename = self.setupPath + '/nginx/conf/nginx.conf'
        if os.path.exists(filename):
            conf = public.readFile(filename)
            if conf.find('include vhost/*.conf;') != -1:
                conf = conf.replace('include vhost/*.conf;', 'include ' + self.setupPath + '/panel/vhost/nginx/*.conf;')
                public.writeFile(filename, conf)
        self.moveConf(self.setupPath + '/nginx/conf/vhost', self.setupPath + '/panel/vhost/nginx', 'rewrite', self.setupPath + '/panel/vhost/rewrite')
        self.moveConf(self.setupPath + '/nginx/conf/rewrite', self.setupPath + '/panel/vhost/rewrite')
        filename = self.setupPath + '/apache/conf/httpd.conf'
        if os.path.exists(filename):
            conf = public.readFile(filename)
            if conf.find('IncludeOptional conf/vhost/*.conf') != -1:
                conf = conf.replace('IncludeOptional conf/vhost/*.conf', 'IncludeOptional ' + self.setupPath + '/panel/vhost/apache/*.conf')
                public.writeFile(filename, conf)
        self.moveConf(self.setupPath + '/apache/conf/vhost', self.setupPath + '/panel/vhost/apache')
        public.writeFile(moveTo, 'True')
        public.serviceReload()

    def moveConf(self, Path, toPath, Replace=None, ReplaceTo=None):
        if False:
            print('Hello World!')
        if not os.path.exists(Path):
            return
        import shutil
        letPath = '/etc/letsencrypt/live'
        nginxPath = self.setupPath + '/nginx/conf/key'
        apachePath = self.setupPath + '/apache/conf/key'
        for filename in os.listdir(Path):
            name = filename[0:len(filename) - 5]
            filename = Path + '/' + filename
            conf = public.readFile(filename)
            if Replace:
                conf = conf.replace(Replace, ReplaceTo)
            ReplaceTo = letPath + name
            Replace = 'conf/key/' + name
            if conf.find(Replace) != -1:
                conf = conf.replace(Replace, ReplaceTo)
            Replace = 'key/' + name
            if conf.find(Replace) != -1:
                conf = conf.replace(Replace, ReplaceTo)
            public.writeFile(filename, conf)
            if conf.find('server_name') != -1:
                self.formatNginxConf(filename)
            elif conf.find('<Directory') != -1:
                pass
            shutil.move(filename, toPath + '/' + name + '.conf')
            self.moveKey(nginxPath + '/' + name, letPath + '/' + name)
            self.moveKey(apachePath + '/' + name, letPath + '/' + name)
        shutil.rmtree(Path)
        public.serviceReload()

    def formatNginxConf(self, filename):
        if False:
            return 10
        name = os.path.basename(filename[0:len(filename) - 5])
        if name.find('.') == -1:
            return
        conf = public.readFile(filename)
        rep = 'server_name\\s+(.+);'
        tmp = re.search(rep, conf)
        if not tmp:
            return
        domains = tmp.groups()[0].split(' ')
        rep = 'root\\s+(.+);'
        tmp = re.search(rep, conf)
        if not tmp:
            return
        path = tmp.groups()[0]
        self.toSiteDatabase(name, domains, path)

    def formatApacheConf(self, filename):
        if False:
            return 10
        name = os.path.basename(filename[0:len(filename) - 5])
        if name.find('.') == -1:
            return
        conf = public.readFile(filename)
        rep = 'ServerAlias\\s+(.+)\n'
        tmp = re.search(rep, conf)
        if not tmp:
            return
        domains = tmp.groups()[0].split(' ')
        rep = u'DocumentRoot\\s+"(.+)"\n'
        tmp = re.search(rep, conf)
        if not tmp:
            return
        path = tmp.groups()[0]
        self.toSiteDatabase(name, domains, path)

    def toSiteDatabase(self, name, domains, path):
        if False:
            print('Hello World!')
        if public.M('sites').where('name=?', (name,)).count() > 0:
            return
        public.M('sites').add('name,path,status,ps,addtime', (name, path, '1', '请输入备注', public.getDate()))
        pid = public.M('sites').where('name=?', (name,)).getField('id')
        for domain in domains:
            public.M('domain').add('pid,name,port,addtime', (pid, domain, '80', public.getDate()))

    def moveKey(self, srcPath, dstPath):
        if False:
            return 10
        if not os.path.exists(srcPath):
            return
        import shutil
        os.makedirs(dstPath)
        srcKey = srcPath + '/key.key'
        srcCsr = srcPath + '/csr.key'
        if os.path.exists(srcKey):
            shutil.move(srcKey, dstPath + '/privkey.pem')
        if os.path.exists(srcCsr):
            shutil.move(srcCsr, dstPath + '/fullchain.pem')

    def GetPath(self, path):
        if False:
            while True:
                i = 10
        if path[-1] == '/':
            return path[0:-1]
        return path

    def logsOpen(self, get):
        if False:
            while True:
                i = 10
        get.name = public.M('sites').where('id=?', (get.id,)).getField('name')
        filename = public.GetConfigValue('setup_path') + '/panel/vhost/apache/' + get.name + '.conf'
        if os.path.exists(filename):
            conf = public.readFile(filename)
            if conf.find('#ErrorLog') != -1:
                conf = conf.replace('#ErrorLog', 'ErrorLog').replace('#CustomLog', 'CustomLog')
            else:
                conf = conf.replace('ErrorLog', '#ErrorLog').replace('CustomLog', '#CustomLog')
            public.writeFile(filename, conf)
        filename = public.GetConfigValue('setup_path') + '/panel/vhost/nginx/' + get.name + '.conf'
        if os.path.exists(filename):
            conf = public.readFile(filename)
            rep = public.GetConfigValue('logs_path') + '/' + get.name + '.log'
            if conf.find(rep) != -1:
                conf = conf.replace(rep, '/dev/null')
            else:
                conf = conf.replace('access_log  /dev/null', 'access_log  ' + rep)
            public.writeFile(filename, conf)
        filename = public.GetConfigValue('setup_path') + '/panel/vhost/openlitespeed/detail/' + get.name + '.conf'
        conf = public.readFile(filename)
        if conf:
            rep = '\nerrorlog(.|\n)*compressArchive\\s*1\\s*\n}'
            tmp = re.search(rep, conf)
            s = 'on'
            if not tmp:
                s = 'off'
                rep = '\n#errorlog(.|\n)*compressArchive\\s*1\\s*\n#}'
                tmp = re.search(rep, conf)
            tmp = tmp.group()
            if tmp:
                result = ''
                if s == 'on':
                    for l in tmp.strip().splitlines():
                        result += '\n#' + l
                else:
                    for l in tmp.splitlines():
                        result += '\n' + l[1:]
                conf = re.sub(rep, '\n' + result.strip(), conf)
                public.writeFile(filename, conf)
        public.serviceReload()
        return public.returnMsg(True, 'SUCCESS')

    def GetLogsStatus(self, get):
        if False:
            for i in range(10):
                print('nop')
        filename = public.GetConfigValue('setup_path') + '/panel/vhost/' + public.get_webserver() + '/' + get.name + '.conf'
        if public.get_webserver() == 'openlitespeed':
            filename = public.GetConfigValue('setup_path') + '/panel/vhost/' + public.get_webserver() + '/detail/' + get.name + '.conf'
        conf = public.readFile(filename)
        if not conf:
            return True
        if conf.find('#ErrorLog') != -1:
            return False
        if conf.find('access_log  /dev/null') != -1:
            return False
        if re.search('\n#accesslog', conf):
            return False
        return True

    def GetHasPwd(self, get):
        if False:
            while True:
                i = 10
        if not hasattr(get, 'siteName'):
            get.siteName = public.M('sites').where('id=?', (get.id,)).getField('name')
            get.configFile = self.setupPath + '/panel/vhost/nginx/' + get.siteName + '.conf'
        conf = public.readFile(get.configFile)
        if type(conf) == bool:
            return False
        if conf.find('#AUTH_START') != -1:
            return True
        return False

    def SetHasPwd(self, get):
        if False:
            while True:
                i = 10
        if public.get_webserver() == 'openlitespeed':
            return public.returnMsg(False, '该功能暂时还不支持OpenLiteSpeed')
        if len(get.username.strip()) < 3 or len(get.password.strip()) < 3:
            return public.returnMsg(False, '用户名或密码不能小于3位！')
        if not hasattr(get, 'siteName'):
            get.siteName = public.M('sites').where('id=?', (get.id,)).getField('name')
        self.CloseHasPwd(get)
        filename = public.GetConfigValue('setup_path') + '/pass/' + get.siteName + '.pass'
        passconf = get.username + ':' + public.hasPwd(get.password)
        if get.siteName == 'phpmyadmin':
            get.configFile = self.setupPath + '/nginx/conf/nginx.conf'
            if os.path.exists(self.setupPath + '/panel/vhost/nginx/phpmyadmin.conf'):
                get.configFile = self.setupPath + '/panel/vhost/nginx/phpmyadmin.conf'
        else:
            get.configFile = self.setupPath + '/panel/vhost/nginx/' + get.siteName + '.conf'
        conf = public.readFile(get.configFile)
        if conf:
            rep = '#error_page   404   /404.html;'
            if conf.find(rep) == -1:
                rep = '#error_page 404/404.html;'
            data = '\n    #AUTH_START\n    auth_basic "Authorization";\n    auth_basic_user_file %s;\n    #AUTH_END' % (filename,)
            conf = conf.replace(rep, rep + data)
            public.writeFile(get.configFile, conf)
        if get.siteName == 'phpmyadmin':
            get.configFile = self.setupPath + '/apache/conf/extra/httpd-vhosts.conf'
            if os.path.exists(self.setupPath + '/panel/vhost/apache/phpmyadmin.conf'):
                get.configFile = self.setupPath + '/panel/vhost/apache/phpmyadmin.conf'
        else:
            get.configFile = self.setupPath + '/panel/vhost/apache/' + get.siteName + '.conf'
        conf = public.readFile(get.configFile)
        if conf:
            rep = 'SetOutputFilter'
            if conf.find(rep) != -1:
                data = '#AUTH_START\n        AuthType basic\n        AuthName "Authorization "\n        AuthUserFile %s\n        Require user %s\n        #AUTH_END\n        ' % (filename, get.username)
                conf = conf.replace(rep, data + rep)
                conf = conf.replace(' Require all granted', ' #Require all granted')
                public.writeFile(get.configFile, conf)
        passDir = public.GetConfigValue('setup_path') + '/pass'
        if not os.path.exists(passDir):
            public.ExecShell('mkdir -p ' + passDir)
        public.writeFile(filename, passconf)
        public.serviceReload()
        public.WriteLog('TYPE_SITE', 'SITE_AUTH_OPEN_SUCCESS', (get.siteName,))
        return public.returnMsg(True, 'SET_SUCCESS')

    def CloseHasPwd(self, get):
        if False:
            for i in range(10):
                print('nop')
        if not hasattr(get, 'siteName'):
            get.siteName = public.M('sites').where('id=?', (get.id,)).getField('name')
        if get.siteName == 'phpmyadmin':
            get.configFile = self.setupPath + '/nginx/conf/nginx.conf'
        else:
            get.configFile = self.setupPath + '/panel/vhost/nginx/' + get.siteName + '.conf'
        if os.path.exists(get.configFile):
            conf = public.readFile(get.configFile)
            rep = '\n\\s*#AUTH_START(.|\n){1,200}#AUTH_END'
            conf = re.sub(rep, '', conf)
            public.writeFile(get.configFile, conf)
        if get.siteName == 'phpmyadmin':
            get.configFile = self.setupPath + '/apache/conf/extra/httpd-vhosts.conf'
        else:
            get.configFile = self.setupPath + '/panel/vhost/apache/' + get.siteName + '.conf'
        if os.path.exists(get.configFile):
            conf = public.readFile(get.configFile)
            rep = '\n\\s*#AUTH_START(.|\n){1,200}#AUTH_END'
            conf = re.sub(rep, '', conf)
            conf = conf.replace(' #Require all granted', ' Require all granted')
            public.writeFile(get.configFile, conf)
        public.serviceReload()
        public.WriteLog('TYPE_SITE', 'SITE_AUTH_CLOSE_SUCCESS', (get.siteName,))
        return public.returnMsg(True, 'SET_SUCCESS')

    def SetTomcat(self, get):
        if False:
            return 10
        siteName = get.siteName
        name = siteName.replace('.', '_')
        rep = '^(\\d{1,3}\\.){3,3}\\d{1,3}$'
        if re.match(rep, siteName):
            return public.returnMsg(False, 'TOMCAT_IP')
        filename = self.setupPath + '/panel/vhost/nginx/' + siteName + '.conf'
        if os.path.exists(filename):
            conf = public.readFile(filename)
            if conf.find('#TOMCAT-START') != -1:
                return self.CloseTomcat(get)
            tomcatConf = '#TOMCAT-START\n    location /\n    {\n        proxy_pass "http://%s:8080";\n        proxy_set_header Host %s;\n        proxy_set_header X-Forwarded-For $remote_addr;\n    }\n    location ~ .*\\.(gif|jpg|jpeg|bmp|png|ico|txt|js|css)$\n    {\n        expires      12h;\n    }\n    \n    location ~ .*\\.war$\n    {\n        return 404;\n    }\n    #TOMCAT-END\n    ' % (siteName, siteName)
            rep = 'include enable-php'
            conf = conf.replace(rep, tomcatConf + rep)
            public.writeFile(filename, conf)
        filename = self.setupPath + '/panel/vhost/apache/' + siteName + '.conf'
        if os.path.exists(filename):
            conf = public.readFile(filename)
            if conf.find('#TOMCAT-START') != -1:
                return self.CloseTomcat(get)
            tomcatConf = '#TOMCAT-START\n    <IfModule mod_proxy.c>\n        ProxyRequests Off\n        SSLProxyEngine on\n        ProxyPass / http://%s:8080/\n        ProxyPassReverse / http://%s:8080/\n        RequestHeader unset Accept-Encoding\n        ExtFilterDefine fixtext mode=output intype=text/html cmd="/bin/sed \'s,:8080,,g\'"\n        SetOutputFilter fixtext\n    </IfModule>\n    #TOMCAT-END\n    ' % (siteName, siteName)
            rep = '#PATH'
            conf = conf.replace(rep, tomcatConf + rep)
            public.writeFile(filename, conf)
        path = public.M('sites').where('name=?', (siteName,)).getField('path')
        import tomcat
        tomcat.tomcat().AddVhost(path, siteName)
        public.serviceReload()
        public.ExecShell('/etc/init.d/tomcat stop')
        public.ExecShell('/etc/init.d/tomcat start')
        public.ExecShell('echo "127.0.0.1 ' + siteName + '" >> /etc/hosts')
        public.WriteLog('TYPE_SITE', 'SITE_TOMCAT_OPEN', (siteName,))
        return public.returnMsg(True, 'SITE_TOMCAT_OPEN')

    def CloseTomcat(self, get):
        if False:
            for i in range(10):
                print('nop')
        if not os.path.exists('/etc/init.d/tomcat'):
            return False
        siteName = get.siteName
        name = siteName.replace('.', '_')
        filename = self.setupPath + '/panel/vhost/nginx/' + siteName + '.conf'
        if os.path.exists(filename):
            conf = public.readFile(filename)
            rep = '\\s*#TOMCAT-START(.|\n)+#TOMCAT-END'
            conf = re.sub(rep, '', conf)
            public.writeFile(filename, conf)
        filename = self.setupPath + '/panel/vhost/apache/' + siteName + '.conf'
        if os.path.exists(filename):
            conf = public.readFile(filename)
            rep = '\\s*#TOMCAT-START(.|\n)+#TOMCAT-END'
            conf = re.sub(rep, '', conf)
            public.writeFile(filename, conf)
        public.ExecShell('rm -rf ' + self.setupPath + '/panel/vhost/tomcat/' + name)
        try:
            import tomcat
            tomcat.tomcat().DelVhost(siteName)
        except:
            pass
        public.serviceReload()
        public.ExecShell('/etc/init.d/tomcat restart')
        public.ExecShell("sed -i '/" + siteName + "/d' /etc/hosts")
        public.WriteLog('TYPE_SITE', 'SITE_TOMCAT_CLOSE', (siteName,))
        return public.returnMsg(True, 'SITE_TOMCAT_CLOSE')

    def GetSiteRunPath(self, get):
        if False:
            while True:
                i = 10
        siteName = public.M('sites').where('id=?', (get.id,)).getField('name')
        sitePath = public.M('sites').where('id=?', (get.id,)).getField('path')
        if not siteName or os.path.isfile(sitePath):
            return {'runPath': '/', 'dirs': []}
        path = sitePath
        if public.get_webserver() == 'nginx':
            filename = self.setupPath + '/panel/vhost/nginx/' + siteName + '.conf'
            if os.path.exists(filename):
                conf = public.readFile(filename)
                rep = '\\s*root\\s+(.+);'
                tmp1 = re.search(rep, conf)
                if tmp1:
                    path = tmp1.groups()[0]
        elif public.get_webserver() == 'apache':
            filename = self.setupPath + '/panel/vhost/apache/' + siteName + '.conf'
            if os.path.exists(filename):
                conf = public.readFile(filename)
                rep = '\\s*DocumentRoot\\s*"(.+)"\\s*\n'
                tmp1 = re.search(rep, conf)
                if tmp1:
                    path = tmp1.groups()[0]
        else:
            filename = self.setupPath + '/panel/vhost/openlitespeed/' + siteName + '.conf'
            if os.path.exists(filename):
                conf = public.readFile(filename)
                rep = 'vhRoot\\s*(.*)'
                path = re.search(rep, conf)
                if not path:
                    return public.returnMsg(False, 'Get Site run path false')
                path = path.groups()[0]
        data = {}
        if sitePath == path:
            data['runPath'] = '/'
        else:
            data['runPath'] = path.replace(sitePath, '')
        dirnames = []
        dirnames.append('/')
        if not os.path.exists(sitePath):
            os.makedirs(sitePath)
        for filename in os.listdir(sitePath):
            try:
                json.dumps(filename)
                if sys.version_info[0] == 2:
                    filename = filename.encode('utf-8')
                else:
                    filename.encode('utf-8')
                filePath = sitePath + '/' + filename
                if not os.path.exists(filePath):
                    continue
                if os.path.islink(filePath):
                    continue
                if os.path.isdir(filePath):
                    dirnames.append('/' + filename)
            except:
                pass
        data['dirs'] = dirnames
        return data

    def SetSiteRunPath(self, get):
        if False:
            while True:
                i = 10
        siteName = public.M('sites').where('id=?', (get.id,)).getField('name')
        sitePath = public.M('sites').where('id=?', (get.id,)).getField('path')
        old_run_path = self.GetRunPath(get)
        filename = self.setupPath + '/panel/vhost/nginx/' + siteName + '.conf'
        if os.path.exists(filename):
            conf = public.readFile(filename)
            if conf:
                rep = '\\s*root\\s+(.+);'
                tmp = re.search(rep, conf)
                if tmp:
                    path = tmp.groups()[0]
                    conf = conf.replace(path, sitePath + get.runPath)
                    public.writeFile(filename, conf)
        filename = self.setupPath + '/panel/vhost/apache/' + siteName + '.conf'
        if os.path.exists(filename):
            conf = public.readFile(filename)
            if conf:
                rep = '\\s*DocumentRoot\\s*"(.+)"\\s*\n'
                tmp = re.search(rep, conf)
                if tmp:
                    path = tmp.groups()[0]
                    conf = conf.replace(path, sitePath + get.runPath)
                    public.writeFile(filename, conf)
        self._set_ols_run_path(sitePath, get.runPath, siteName)
        s_path = sitePath + old_run_path + '/.user.ini'
        d_path = sitePath + get.runPath + '/.user.ini'
        if s_path != d_path:
            public.ExecShell('chattr -i {}'.format(s_path))
            public.ExecShell('mv {} {}'.format(s_path, d_path))
            public.ExecShell('chattr +i {}'.format(d_path))
        public.serviceReload()
        return public.returnMsg(True, 'SET_SUCCESS')

    def _set_ols_run_path(self, site_path, run_path, sitename):
        if False:
            return 10
        ols_conf_file = '{}/panel/vhost/openlitespeed/{}.conf'.format(self.setupPath, sitename)
        ols_conf = public.readFile(ols_conf_file)
        if not ols_conf:
            return
        reg = '#VHOST\\s*{s}\\s*START(.|\n)+#VHOST\\s*{s}\\s*END'.format(s=sitename)
        tmp = re.search(reg, ols_conf)
        if not tmp:
            return
        reg = 'vhRoot\\s*(.*)'
        tmp = 'vhRoot ' + site_path + run_path
        ols_conf = re.sub(reg, tmp, ols_conf)
        public.writeFile(ols_conf_file, ols_conf)

    def SetDefaultSite(self, get):
        if False:
            while True:
                i = 10
        import time
        default_site_save = 'data/defaultSite.pl'
        defaultSite = public.readFile(default_site_save)
        http2 = ''
        versionStr = public.readFile('/www/server/nginx/version.pl')
        if versionStr:
            if versionStr.find('1.8.1') == -1:
                http2 = ' http2'
        if defaultSite:
            path = self.setupPath + '/panel/vhost/nginx/' + defaultSite + '.conf'
            if os.path.exists(path):
                conf = public.readFile(path)
                rep = 'listen\\s+80.+;'
                conf = re.sub(rep, 'listen 80;', conf, 1)
                rep = 'listen\\s+\\[::\\]:80.+;'
                conf = re.sub(rep, 'listen [::]:80;', conf, 1)
                rep = 'listen\\s+443.+;'
                conf = re.sub(rep, 'listen 443 ssl' + http2 + ';', conf, 1)
                rep = 'listen\\s+\\[::\\]:443.+;'
                conf = re.sub(rep, 'listen [::]:443 ssl' + http2 + ';', conf, 1)
                public.writeFile(path, conf)
            path = self.setupPath + '/apache/htdocs/.htaccess'
            if os.path.exists(path):
                os.remove(path)
        if get.name == '0':
            if os.path.exists(default_site_save):
                os.remove(default_site_save)
            public.serviceReload()
            return public.returnMsg(True, '设置成功!')
        path = self.setupPath + '/apache/htdocs'
        if os.path.exists(path):
            conf = '<IfModule mod_rewrite.c>\n  RewriteEngine on\n  RewriteCond %{HTTP_HOST} !^127.0.0.1 [NC] \n  RewriteRule (.*) http://%s/$1 [L]\n</IfModule>'
            conf = conf.replace('%s', get.name)
            if get.name == 'off':
                conf = ''
            public.writeFile(path + '/.htaccess', conf)
        path = self.setupPath + '/panel/vhost/nginx/' + get.name + '.conf'
        if os.path.exists(path):
            conf = public.readFile(path)
            rep = 'listen\\s+80\\s*;'
            conf = re.sub(rep, 'listen 80 default_server;', conf, 1)
            rep = 'listen\\s+\\[::\\]:80\\s*;'
            conf = re.sub(rep, 'listen [::]:80 default_server;', conf, 1)
            rep = 'listen\\s+443\\s*ssl\\s*\\w*\\s*;'
            conf = re.sub(rep, 'listen 443 ssl' + http2 + ' default_server;', conf, 1)
            rep = 'listen\\s+\\[::\\]:443\\s*ssl\\s*\\w*\\s*;'
            conf = re.sub(rep, 'listen [::]:443 ssl' + http2 + ' default_server;', conf, 1)
            public.writeFile(path, conf)
        path = self.setupPath + '/panel/vhost/nginx/default.conf'
        if os.path.exists(path):
            public.ExecShell('rm -f ' + path)
        public.writeFile(default_site_save, get.name)
        public.serviceReload()
        return public.returnMsg(True, 'SET_SUCCESS')

    def GetDefaultSite(self, get):
        if False:
            for i in range(10):
                print('nop')
        data = {}
        data['sites'] = public.M('sites').where('project_type=?', 'PHP').field('name').order('id desc').select()
        data['defaultSite'] = public.readFile('data/defaultSite.pl')
        return data

    def CheckSafe(self, get):
        if False:
            i = 10
            return i + 15
        import db, time
        isTask = '/tmp/panelTask.pl'
        if os.path.exists(self.setupPath + '/panel/class/panelSafe.py'):
            import py_compile
            py_compile.compile(self.setupPath + '/panel/class/panelSafe.py')
        get.path = public.M('sites').where('id=?', (get.id,)).getField('path')
        execstr = 'cd ' + public.GetConfigValue('setup_path') + '/panel/class && ' + public.get_python_bin() + ' panelSafe.pyc ' + get.path
        sql = db.Sql()
        sql.table('tasks').add('id,name,type,status,addtime,execstr', (None, '扫描目录 [' + get.path + ']', 'execshell', '0', time.strftime('%Y-%m-%d %H:%M:%S'), execstr))
        public.writeFile(isTask, 'True')
        public.WriteLog('TYPE_SETUP', 'SITE_SCAN_ADD', (get.path,))
        return public.returnMsg(True, 'SITE_SCAN_ADD')

    def GetCheckSafe(self, get):
        if False:
            print('Hello World!')
        get.path = public.M('sites').where('id=?', (get.id,)).getField('path')
        path = get.path + '/scan.pl'
        result = {}
        result['data'] = []
        result['phpini'] = []
        result['userini'] = result['sshd'] = True
        result['scan'] = False
        result['outime'] = result['count'] = result['error'] = 0
        if not os.path.exists(path):
            return result
        import json
        return json.loads(public.readFile(path))

    def UpdateRulelist(self, get):
        if False:
            print('Hello World!')
        try:
            conf = public.httpGet(public.getUrl() + '/install/ruleList.conf')
            if conf:
                public.writeFile(self.setupPath + '/panel/data/ruleList.conf', conf)
                return public.returnMsg(True, 'UPDATE_SUCCESS')
            return public.returnMsg(False, 'CONNECT_ERR')
        except:
            return public.returnMsg(False, 'CONNECT_ERR')

    def set_site_etime_multiple(self, get):
        if False:
            return 10
        '\n            @name 批量网站到期时间\n            @author zhwen<2020-11-17>\n            @param sites_id "1,2"\n            @param edate 2020-11-18\n        '
        sites_id = get.sites_id.split(',')
        set_edate_successfully = []
        set_edate_failed = {}
        for site_id in sites_id:
            get.id = site_id
            site_name = public.M('sites').where('id=?', (site_id,)).getField('name')
            if not site_name:
                continue
            try:
                self.SetEdate(get)
                set_edate_successfully.append(site_name)
            except:
                set_edate_failed[site_name] = '设置时错误了，请再试一次'
                pass
        return {'status': True, 'msg': '设置网站 [ {} ] 到期时间成功'.format(','.join(set_edate_successfully)), 'error': set_edate_failed, 'success': set_edate_successfully}

    def SetEdate(self, get):
        if False:
            return 10
        result = public.M('sites').where('id=?', (get.id,)).setField('edate', get.edate)
        siteName = public.M('sites').where('id=?', (get.id,)).getField('name')
        public.WriteLog('TYPE_SITE', 'SITE_EXPIRE_SUCCESS', (siteName, get.edate))
        return public.returnMsg(True, 'SITE_EXPIRE_SUCCESS')

    def GetSecurity(self, get):
        if False:
            print('Hello World!')
        file = '/www/server/panel/vhost/nginx/' + get.name + '.conf'
        conf = public.readFile(file)
        data = {}
        if type(conf) == bool:
            return public.returnMsg(False, '读取配置文件失败!')
        if conf.find('SECURITY-START') != -1:
            rep = '#SECURITY-START(\n|.)+#SECURITY-END'
            tmp = re.search(rep, conf).group()
            data['fix'] = re.search('\\(.+\\)\\$', tmp).group().replace('(', '').replace(')$', '').replace('|', ',')
            try:
                data['domains'] = ','.join(list(set(re.search('valid_referers\\s+none\\s+blocked\\s+(.+);\n', tmp).groups()[0].split())))
            except:
                data['domains'] = ','.join(list(set(re.search('valid_referers\\s+(.+);\n', tmp).groups()[0].split())))
            data['status'] = True
            data['none'] = tmp.find('none blocked') != -1
            try:
                data['return_rule'] = re.findall('(return|rewrite)\\s+.*(\\d{3}|(/.+)\\s+(break|last));', conf)[0][1].replace('break', '').strip()
            except:
                data['return_rule'] = '404'
        else:
            data['fix'] = 'jpg,jpeg,gif,png,js,css'
            domains = public.M('domain').where('pid=?', (get.id,)).field('name').select()
            tmp = []
            for domain in domains:
                tmp.append(domain['name'])
            data['return_rule'] = '404'
            data['domains'] = ','.join(tmp)
            data['status'] = False
            data['none'] = False
        return data

    def SetSecurity(self, get):
        if False:
            i = 10
            return i + 15
        if len(get.fix) < 2:
            return public.returnMsg(False, 'URL后缀不能为空!')
        if len(get.domains) < 3:
            return public.returnMsg(False, '防盗链域名不能为空!')
        file = '/www/server/panel/vhost/nginx/' + get.name + '.conf'
        if os.path.exists(file):
            conf = public.readFile(file)
            if get.status == '1':
                if conf.find('SECURITY-START') == -1:
                    return public.returnMsg(False, '请先开启防盗链!')
                r_key = 'valid_referers none blocked'
                d_key = 'valid_referers'
                if conf.find(r_key) == -1:
                    conf = conf.replace(d_key, r_key)
                else:
                    conf = conf.replace(r_key, d_key)
            elif conf.find('SECURITY-START') != -1:
                rep = '\\s{0,4}#SECURITY-START(\n|.){1,500}#SECURITY-END\n?'
                conf = re.sub(rep, '', conf)
                public.WriteLog('网站管理', '站点[' + get.name + ']已关闭防盗链设置!')
            else:
                return_rule = 'return 404'
                if 'return_rule' in get:
                    get.return_rule = get.return_rule.strip()
                    if get.return_rule in ['404', '403', '200', '301', '302', '401', '201']:
                        return_rule = 'return {}'.format(get.return_rule)
                    else:
                        if get.return_rule[0] != '/':
                            return public.returnMsg(False, '响应资源应使用URI路径或HTTP状态码，如：/test.png 或 404')
                        return_rule = 'rewrite /.* {} break'.format(get.return_rule)
                rconf = '#SECURITY-START 防盗链配置\n    location ~ .*\\.(%s)$\n    {\n        expires      30d;\n        access_log /dev/null;\n        valid_referers %s;\n        if ($invalid_referer){\n           %s;\n        }\n    }\n    #SECURITY-END\n    include enable-php-' % (get.fix.strip().replace(',', '|'), get.domains.strip().replace(',', ' '), return_rule)
                conf = re.sub('include\\s+enable-php-', rconf, conf)
                public.WriteLog('网站管理', '站点[' + get.name + ']已开启防盗链!')
            public.writeFile(file, conf)
        file = '/www/server/panel/vhost/apache/' + get.name + '.conf'
        if os.path.exists(file):
            conf = public.readFile(file)
            if get.status == '1':
                r_key = '#SECURITY-START 防盗链配置\n    RewriteEngine on\n    RewriteCond %{HTTP_REFERER} !^$ [NC]\n'
                d_key = '#SECURITY-START 防盗链配置\n    RewriteEngine on\n'
                if conf.find(r_key) == -1:
                    conf = conf.replace(d_key, r_key)
                else:
                    if conf.find('SECURITY-START') == -1:
                        return public.returnMsg(False, '请先开启防盗链!')
                    conf = conf.replace(r_key, d_key)
            elif conf.find('SECURITY-START') != -1:
                rep = '#SECURITY-START(\n|.){1,500}#SECURITY-END\n'
                conf = re.sub(rep, '', conf)
            else:
                return_rule = '/404.html [R=404,NC,L]'
                if 'return_rule' in get:
                    get.return_rule = get.return_rule.strip()
                    if get.return_rule in ['404', '403', '200', '301', '302', '401', '201']:
                        return_rule = '/{s}.html [R={s},NC,L]'.format(s=get.return_rule)
                    else:
                        if get.return_rule[0] != '/':
                            return public.returnMsg(False, '响应资源应使用URI路径或HTTP状态码，如：/test.png 或 404')
                        return_rule = '{}'.format(get.return_rule)
                tmp = '    RewriteCond %{HTTP_REFERER} !{DOMAIN} [NC]'
                tmps = []
                for d in get.domains.split(','):
                    tmps.append(tmp.replace('{DOMAIN}', d))
                domains = '\n'.join(tmps)
                rconf = 'combined\n    #SECURITY-START 防盗链配置\n    RewriteEngine on\n' + domains + '\n    RewriteRule .(' + get.fix.strip().replace(',', '|') + ') ' + return_rule + '\n    #SECURITY-END'
                conf = conf.replace('combined', rconf)
            public.writeFile(file, conf)
        cond_dir = '/www/server/panel/vhost/openlitespeed/prevent_hotlink/'
        if not os.path.exists(cond_dir):
            os.makedirs(cond_dir)
        file = cond_dir + get.name + '.conf'
        if get.status == '1':
            conf = '\nRewriteCond %{HTTP_REFERER} !^$\nRewriteCond %{HTTP_REFERER} !BTDOMAIN_NAME [NC]\nRewriteRule \\.(BTPFILE)$    /404.html   [R,NC]\n'
            conf = conf.replace('BTDOMAIN_NAME', get.domains.replace(',', ' ')).replace('BTPFILE', get.fix.replace(',', '|'))
        else:
            conf = '\nRewriteCond %{HTTP_REFERER} !BTDOMAIN_NAME [NC]\nRewriteRule \\.(BTPFILE)$    /404.html   [R,NC]\n'
            conf = conf.replace('BTDOMAIN_NAME', get.domains.replace(',', ' ')).replace('BTPFILE', get.fix.replace(',', '|'))
        public.writeFile(file, conf)
        if get.status == 'false':
            public.ExecShell('rm -f {}'.format(file))
        public.serviceReload()
        return public.returnMsg(True, 'SET_SUCCESS')

    def GetSiteLogs(self, get):
        if False:
            return 10
        serverType = public.get_webserver()
        if serverType == 'nginx':
            logPath = '/www/wwwlogs/' + get.siteName + '.log'
        elif serverType == 'apache':
            logPath = '/www/wwwlogs/' + get.siteName + '-access_log'
        else:
            logPath = '/www/wwwlogs/' + get.siteName + '_ols.access_log'
        if not os.path.exists(logPath):
            return public.returnMsg(False, '日志为空')
        return public.returnMsg(True, public.GetNumLines(logPath, 1000))

    def get_site_errlog(self, get):
        if False:
            i = 10
            return i + 15
        serverType = public.get_webserver()
        if serverType == 'nginx':
            logPath = '/www/wwwlogs/' + get.siteName + '.error.log'
        elif serverType == 'apache':
            logPath = '/www/wwwlogs/' + get.siteName + '-error_log'
        else:
            logPath = '/www/wwwlogs/' + get.siteName + '_ols.error_log'
        if not os.path.exists(logPath):
            return public.returnMsg(False, '日志为空')
        return public.returnMsg(True, public.GetNumLines(logPath, 1000))

    def get_site_types(self, get):
        if False:
            return 10
        data = public.M('site_types').field('id,name').order('id asc').select()
        data.insert(0, {'id': 0, 'name': '默认分类'})
        return data

    def add_site_type(self, get):
        if False:
            while True:
                i = 10
        get.name = get.name.strip()
        if not get.name:
            return public.returnMsg(False, '分类名称不能为空')
        if len(get.name) > 16:
            return public.returnMsg(False, '分类名称长度不能超过16位')
        type_sql = public.M('site_types')
        if type_sql.count() >= 10:
            return public.returnMsg(False, '最多添加10个分类!')
        if type_sql.where('name=?', (get.name,)).count() > 0:
            return public.returnMsg(False, '指定分类名称已存在!')
        type_sql.add('name', (get.name,))
        return public.returnMsg(True, '添加成功!')

    def remove_site_type(self, get):
        if False:
            i = 10
            return i + 15
        type_sql = public.M('site_types')
        if type_sql.where('id=?', (get.id,)).count() == 0:
            return public.returnMsg(False, '指定分类不存在!')
        type_sql.where('id=?', (get.id,)).delete()
        public.M('sites').where('type_id=?', (get.id,)).save('type_id', (0,))
        return public.returnMsg(True, '分类已删除!')

    def modify_site_type_name(self, get):
        if False:
            for i in range(10):
                print('nop')
        get.name = get.name.strip()
        if not get.name:
            return public.returnMsg(False, '分类名称不能为空')
        if len(get.name) > 16:
            return public.returnMsg(False, '分类名称长度不能超过16位')
        type_sql = public.M('site_types')
        if type_sql.where('id=?', (get.id,)).count() == 0:
            return public.returnMsg(False, '指定分类不存在!')
        if type_sql.where('name=? AND id!=?', (get.name, get.id)).count() > 0:
            return public.returnMsg(False, '指定分类名称已存在!')
        type_sql.where('id=?', (get.id,)).setField('name', get.name)
        return public.returnMsg(True, '修改成功!')

    def set_site_type(self, get):
        if False:
            i = 10
            return i + 15
        site_ids = json.loads(get.site_ids)
        site_sql = public.M('sites')
        for s_id in site_ids:
            site_sql.where('id=?', (s_id,)).setField('type_id', get.id)
        return public.returnMsg(True, '设置成功!')

    def set_dir_auth(self, get):
        if False:
            i = 10
            return i + 15
        sd = site_dir_auth.SiteDirAuth()
        return sd.set_dir_auth(get)

    def delete_dir_auth_multiple(self, get):
        if False:
            for i in range(10):
                print('nop')
        '\n            @name 批量目录保护\n            @author zhwen<2020-11-17>\n            @param site_id 1\n            @param names test,baohu\n        '
        names = get.names.split(',')
        del_successfully = []
        del_failed = {}
        for name in names:
            get.name = name
            get.id = get.site_id
            try:
                get.multiple = 1
                result = self.delete_dir_auth(get)
                if not result['status']:
                    del_failed[name] = result['msg']
                    continue
                del_successfully.append(name)
            except:
                del_failed[name] = '删除时错误了，请再试一次'
        public.serviceReload()
        return {'status': True, 'msg': '删除目录保护 [ {} ] 成功'.format(','.join(del_successfully)), 'error': del_failed, 'success': del_successfully}

    def delete_dir_auth(self, get):
        if False:
            for i in range(10):
                print('nop')
        sd = site_dir_auth.SiteDirAuth()
        return sd.delete_dir_auth(get)

    def get_dir_auth(self, get):
        if False:
            print('Hello World!')
        sd = site_dir_auth.SiteDirAuth()
        return sd.get_dir_auth(get)

    def modify_dir_auth_pass(self, get):
        if False:
            print('Hello World!')
        sd = site_dir_auth.SiteDirAuth()
        return sd.modify_dir_auth_pass(get)

    def _check_path_total(self, path, limit):
        if False:
            for i in range(10):
                print('nop')
        '\n        根据路径获取文件/目录大小\n        @path 文件或者目录路径\n        return int \n        '
        if not os.path.exists(path):
            return 0
        if not os.path.isdir(path):
            return os.path.getsize(path)
        size_total = 0
        for nf in os.walk(path):
            for f in nf[2]:
                filename = nf[0] + '/' + f
                if not os.path.exists(filename):
                    continue
                if os.path.islink(filename):
                    continue
                size_total += os.path.getsize(filename)
                if size_total >= limit:
                    return limit
        return size_total

    def get_average_num(self, slist):
        if False:
            for i in range(10):
                print('nop')
        '\n        @获取平均值\n        '
        count = len(slist)
        limit_size = 1 * 1024 * 1024
        if count <= 0:
            return limit_size
        print(slist)
        if len(slist) > 1:
            slist = sorted(slist)
            limit_size = int((slist[0] + slist[-1]) / 2 * 0.85)
        return limit_size

    def check_del_data(self, get):
        if False:
            for i in range(10):
                print('nop')
        '\n        @删除前置检测\n        @ids = [1,2,3]\n        '
        ids = json.loads(get['ids'])
        slist = {}
        result = []
        import database
        db_data = database.database().get_database_size(ids, True)
        limit_size = 50 * 1024 * 1024
        f_list_size = []
        db_list_size = []
        for id in ids:
            data = public.M('sites').where('id=?', (id,)).field('id,name,path,addtime').find()
            if not data:
                continue
            addtime = public.to_date(times=data['addtime'])
            data['st_time'] = addtime
            data['limit'] = False
            data['backup_count'] = public.M('backup').where('pid=? AND type=?', (data['id'], '0')).count()
            f_size = self._check_path_total(data['path'], limit_size)
            data['total'] = f_size
            data['score'] = 0
            if f_size > 0:
                f_list_size.append(f_size)
                if f_size > 10 * 1024:
                    data['score'] = int(time.time() - addtime) + f_size
            if data['total'] >= limit_size:
                data['limit'] = True
            data['database'] = False
            find = public.M('databases').field('id,pid,name,ps,addtime').where('pid=?', (data['id'],)).find()
            if find:
                db_addtime = public.to_date(times=find['addtime'])
                data['database'] = db_data[find['name']]
                data['database']['st_time'] = db_addtime
                db_score = 0
                db_size = data['database']['total']
                if db_size > 0:
                    db_list_size.append(db_size)
                    if db_size > 50 * 1024:
                        db_score += int(time.time() - db_addtime) + db_size
                data['score'] += db_score
            result.append(data)
        slist['data'] = sorted(result, key=lambda x: x['score'], reverse=True)
        slist['file_size'] = self.get_average_num(f_list_size)
        slist['db_size'] = self.get_average_num(db_list_size)
        return slist

    def get_https_mode(self, get=None):
        if False:
            i = 10
            return i + 15
        '\n            @name 获取https模式\n            @author hwliang<2022-01-14>\n            @return bool False.宽松模式 True.严格模式\n        '
        web_server = public.get_webserver()
        if web_server not in ['nginx', 'apache']:
            return False
        if web_server == 'nginx':
            default_conf_file = '{}/nginx/0.default.conf'.format(public.get_vhost_path())
        else:
            default_conf_file = '{}/apache/0.default.conf'.format(public.get_vhost_path())
        if not os.path.exists(default_conf_file):
            return False
        default_conf = public.readFile(default_conf_file)
        if not default_conf:
            return False
        if default_conf.find('DEFAULT SSL CONFI') != -1:
            return True
        return False

    def write_ngx_default_conf_by_ssl(self):
        if False:
            i = 10
            return i + 15
        '\n            @name 写nginx默认配置文件（含SSL配置）\n            @author hwliang<2022-01-14>\n            @return bool\n        '
        default_conf_body = 'server\n{\n    listen 80;\n    listen 443 ssl;\n    server_name _;\n    index index.html;\n    root /www/server/nginx/html;\n    \n    # DEFAULT SSL CONFIG\n    ssl_certificate    /www/server/panel/vhost/cert/0.default/fullchain.pem;\n    ssl_certificate_key    /www/server/panel/vhost/cert/0.default/privkey.pem;\n    ssl_protocols TLSv1.2 TLSv1.3;\n    ssl_ciphers EECDH+CHACHA20:EECDH+CHACHA20-draft:EECDH+AES128:RSA+AES128:EECDH+AES256:RSA+AES256:EECDH+3DES:RSA+3DES:!MD5;\n    ssl_prefer_server_ciphers off;\n    ssl_session_cache shared:SSL:10m;\n    ssl_session_timeout 10m;\n    add_header Strict-Transport-Security "max-age=31536000";\n}'
        ngx_default_conf_file = '{}/nginx/0.default.conf'.format(public.get_vhost_path())
        self.create_default_cert()
        return public.writeFile(ngx_default_conf_file, default_conf_body)

    def write_ngx_default_conf(self):
        if False:
            for i in range(10):
                print('nop')
        '\n            @name 写nginx默认配置文件\n            @author hwliang<2022-01-14>\n            @return bool\n        '
        default_conf_body = 'server\n{\n    listen 80;\n    server_name _;\n    index index.html;\n    root /www/server/nginx/html;\n}'
        ngx_default_conf_file = '{}/nginx/0.default.conf'.format(public.get_vhost_path())
        return public.writeFile(ngx_default_conf_file, default_conf_body)

    def write_apa_default_conf_by_ssl(self):
        if False:
            print('Hello World!')
        '\n            @name 写nginx默认配置文件（含SSL配置）\n            @author hwliang<2022-01-14>\n            @return bool\n        '
        default_conf_body = '<VirtualHost *:80>\n    ServerAdmin webmaster@example.com\n    DocumentRoot "/www/server/apache/htdocs"\n    ServerName bt.default.com\n    <Directory "/www/server/apache/htdocs">\n        SetOutputFilter DEFLATE\n        Options FollowSymLinks\n        AllowOverride All\n        Order allow,deny\n        Allow from all\n        DirectoryIndex index.html\n    </Directory>\n</VirtualHost>\n<VirtualHost *:443>\n    ServerAdmin webmaster@example.com\n    DocumentRoot "/www/server/apache/htdocs"\n    ServerName ssl.default.com\n    \n    # DEFAULT SSL CONFIG\n    SSLEngine On\n    SSLCertificateFile /www/server/panel/vhost/cert/0.default/fullchain.pem\n    SSLCertificateKeyFile /www/server/panel/vhost/cert/0.default/privkey.pem\n    SSLCipherSuite EECDH+CHACHA20:EECDH+CHACHA20-draft:EECDH+AES128:RSA+AES128:EECDH+AES256:RSA+AES256:EECDH+3DES:RSA+3DES:!MD5\n    SSLProtocol All -SSLv2 -SSLv3 -TLSv1\n    SSLHonorCipherOrder On\n    \n    <Directory "/www/server/apache/htdocs">\n        SetOutputFilter DEFLATE\n        Options FollowSymLinks\n        AllowOverride All\n        Order allow,deny\n        Allow from all\n        DirectoryIndex index.html\n    </Directory>\n</VirtualHost>'
        apa_default_conf_file = '{}/apache/0.default.conf'.format(public.get_vhost_path())
        self.create_default_cert()
        return public.writeFile(apa_default_conf_file, default_conf_body)

    def write_apa_default_conf(self):
        if False:
            i = 10
            return i + 15
        '\n            @name 写apache默认配置文件\n            @author hwliang<2022-01-14>\n            @return bool\n        '
        default_conf_body = '<VirtualHost *:80>\n    ServerAdmin webmaster@example.com\n    DocumentRoot "/www/server/apache/htdocs"\n    ServerName bt.default.com\n    <Directory "/www/server/apache/htdocs">\n        SetOutputFilter DEFLATE\n        Options FollowSymLinks\n        AllowOverride All\n        Order allow,deny\n        Allow from all\n        DirectoryIndex index.html\n    </Directory>\n</VirtualHost>'
        apa_default_conf_file = '{}/apache/0.default.conf'.format(public.get_vhost_path())
        return public.writeFile(apa_default_conf_file, default_conf_body)

    def set_https_mode(self, get=None):
        if False:
            for i in range(10):
                print('nop')
        '\n            @name 设置https模式\n            @author hwliang<2022-01-14>\n            @return dict\n        '
        web_server = public.get_webserver()
        if web_server not in ['nginx', 'apache']:
            return public.returnMsg(False, '该功能只支持Nginx/Apache')
        ngx_default_conf_file = '{}/nginx/0.default.conf'.format(public.get_vhost_path())
        apa_default_conf_file = '{}/apache/0.default.conf'.format(public.get_vhost_path())
        ngx_default_conf = public.readFile(ngx_default_conf_file)
        apa_default_conf = public.readFile(apa_default_conf_file)
        status = False
        if ngx_default_conf:
            if ngx_default_conf.find('DEFAULT SSL CONFIG') != -1:
                status = False
                self.write_ngx_default_conf()
                self.write_apa_default_conf()
            else:
                status = True
                self.write_ngx_default_conf_by_ssl()
                self.write_apa_default_conf_by_ssl()
        else:
            status = True
            self.write_ngx_default_conf_by_ssl()
            self.write_apa_default_conf_by_ssl()
        public.serviceReload()
        status_msg = {True: '开启', False: '关闭'}
        msg = '已{}HTTPS严格模式'.format(status_msg[status])
        public.WriteLog('网站管理', msg)
        return public.returnMsg(True, msg)

    def create_default_cert(self):
        if False:
            while True:
                i = 10
        '\n            @name 创建默认SSL证书\n            @author hwliang<2022-01-14>\n            @return bool\n        '
        cert_pem = '/www/server/panel/vhost/cert/0.default/fullchain.pem'
        cert_key = '/www/server/panel/vhost/cert/0.default/privkey.pem'
        if os.path.exists(cert_pem) and os.path.exists(cert_key):
            return True
        cert_path = os.path.dirname(cert_pem)
        if not os.path.exists(cert_path):
            os.makedirs(cert_path)
        import OpenSSL
        key = OpenSSL.crypto.PKey()
        key.generate_key(OpenSSL.crypto.TYPE_RSA, 2048)
        cert = OpenSSL.crypto.X509()
        cert.set_serial_number(0)
        cert.set_issuer(cert.get_subject())
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(86400 * 3650)
        cert.set_pubkey(key)
        cert.sign(key, 'md5')
        cert_ca = OpenSSL.crypto.dump_certificate(OpenSSL.crypto.FILETYPE_PEM, cert)
        private_key = OpenSSL.crypto.dump_privatekey(OpenSSL.crypto.FILETYPE_PEM, key)
        if len(cert_ca) > 100 and len(private_key) > 100:
            public.writeFile(cert_pem, cert_ca, 'wb+')
            public.writeFile(cert_key, private_key, 'wb+')
            return True
        return False