from panelAuth import panelAuth
import public, os, sys, binascii, urllib, json, time, datetime, re
try:
    from BTPanel import cache, session
except:
    pass

class panelSSL:
    __APIURL = public.GetConfigValue('home') + '/api/Auth'
    __APIURL2 = public.GetConfigValue('home') + '/api/Cert'
    __UPATH = 'data/userInfo.json'
    __userInfo = None
    __PDATA = None
    _check_url = None

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pdata = {}
        data = {}
        if os.path.exists(self.__UPATH):
            my_tmp = public.readFile(self.__UPATH)
            if my_tmp:
                try:
                    self.__userInfo = json.loads(my_tmp)
                except:
                    self.__userInfo = {}
            else:
                self.__userInfo = {}
            try:
                if self.__userInfo:
                    pdata['access_key'] = self.__userInfo['access_key']
                    data['secret_key'] = self.__userInfo['secret_key']
            except:
                self.__userInfo = {}
                pdata['access_key'] = 'test'
                data['secret_key'] = '123456'
        else:
            pdata['access_key'] = 'test'
            data['secret_key'] = '123456'
        pdata['data'] = data
        self.__PDATA = pdata

    def GetToken(self, get):
        if False:
            return 10
        rtmp = ''
        data = {}
        data['username'] = get.username
        data['password'] = public.md5(get.password)
        data['serverid'] = panelAuth().get_serverid()
        pdata = {}
        pdata['data'] = self.De_Code(data)
        try:
            rtmp = public.httpPost(self.__APIURL + '/GetToken', pdata)
            result = json.loads(rtmp)
            result['data'] = self.En_Code(result['data'])
            if result['data']:
                result['data']['serverid'] = data['serverid']
                public.writeFile(self.__UPATH, json.dumps(result['data']))
                public.flush_plugin_list()
            del result['data']
            session['focre_cloud'] = True
            return result
        except Exception as ex:
            raise public.error_conn_cloud(str(ex))

    def DelToken(self, get):
        if False:
            i = 10
            return i + 15
        if os.path.exists(self.__UPATH):
            os.remove(self.__UPATH)
        session['focre_cloud'] = True
        return public.returnMsg(True, 'SSL_BTUSER_UN')

    def GetUserInfo(self, get):
        if False:
            i = 10
            return i + 15
        result = {}
        if self.__userInfo:
            userTmp = {}
            userTmp['username'] = self.__userInfo['username'][0:3] + '****' + self.__userInfo['username'][-4:]
            result['status'] = True
            result['msg'] = public.getMsg('SSL_GET_SUCCESS')
            result['data'] = userTmp
        else:
            userTmp = {}
            userTmp['username'] = public.getMsg('SSL_NOT_BTUSER')
            result['status'] = False
            result['msg'] = public.getMsg('SSL_NOT_BTUSER')
            result['data'] = userTmp
        return result

    def get_product_list(self, get):
        if False:
            for i in range(10):
                print('nop')
        p_type = 'dv'
        if 'p_type' in get:
            p_type = get.p_type
        result = self.request('get_product_list?p_type={}'.format(p_type))
        return result

    def get_order_list(self, get):
        if False:
            i = 10
            return i + 15
        result = self.request('get_order_list')
        return result

    def get_order_find(self, get):
        if False:
            while True:
                i = 10
        self.__PDATA['data']['oid'] = get.oid
        result = self.request('get_order_find')
        return result

    def download_cert(self, get):
        if False:
            for i in range(10):
                print('nop')
        self.__PDATA['data']['oid'] = get.oid
        result = self.request('download_cert')
        return result

    def set_cert(self, get):
        if False:
            return 10
        siteName = get.siteName
        certInfo = self.get_order_find(get)
        path = '/www/server/panel/vhost/cert/' + siteName
        if not os.path.exists(path):
            public.ExecShell('mkdir -p ' + path)
        csrpath = path + '/fullchain.pem'
        keypath = path + '/privkey.pem'
        pidpath = path + '/certOrderId'
        other_file = path + '/partnerOrderId'
        if os.path.exists(other_file):
            os.remove(other_file)
        other_file = path + '/README'
        if os.path.exists(other_file):
            os.remove(other_file)
        public.writeFile(keypath, certInfo['privateKey'])
        public.writeFile(csrpath, certInfo['certificate'] + '\n' + certInfo['caCertificate'])
        public.writeFile(pidpath, get.oid)
        import panelSite
        panelSite.panelSite().SetSSLConf(get)
        public.serviceReload()
        return public.returnMsg(True, 'SET_SUCCESS')

    def apply_order_pay(self, args):
        if False:
            return 10
        self.__PDATA['data'] = json.loads(args.pdata)
        result = self.check_ssl_caa(self.__PDATA['data']['domains'])
        if result:
            return result
        result = self.request('apply_cert_order')
        return result

    def check_ssl_caa(self, domains, clist=['sectigo.com', 'digicert.com']):
        if False:
            while True:
                i = 10
        '\n            @name 检查CAA记录是否正确\n            @param domains 域名列表\n            @param clist 正确的记录值关键词\n            @return bool\n        '
        try:
            data = {}
            for x in domains:
                (root, zone) = public.get_root_domain(x)
                ret = public.query_dns(root, 'CAA')
                if ret:
                    slist = []
                    for x in ret:
                        if x['value'] in clist:
                            continue
                        slist.append(x)
                    if len(slist) > 0:
                        data[root] = slist
            if data:
                result = {}
                result['status'] = False
                result['msg'] = 'error:域名的DNS解析中存在CAA记录，请删除后重新申请'
                result['data'] = json.dumps(data)
                return result
        except:
            pass
        return False

    def get_pay_status(self, args):
        if False:
            print('Hello World!')
        self.__PDATA['data']['oid'] = args.oid
        result = self.request('get_pay_status')
        return result

    def apply_order(self, args):
        if False:
            return 10
        self.__PDATA['data']['oid'] = args.oid
        result = self.request('apply_cert')
        if result['status'] == True:
            self.__PDATA['data'] = {}
            result['verify_info'] = self.get_verify_info(args)
        return result

    def get_verify_info(self, args):
        if False:
            return 10
        self.__PDATA['data']['oid'] = args.oid
        verify_info = self.request('get_verify_info')
        is_file_verify = 'fileName' in verify_info
        verify_info['paths'] = []
        verify_info['hosts'] = []
        for domain in verify_info['domains']:
            if is_file_verify:
                siteRunPath = self.get_domain_run_path(domain)
                if not siteRunPath:
                    if domain[:4] == 'www.':
                        domain = domain[:4]
                    verify_info['paths'].append(verify_info['path'].replace('example.com', domain))
                    continue
                verify_path = siteRunPath + '/.well-known/pki-validation'
                if not os.path.exists(verify_path):
                    os.makedirs(verify_path)
                verify_file = verify_path + '/' + verify_info['fileName']
                if os.path.exists(verify_file):
                    continue
                public.writeFile(verify_file, verify_info['content'])
            else:
                if domain[:4] == 'www.':
                    domain = domain[:4]
                verify_info['hosts'].append(verify_info['host'] + '.' + domain)
                if 'auth_to' in args:
                    (root, zone) = public.get_root_domain(domain)
                    res = self.create_dns_record(args['auth_to'], verify_info['host'] + '.' + root, verify_info['value'])
                    print(res)
        return verify_info

    def set_verify_info(self, args):
        if False:
            while True:
                i = 10
        verify_info = self.get_verify_info(args)
        is_file_verify = 'fileName' in verify_info
        verify_info['paths'] = []
        verify_info['hosts'] = []
        for domain in verify_info['domains']:
            if domain[:2] == '*.':
                domain = domain[2:]
            if is_file_verify:
                siteRunPath = self.get_domain_run_path(domain)
                if not siteRunPath:
                    verify_info['paths'].append(verify_info['path'].replace('example.com', domain))
                    continue
                verify_path = siteRunPath + '/.well-known/pki-validation'
                if not os.path.exists(verify_path):
                    os.makedirs(verify_path)
                verify_file = verify_path + '/' + verify_info['fileName']
                if os.path.exists(verify_file):
                    continue
                public.writeFile(verify_file, verify_info['content'])
            else:
                verify_info['hosts'].append(verify_info['host'] + '.' + domain)
                if 'auth_to' in args:
                    (root, zone) = public.get_root_domain(domain)
                    self.create_dns_record(args['auth_to'], verify_info['host'] + '.' + root, verify_info['value'])
        return verify_info

    def get_domain_run_path(self, domain):
        if False:
            i = 10
            return i + 15
        pid = public.M('domain').where('name=?', (domain,)).getField('pid')
        if not pid:
            return False
        return self.get_site_run_path(pid)

    def get_site_run_path(self, pid):
        if False:
            for i in range(10):
                print('nop')
        '\n            @name 获取网站运行目录\n            @author hwliang<2020-08-05>\n            @param pid(int) 网站标识\n            @return string\n        '
        siteInfo = public.M('sites').where('id=?', (pid,)).find()
        siteName = siteInfo['name']
        sitePath = siteInfo['path']
        webserver_type = public.get_webserver()
        setupPath = '/www/server'
        path = None
        if webserver_type == 'nginx':
            filename = setupPath + '/panel/vhost/nginx/' + siteName + '.conf'
            if os.path.exists(filename):
                conf = public.readFile(filename)
                rep = '\\s*root\\s+(.+);'
                tmp1 = re.search(rep, conf)
                if tmp1:
                    path = tmp1.groups()[0]
        elif webserver_type == 'apache':
            filename = setupPath + '/panel/vhost/apache/' + siteName + '.conf'
            if os.path.exists(filename):
                conf = public.readFile(filename)
                rep = '\\s*DocumentRoot\\s*"(.+)"\\s*\\n'
                tmp1 = re.search(rep, conf)
                if tmp1:
                    path = tmp1.groups()[0]
        else:
            filename = setupPath + '/panel/vhost/openlitespeed/' + siteName + '.conf'
            if os.path.exists(filename):
                conf = public.readFile(filename)
                rep = 'vhRoot\\s*(.*)'
                path = re.search(rep, conf)
                if not path:
                    path = None
                else:
                    path = path.groups()[0]
        if not path:
            path = sitePath
        return path

    def check_url_txt(self, args):
        if False:
            return 10
        url = args.url
        content = args.content
        import http_requests
        res = http_requests.get(url, s_type='curl', timeout=6)
        result = res.text
        if not result:
            return 0
        if result.find('11001') != -1 or result.find('curl: (6)') != -1:
            return -1
        if result.find('curl: (7)') != -1 or res.status_code in [403, 401]:
            return -5
        if result.find('Not Found') != -1 or result.find('not found') != -1 or res.status_code in [404]:
            return -2
        if result.find('timed out') != -1:
            return -3
        if result.find('301') != -1 or result.find('302') != -1 or result.find('Redirecting...') != -1 or (res.status_code in [301, 302]):
            return -4
        if result == content:
            return 1
        return 0

    def again_verify(self, args):
        if False:
            print('Hello World!')
        self.__PDATA['data']['oid'] = args.oid
        self.__PDATA['data']['dcvMethod'] = args.dcvMethod
        result = self.request('again_verify')
        return result

    def get_verify_result(self, args):
        if False:
            print('Hello World!')
        self.__PDATA['data']['oid'] = args.oid
        verify_info = self.request('get_verify_result')
        if verify_info['status'] in ['COMPLETE', False]:
            return verify_info
        is_file_verify = 'CNAME_CSR_HASH' != verify_info['data']['dcvList'][0]['dcvMethod']
        verify_info['paths'] = []
        verify_info['hosts'] = []
        if verify_info['data']['application']['status'] == 'ongoing':
            return public.returnMsg(False, '订单出现问题，CA正在人工验证，若24小时内依然出现此提示，请联系宝塔')
        for dinfo in verify_info['data']['dcvList']:
            is_https = dinfo['dcvMethod'] == 'HTTPS_CSR_HASH'
            if is_https:
                is_https = 's'
            else:
                is_https = ''
            domain = dinfo['domainName']
            if domain[:2] == '*.':
                domain = domain[2:]
            dinfo['domainName'] = domain
            if is_file_verify:
                if public.M('sites').where('id=?', (public.M('domain').where('name=?', dinfo['domainName']).getField('pid'),)).getField('project_type') == 'Java':
                    siteRunPath = '/www/wwwroot/java_node_ssl'
                else:
                    siteRunPath = self.get_domain_run_path(domain)
                status = 0
                url = 'http' + is_https + '://' + domain + '/.well-known/pki-validation/' + verify_info['data']['DCVfileName']
                get = public.dict_obj()
                get.url = url
                get.content = verify_info['data']['DCVfileContent']
                status = self.check_url_txt(get)
                verify_info['paths'].append({'url': url, 'status': status})
                if not siteRunPath:
                    continue
                verify_path = siteRunPath + '/.well-known/pki-validation'
                if not os.path.exists(verify_path):
                    os.makedirs(verify_path)
                verify_file = verify_path + '/' + verify_info['data']['DCVfileName']
                if os.path.exists(verify_file):
                    continue
                public.writeFile(verify_file, verify_info['data']['DCVfileContent'])
            else:
                (domain, subb) = public.get_root_domain(domain)
                dinfo['domainName'] = domain
                verify_info['hosts'].append(verify_info['data']['DCVdnsHost'] + '.' + domain)
        return verify_info

    def cancel_cert_order(self, args):
        if False:
            print('Hello World!')
        self.__PDATA['data']['oid'] = args.oid
        result = self.request('cancel_cert_order')
        return result

    def apply_cert_order_pay(self, args):
        if False:
            while True:
                i = 10
        pdata = json.loads(args.pdata)
        self.__PDATA['data'] = pdata
        result = self.request('apply_cert_order_pay')
        return result

    def get_cert_admin(self, get):
        if False:
            print('Hello World!')
        result = self.request('get_cert_admin')
        return result

    def ApplyDVSSL(self, get):
        if False:
            print('Hello World!')
        '\n        申请证书\n        '
        if not 'orgName' in get:
            return public.returnMsg(False, '确实必要参数 orgName')
        if not 'orgPhone' in get:
            return public.returnMsg(False, '确实必要参数 orgPhone')
        if not 'orgPostalCode' in get:
            return public.returnMsg(False, '确实必要参数 orgPostalCode')
        if not 'orgRegion' in get:
            return public.returnMsg(False, '确实必要参数 orgRegion')
        if not 'orgCity' in get:
            return public.returnMsg(False, '确实必要参数 orgCity')
        if not 'orgAddress' in get:
            return public.returnMsg(False, '确实必要参数 orgAddress')
        if not 'orgDivision' in get:
            return public.returnMsg(False, '确实必要参数 orgDivision')
        get.id = public.M('domain').where('name=?', (get.domain,)).getField('pid')
        if hasattr(get, 'siteName'):
            get.path = public.M('sites').where('id=?', (get.id,)).getField('path')
        else:
            get.siteName = public.M('sites').where('id=?', (get.id,)).getField('name')
        if get.domain[:4] == 'www.':
            if not public.M('domain').where('name=? AND pid=?', (get.domain[4:], get.id)).count():
                return public.returnMsg(False, '申请[%s]证书需要验证[%s]请将[%s]绑定并解析到站点!' % (get.domain, get.domain[4:], get.domain[4:]))
        if public.M('sites').where('id=?', (get.id,)).getField('project_type') == 'Java':
            get.path = '/www/wwwroot/java_node_ssl/'
            runPath = ''
        elif public.M('sites').where('id=?', (get.id,)).getField('project_type') == 'Node':
            get.path = public.M('sites').where('id=?', (get.id,)).getField('path')
            runPath = ''
        else:
            runPath = self.GetRunPath(get)
        if runPath != False and runPath != '/':
            get.path += runPath
        authfile = get.path + '/.well-known/pki-validation/fileauth.txt'
        if not self.CheckDomain(get):
            if not os.path.exists(authfile):
                return public.returnMsg(False, '无法写入验证文件: {}'.format(authfile))
            else:
                msg = '无法正确访问验证文件<br><a class="btlink" href="{c_url}" target="_blank">{c_url}</a> <br><br>\n                <p></b>可能的原因：</b></p>\n                1、未正确解析，或解析未生效 [请正确解析域名，或等待解析生效后重试]<br>\n                2、检查是否有设置301/302重定向 [请暂时关闭重定向相关配置]<br>\n                3、检查该网站是否已部署HTTPS并设置强制HTTPS [请暂时关闭强制HTTPS功能]<br>'.format(c_url=self._check_url)
                return public.returnMsg(False, msg)
        action = 'ApplyDVSSL'
        if hasattr(get, 'partnerOrderId'):
            self.__PDATA['data']['partnerOrderId'] = get.partnerOrderId
            action = 'ReDVSSL'
        self.__PDATA['data']['domain'] = get.domain
        self.__PDATA['data']['orgPhone'] = get.orgPhone
        self.__PDATA['data']['orgPostalCode'] = get.orgPostalCode
        self.__PDATA['data']['orgRegion'] = get.orgRegion
        self.__PDATA['data']['orgCity'] = get.orgCity
        self.__PDATA['data']['orgAddress'] = get.orgAddress
        self.__PDATA['data']['orgDivision'] = get.orgDivision
        self.__PDATA['data']['orgName'] = get.orgName
        self.__PDATA['data'] = self.De_Code(self.__PDATA['data'])
        try:
            result = public.httpPost(self.__APIURL + '/' + action, self.__PDATA)
        except Exception as ex:
            raise public.error_conn_cloud(str(ex))
        try:
            result = json.loads(result)
        except:
            return result
        result['data'] = self.En_Code(result['data'])
        try:
            if not 'authPath' in result['data']:
                result['data']['authPath'] = '/.well-known/pki-validation/'
            authfile = get.path + result['data']['authPath'] + result['data']['authKey']
        except:
            authfile = get.path + '/.well-known/pki-validation/' + result['data']['authKey']
        if 'authValue' in result['data']:
            public.writeFile(authfile, result['data']['authValue'])
        return result

    def apply_order_ca(self, args):
        if False:
            while True:
                i = 10
        pdata = json.loads(args.pdata)
        result = self.check_ssl_caa(pdata['domains'])
        if result:
            return result
        self.__PDATA['data'] = pdata
        result = self.request('apply_cert_ca')
        if result['status'] == True:
            self.__PDATA['data'] = {}
            args['oid'] = pdata['oid']
            if 'auth_to' in pdata:
                args['auth_to'] = pdata['auth_to']
            result['verify_info'] = self.get_verify_info(args)
        return result

    def request(self, dname):
        if False:
            for i in range(10):
                print('nop')
        self.__PDATA['data'] = json.dumps(self.__PDATA['data'])
        result = public.returnMsg(False, '请求失败,请稍候重试!')
        try:
            result = public.httpPost(self.__APIURL2 + '/' + dname, self.__PDATA)
        except Exception as ex:
            raise public.error_conn_cloud(str(ex))
        try:
            result = json.loads(result)
        except:
            pass
        return result

    def GetOrderList(self, get):
        if False:
            i = 10
            return i + 15
        if hasattr(get, 'siteName'):
            path = '/etc/letsencrypt/live/' + get.siteName + '/partnerOrderId'
            if os.path.exists(path):
                self.__PDATA['data']['partnerOrderId'] = public.readFile(path)
            else:
                path = '/www/server/panel/vhost/cert/' + get.siteName + '/partnerOrderId'
                if os.path.exists(path):
                    self.__PDATA['data']['partnerOrderId'] = public.readFile(path)
        self.__PDATA['data'] = self.De_Code(self.__PDATA['data'])
        try:
            rs = public.httpPost(self.__APIURL + '/GetSSLList', self.__PDATA)
        except Exception as ex:
            raise public.error_conn_cloud(str(ex))
        try:
            result = json.loads(rs)
        except:
            return public.returnMsg(False, '获取失败，请稍候重试!')
        result['data'] = self.En_Code(result['data'])
        for i in range(len(result['data'])):
            result['data'][i]['endtime'] = self.add_months(result['data'][i]['createTime'], result['data'][i]['validityPeriod'])
        return result

    def add_months(self, dt, months):
        if False:
            while True:
                i = 10
        import calendar
        dt = datetime.datetime.fromtimestamp(dt / 1000)
        month = dt.month - 1 + months
        year = dt.year + month // 12
        month = month % 12 + 1
        day = min(dt.day, calendar.monthrange(year, month)[1])
        return (time.mktime(dt.replace(year=year, month=month, day=day).timetuple()) + 86400) * 1000

    def GetDVSSL(self, get):
        if False:
            while True:
                i = 10
        get.id = public.M('domain').where('name=?', (get.domain,)).getField('pid')
        if hasattr(get, 'siteName'):
            get.path = public.M('sites').where('id=?', (get.id,)).getField('path')
        else:
            get.siteName = public.M('sites').where('id=?', (get.id,)).getField('name')
        if get.domain[:4] == 'www.':
            if not public.M('domain').where('name=? AND pid=?', (get.domain[4:], get.id)).count():
                return public.returnMsg(False, '申请[%s]证书需要验证[%s]请将[%s]绑定并解析到站点!' % (get.domain, get.domain[4:], get.domain[4:]))
        if not self.CheckForceHTTPS(get.siteName):
            return public.returnMsg(False, '当前网站已开启【强制HTTPS】,请先关闭此功能再申请SSL证书!')
        runPath = self.GetRunPath(get)
        if runPath != False and runPath != '/':
            get.path += runPath
        authfile = get.path + '/.well-known/pki-validation/fileauth.txt'
        if not self.CheckDomain(get):
            if not os.path.exists(authfile):
                return public.returnMsg(False, '无法写入验证文件: {}'.format(authfile))
            else:
                msg = '无法正确访问验证文件<br><a class="btlink" href="{c_url}" target="_blank">{c_url}</a> <br><br>\n                <p></b>可能的原因：</b></p>\n                1、未正确解析，或解析未生效 [请正确解析域名，或等待解析生效后重试]<br>\n                2、检查是否有设置301/302重定向 [请暂时关闭重定向相关配置]<br>\n                3、检查该网站是否设置强制HTTPS [请暂时关闭强制HTTPS功能]<br>'.format(c_url=self._check_url)
                return public.returnMsg(False, msg)
        action = 'GetDVSSL'
        if hasattr(get, 'partnerOrderId'):
            self.__PDATA['data']['partnerOrderId'] = get.partnerOrderId
            action = 'ReDVSSL'
        self.__PDATA['data']['domain'] = get.domain
        self.__PDATA['data'] = self.De_Code(self.__PDATA['data'])
        result = public.httpPost(self.__APIURL + '/' + action, self.__PDATA)
        try:
            result = json.loads(result)
        except:
            return result
        result['data'] = self.En_Code(result['data'])
        try:
            if 'authValue' in result['data'].keys():
                public.writeFile(authfile, result['data']['authValue'])
        except:
            try:
                public.writeFile(authfile, result['data']['authValue'])
            except:
                return result
        return result

    def CheckForceHTTPS(self, siteName):
        if False:
            i = 10
            return i + 15
        conf_file = '/www/server/panel/vhost/nginx/{}.conf'.format(siteName)
        if not os.path.exists(conf_file):
            return True
        conf_body = public.readFile(conf_file)
        if not conf_body:
            return True
        if conf_body.find('HTTP_TO_HTTPS_START') != -1:
            return False
        return True

    def GetRunPath(self, get):
        if False:
            while True:
                i = 10
        if hasattr(get, 'siteName'):
            get.id = public.M('sites').where('name=?', (get.siteName,)).getField('id')
        else:
            get.id = public.M('sites').where('path=?', (get.path,)).getField('id')
        if not get.id:
            return False
        import panelSite
        result = panelSite.panelSite().GetSiteRunPath(get)
        return result['runPath']

    def CheckDomain(self, get):
        if False:
            return 10
        try:
            spath = get.path + '/.well-known/pki-validation'
            if not os.path.exists(spath):
                public.ExecShell("mkdir -p '" + spath + "'")
            epass = public.GetRandomString(32)
            public.writeFile(spath + '/fileauth.txt', epass)
            if get.domain[:4] == 'www.':
                get.domain = get.domain[4:]
            import http_requests
            self._check_url = 'http://127.0.0.1/.well-known/pki-validation/fileauth.txt'
            result = http_requests.get(self._check_url, s_type='curl', timeout=6, headers={'host': get.domain}).text
            self.__test = result
            if result == epass:
                return True
            self._check_url = self._check_url.replace('127.0.0.1', get.domain)
            return False
        except:
            self._check_url = self._check_url.replace('127.0.0.1', get.domain)
            return False

    def Completed(self, get):
        if False:
            i = 10
            return i + 15
        self.__PDATA['data']['partnerOrderId'] = get.partnerOrderId
        self.__PDATA['data'] = self.De_Code(self.__PDATA['data'])
        if hasattr(get, 'siteName'):
            get.path = public.M('sites').where('name=?', (get.siteName,)).getField('path')
            if public.M('sites').where('id=?', (public.M('domain').where('name=?', get.siteName).getField('pid'),)).getField('project_type') == 'Java':
                runPath = '/www/wwwroot/java_node_ssl'
            else:
                runPath = self.GetRunPath(get)
            if runPath != False and runPath != '/':
                get.path += runPath
            tmp = public.httpPost(self.__APIURL + '/SyncOrder', self.__PDATA)
            try:
                sslInfo = json.loads(tmp)
            except:
                return public.returnMsg(False, tmp)
            sslInfo['data'] = self.En_Code(sslInfo['data'])
            try:
                if public.M('sites').where('id=?', (public.M('domain').where('name=?', get.siteName).getField('pid'),)).getField('project_type') == 'Java':
                    spath = '/www/wwwroot/java_node_ssl/.well-known/pki-validation'
                else:
                    spath = get.path + '/.well-known/pki-validation'
                if not os.path.exists(spath):
                    public.ExecShell("mkdir -p '" + spath + "'")
                public.writeFile(spath + '/' + sslInfo['data']['authKey'], sslInfo['data']['authValue'])
            except:
                return public.returnMsg(False, 'SSL_CHECK_WRITE_ERR')
        try:
            result = json.loads(public.httpPost(self.__APIURL + '/Completed', self.__PDATA))
            if 'data' in result:
                result['data'] = self.En_Code(result['data'])
        except:
            result = public.returnMsg(True, '检测中..')
        n = 0
        my_ok = False
        while True:
            if n > 5:
                break
            time.sleep(5)
            rRet = json.loads(public.httpPost(self.__APIURL + '/SyncOrder', self.__PDATA))
            n += 1
            rRet['data'] = self.En_Code(rRet['data'])
            try:
                if rRet['data']['stateCode'] == 'COMPLETED':
                    my_ok = True
                    break
            except:
                return public.get_error_info()
        if not my_ok:
            return result
        return rRet

    def SyncOrder(self, get):
        if False:
            print('Hello World!')
        self.__PDATA['data']['partnerOrderId'] = get.partnerOrderId
        self.__PDATA['data'] = self.De_Code(self.__PDATA['data'])
        result = json.loads(public.httpPost(self.__APIURL + '/SyncOrder', self.__PDATA))
        result['data'] = self.En_Code(result['data'])
        return result

    def GetSSLInfo(self, get):
        if False:
            i = 10
            return i + 15
        self.__PDATA['data']['partnerOrderId'] = get.partnerOrderId
        self.__PDATA['data'] = self.De_Code(self.__PDATA['data'])
        time.sleep(3)
        result = json.loads(public.httpPost(self.__APIURL + '/GetSSLInfo', self.__PDATA))
        result['data'] = self.En_Code(result['data'])
        if not 'privateKey' in result['data']:
            return result
        if hasattr(get, 'siteName'):
            try:
                siteName = get.siteName
                path = '/www/server/panel/vhost/cert/' + siteName
                if not os.path.exists(path):
                    public.ExecShell('mkdir -p ' + path)
                csrpath = path + '/fullchain.pem'
                keypath = path + '/privkey.pem'
                pidpath = path + '/partnerOrderId'
                public.ExecShell('rm -f ' + keypath)
                public.ExecShell('rm -f ' + csrpath)
                public.ExecShell('rm -rf ' + path + '-00*')
                public.ExecShell('rm -rf /etc/letsencrypt/archive/' + get.siteName)
                public.ExecShell('rm -rf /etc/letsencrypt/archive/' + get.siteName + '-00*')
                public.ExecShell('rm -f /etc/letsencrypt/renewal/' + get.siteName + '.conf')
                public.ExecShell('rm -f /etc/letsencrypt/renewal/' + get.siteName + '-00*.conf')
                public.ExecShell('rm -f ' + path + '/README')
                public.ExecShell('rm -f ' + path + '/certOrderId')
                public.writeFile(keypath, result['data']['privateKey'])
                public.writeFile(csrpath, result['data']['cert'] + result['data']['certCa'])
                public.writeFile(pidpath, get.partnerOrderId)
                import panelSite
                panelSite.panelSite().SetSSLConf(get)
                public.serviceReload()
                return public.returnMsg(True, 'SET_SUCCESS')
            except:
                return public.returnMsg(False, 'SET_ERROR')
        result['data'] = self.En_Code(result['data'])
        return result

    def SetCertToSite(self, get):
        if False:
            while True:
                i = 10
        try:
            result = self.GetCert(get)
            if not 'privkey' in result:
                return result
            siteName = get.siteName
            path = '/www/server/panel/vhost/cert/' + siteName
            if not os.path.exists(path):
                public.ExecShell('mkdir -p ' + path)
            csrpath = path + '/fullchain.pem'
            keypath = path + '/privkey.pem'
            public.ExecShell('rm -f ' + keypath)
            public.ExecShell('rm -f ' + csrpath)
            public.ExecShell('rm -rf ' + path + '-00*')
            public.ExecShell('rm -rf /etc/letsencrypt/archive/' + get.siteName)
            public.ExecShell('rm -rf /etc/letsencrypt/archive/' + get.siteName + '-00*')
            public.ExecShell('rm -f /etc/letsencrypt/renewal/' + get.siteName + '.conf')
            public.ExecShell('rm -f /etc/letsencrypt/renewal/' + get.siteName + '-00*.conf')
            public.ExecShell('rm -f ' + path + '/README')
            if os.path.exists(path + '/certOrderId'):
                os.remove(path + '/certOrderId')
            public.writeFile(keypath, result['privkey'])
            public.writeFile(csrpath, result['fullchain'])
            import panelSite
            return panelSite.panelSite().SetSSLConf(get)
            public.serviceReload()
            return public.returnMsg(True, 'SET_SUCCESS')
        except Exception as ex:
            return public.returnMsg(False, 'SET_ERROR,' + public.get_error_info())

    def GetCertList(self, get):
        if False:
            print('Hello World!')
        try:
            vpath = '/www/server/panel/vhost/ssl'
            if not os.path.exists(vpath):
                public.ExecShell('mkdir -p ' + vpath)
            data = []
            for d in os.listdir(vpath):
                mpath = vpath + '/' + d + '/info.json'
                if not os.path.exists(mpath):
                    continue
                tmp = public.readFile(mpath)
                if not tmp:
                    continue
                tmp1 = json.loads(tmp)
                data.append(tmp1)
            return data
        except:
            return []

    def RemoveCert(self, get):
        if False:
            while True:
                i = 10
        try:
            vpath = '/www/server/panel/vhost/ssl/' + get.certName.replace('*.', '')
            if not os.path.exists(vpath):
                return public.returnMsg(False, '证书不存在!')
            public.ExecShell('rm -rf ' + vpath)
            return public.returnMsg(True, '证书已删除!')
        except:
            return public.returnMsg(False, '删除失败!')

    def SaveCert(self, get):
        if False:
            for i in range(10):
                print('nop')
        try:
            certInfo = self.GetCertName(get)
            if not certInfo:
                return public.returnMsg(False, '证书解析失败!')
            vpath = '/www/server/panel/vhost/ssl/' + certInfo['subject']
            vpath = vpath.replace('*.', '')
            if not os.path.exists(vpath):
                public.ExecShell('mkdir -p ' + vpath)
            public.writeFile(vpath + '/privkey.pem', public.readFile(get.keyPath))
            public.writeFile(vpath + '/fullchain.pem', public.readFile(get.certPath))
            public.writeFile(vpath + '/info.json', json.dumps(certInfo))
            return public.returnMsg(True, '证书保存成功!')
        except:
            return public.returnMsg(False, '证书保存失败!')

    def GetCert(self, get):
        if False:
            i = 10
            return i + 15
        vpath = os.path.join('/www/server/panel/vhost/ssl', get.certName.replace('*.', ''))
        if not os.path.exists(vpath):
            return public.returnMsg(False, '证书不存在!')
        data = {}
        data['privkey'] = public.readFile(vpath + '/privkey.pem')
        data['fullchain'] = public.readFile(vpath + '/fullchain.pem')
        return data

    def GetCertName(self, get):
        if False:
            print('Hello World!')
        return self.get_cert_init(get.certPath)

    def get_cert_init(self, pem_file):
        if False:
            return 10
        if not os.path.exists(pem_file):
            return None
        try:
            import OpenSSL
            result = {}
            x509 = OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_PEM, public.readFile(pem_file))
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
            result['notAfter'] = self.strf_date(bytes.decode(x509.get_notAfter())[:-1])
            result['notBefore'] = self.strf_date(bytes.decode(x509.get_notBefore())[:-1])
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
            return result
        except:
            return None

    def strf_date(self, sdate):
        if False:
            while True:
                i = 10
        return time.strftime('%Y-%m-%d', time.strptime(sdate, '%Y%m%d%H%M%S'))

    def strfToTime(self, sdate):
        if False:
            for i in range(10):
                print('nop')
        import time
        return time.strftime('%Y-%m-%d', time.strptime(sdate, '%b %d %H:%M:%S %Y %Z'))

    def GetSSLProduct(self, get):
        if False:
            for i in range(10):
                print('nop')
        self.__PDATA['data'] = self.De_Code(self.__PDATA['data'])
        result = json.loads(public.httpPost(self.__APIURL + '/GetSSLProduct', self.__PDATA))
        result['data'] = self.En_Code(result['data'])
        return result

    def De_Code(self, data):
        if False:
            return 10
        if sys.version_info[0] == 2:
            import urllib
            pdata = urllib.urlencode(data)
            return binascii.hexlify(pdata)
        else:
            import urllib.parse
            pdata = urllib.parse.urlencode(data)
            if type(pdata) == str:
                pdata = pdata.encode('utf-8')
            return binascii.hexlify(pdata).decode()

    def En_Code(self, data):
        if False:
            return 10
        if sys.version_info[0] == 2:
            import urllib
            result = urllib.unquote(binascii.unhexlify(data))
        else:
            import urllib.parse
            if type(data) == str:
                data = data.encode('utf-8')
            tmp = binascii.unhexlify(data)
            if type(tmp) != str:
                tmp = tmp.decode('utf-8')
            result = urllib.parse.unquote(tmp)
        if type(result) != str:
            result = result.decode('utf-8')
        return json.loads(result)

    def renew_lets_ssl(self, get):
        if False:
            i = 10
            return i + 15
        if not os.path.exists('vhost/cert/crontab.json'):
            return public.returnMsg(False, '当前没有可以续订的证书!')
        old_list = json.loads(public.ReadFile('vhost/cert/crontab.json'))
        cron_list = old_list
        if hasattr(get, 'siteName'):
            if not get.siteName in old_list:
                return public.returnMsg(False, '当前网站没有可以续订的证书.')
            cron_list = {}
            cron_list[get.siteName] = old_list[get.siteName]
        import panelLets
        lets = panelLets.panelLets()
        result = {}
        result['status'] = True
        result['sucess_list'] = []
        result['err_list'] = []
        for siteName in cron_list:
            data = cron_list[siteName]
            ret = lets.renew_lest_cert(data)
            if ret['status']:
                result['sucess_list'].append(siteName)
            else:
                result['err_list'].append({'siteName': siteName, 'msg': ret['msg']})
        return result

    def renew_cert_order(self, args):
        if False:
            while True:
                i = 10
        '\n            @name 续签商用证书\n            @author cjx\n            @version 1.0\n        '
        pdata = json.loads(args.pdata)
        self.__PDATA['data'] = pdata
        result = self.request('renew_cert_order')
        if result['status'] == True:
            self.__PDATA['data'] = {}
            args['oid'] = result['oid']
            result['verify_info'] = self.get_verify_info(args)
        return result

    def GetAuthToken(self, get):
        if False:
            return 10
        '\n        登录官网获取Token\n        @get.username 官网手机号\n        @get.password 官网账号密码\n        '
        rtmp = ''
        data = {}
        data['username'] = get.username
        data['password'] = public.md5(get.password)
        data['serverid'] = panelAuth().get_serverid()
        if 'code' in get:
            data['code'] = get.code
        if 'token' in get:
            data['token'] = get.token
        pdata = {}
        pdata['data'] = self.De_Code(data)
        try:
            rtmp = public.httpPost(self.__APIURL + '/GetAuthToken', pdata)
            result = json.loads(rtmp)
            result['data'] = self.En_Code(result['data'])
            if not result['status']:
                return result
            if result['data']:
                if result['data']['serverid'] != data['serverid']:
                    public.writeFile('data/sid.pl', result['data']['serverid'])
                public.writeFile(self.__UPATH, json.dumps(result['data']))
                if os.path.exists('data/bind_path.pl'):
                    os.remove('data/bind_path.pl')
                public.flush_plugin_list()
            del result['data']
            session['focre_cloud'] = True
            return result
        except Exception as ex:
            raise public.error_conn_cloud(str(ex))

    def GetBindCode(self, get):
        if False:
            for i in range(10):
                print('nop')
        '\n        获取验证码\n        '
        rtmp = ''
        data = {}
        data['username'] = get.username
        data['token'] = get.token
        pdata = {}
        pdata['data'] = self.De_Code(data)
        try:
            rtmp = public.httpPost(self.__APIURL + '/GetBindCode', pdata)
            result = json.loads(rtmp)
            return result
        except Exception as ex:
            raise public.error_conn_cloud(str(ex))

    def get_dnsapi(self, auth_to):
        if False:
            i = 10
            return i + 15
        tmp = auth_to.split('|')
        dns_name = tmp[0]
        key = 'None'
        secret = 'None'
        if len(tmp) < 3:
            dnsapi_config = json.loads(public.readFile('{}/config/dns_api.json'.format(public.get_panel_path())))
            for dc in dnsapi_config:
                if dc['name'] != dns_name:
                    continue
                if not dc['data']:
                    continue
                key = dc['data'][0]['value']
                secret = dc['data'][1]['value']
        else:
            key = tmp[1]
            secret = tmp[2]
        return (dns_name, key, secret)

    def get_dns_class(self, auth_to):
        if False:
            return 10
        try:
            import panelDnsapi
            (dns_name, key, secret) = self.get_dnsapi(auth_to)
            dns_class = getattr(panelDnsapi, dns_name)(key, secret)
            dns_class._type = 1
            return dns_class
        except:
            return None

    def create_dns_record(self, auth_to, domain, dns_value):
        if False:
            return 10
        if auth_to == 'dns':
            return None
        dns_class = self.get_dns_class(auth_to)
        if not dns_class:
            return public.returnMsg(False, '操作失败，请检查密钥是否正确.')
        (root, zone) = public.get_root_domain(domain)
        try:
            dns_class.remove_record(public.de_punycode(root), '@', 'CAA')
        except:
            pass
        try:
            dns_class.create_dns_record(public.de_punycode(domain), dns_value)
            return public.returnMsg(True, '添加成功')
        except:
            return public.returnMsg(False, public.get_error_info())

    def apply_cert_install_pay(self, args):
        if False:
            return 10
        "\n            @name 单独购买人工安装服务\n            @param args<dict_obj>{\n                'oid'<int> 订单ID\n            }\n        "
        pdata = json.loads(args.pdata)
        self.__PDATA['data'] = pdata
        result = self.request('apply_cert_install_pay')
        return result