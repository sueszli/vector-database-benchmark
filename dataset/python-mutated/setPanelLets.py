import os
os.chdir('/www/server/panel')
import public, db, json

class setPanelLets:
    __vhost_cert_path = '/www/server/panel/vhost/cert/'
    __panel_cert_path = '/www/server/panel/ssl/'
    __tmp_key = ''
    __tmp_cert = ''

    def __init__(self):
        if False:
            return 10
        pass

    def __save_panel_cert(self, cert, key):
        if False:
            return 10
        keyPath = 'ssl/privateKey.pem'
        certPath = 'ssl/certificate.pem'
        checkCert = '/tmp/cert.pl'
        public.writeFile(checkCert, cert)
        if key:
            public.writeFile(keyPath, key)
        if cert:
            public.writeFile(certPath, cert)
        if not public.CheckCert(checkCert):
            return public.returnMsg(False, '证书错误,请检查!')
        public.writeFile('ssl/input.pl', 'True')
        return public.returnMsg(True, '证书已保存!')

    def __check_host_name(self, domain):
        if False:
            while True:
                i = 10
        sql = db.Sql()
        path = sql.table('sites').where('name=?', (domain,)).getField('path')
        return path

    def __create_site_of_panel_lets(self, get):
        if False:
            for i in range(10):
                print('nop')
        import panelSite
        ps = panelSite.panelSite()
        get.webname = json.dumps({'domain': get.domain, 'domainlist': [], 'count': 0})
        get.ps = "用于面板Let's Encrypt 证书申请和续签，请勿删除"
        get.path = '/www/wwwroot/panel_ssl_site'
        get.ftp = 'false'
        get.sql = 'false'
        get.codeing = 'utf8'
        get.type = 'PHP'
        get.version = '00'
        get.type_id = '0'
        get.port = '80'
        psa = ps.AddSite(get)
        if 'status' in psa.keys():
            return psa

    def __create_lets(self, get):
        if False:
            while True:
                i = 10
        from acme_v2 import acme_v2
        site_id = str(public.M('sites').where('name=?', (get.domain,)).getField('id'))
        get.auth_type = 'http'
        get.auth_to = site_id
        get.id = site_id
        get.auto_wildcard = '0'
        get.domains = json.dumps([get.domain])
        get.siteName = get.domain
        p = acme_v2()
        cert_info = p.apply_cert_api(get)
        if 'private_key' not in cert_info:
            return public.returnMsg(False, '申请证书失败,请尝试在网站列表内手动为面板域名申请SSL证书后再到此开启SSL！')
        get.key = cert_info['private_key']
        get.csr = cert_info['cert'] + cert_info['root']
        return public.returnMsg(True, self._deploy_cert(get))

    def _deploy_cert(self, get):
        if False:
            for i in range(10):
                print('nop')
        from panelSite import panelSite
        return panelSite().SetSSL(get)

    def __check_cert_dir(self, get):
        if False:
            while True:
                i = 10
        import panelSSL, time
        pssl = panelSSL.panelSSL()
        gcl = pssl.GetCertList(get)
        for i in gcl:
            if get.domain in i['dns'] or get.domain == i['subject']:
                try:
                    time_stamp = int(i['notAfter'])
                except:
                    time_array = time.strptime(i['notAfter'], '%Y-%m-%d')
                    time_stamp = int(time.mktime(time_array))
                now = time.time()
                if time_stamp > int(now):
                    return i
            for d in i['dns']:
                d = d.split('.')
                if '*' in d and d[1:] == get.domain.split('.')[1:]:
                    try:
                        time_stamp = int(i['notAfter'])
                    except:
                        time_array = time.strptime(i['notAfter'], '%Y-%m-%d')
                        time_stamp = int(time.mktime(time_array))
                    now = time.time()
                    if time_stamp > int(now):
                        return i

    def __read_site_cert(self, domain_cert):
        if False:
            print('Hello World!')
        try:
            key_file = '{path}{domain}/{key}'.format(path=self.__vhost_cert_path, domain=domain_cert['subject'], key='privkey.pem')
            cert_file = '{path}{domain}/{cert}'.format(path=self.__vhost_cert_path, domain=domain_cert['subject'], cert='fullchain.pem')
        except:
            key_file = '/www/server/panel/{}/privkey.pem'.format(domain_cert['save_path'])
            cert_file = '/www/server/panel/{}/fullchain.pem'.format(domain_cert['save_path'])
        if not os.path.exists(key_file):
            key_file = '{path}{domain}/{key}'.format(path='/www/server/panel/vhost/ssl/', domain=domain_cert['subject'], key='privkey.pem')
            cert_file = '{path}{domain}/{cert}'.format(path='/www/server/panel/vhost/ssl/', domain=domain_cert['subject'], cert='fullchain.pem')
        if not os.path.exists(key_file) and '*.' in key_file:
            key_file = key_file.replace('*.', '')
            cert_file = cert_file.replace('*.', '')
        if not os.path.exists(key_file):
            return public.returnMsg(False, 'Can not found the ssl file! {}'.format(key_file))
        self.__tmp_key = public.readFile(key_file)
        self.__tmp_cert = public.readFile(cert_file)

    def __check_panel_cert(self):
        if False:
            while True:
                i = 10
        key = public.readFile(self.__panel_cert_path + 'privateKey.pem')
        cert = public.readFile(self.__panel_cert_path + 'certificate.pem')
        if key and cert:
            return {'key': key, 'cert': cert}

    def __write_panel_cert(self):
        if False:
            for i in range(10):
                print('nop')
        public.writeFile(self.__panel_cert_path + 'privateKey.pem', self.__tmp_key)
        public.writeFile(self.__panel_cert_path + 'certificate.pem', self.__tmp_cert)

    def __save_cert_source(self, domain, email):
        if False:
            print('Hello World!')
        public.writeFile(self.__panel_cert_path + 'lets.info', json.dumps({'domain': domain, 'cert_type': '2', 'email': email}))

    def get_cert_source(self):
        if False:
            print('Hello World!')
        data = public.readFile(self.__panel_cert_path + 'lets.info')
        if not data:
            return {'cert_type': '', 'email': '', 'domain': ''}
        return json.loads(data)

    def __check_panel_domain(self):
        if False:
            i = 10
            return i + 15
        domain = public.readFile('/www/server/panel/data/domain.conf')
        if not domain:
            return False
        return domain.split('\n')[0]

    def check_cert_update(self, sitename):
        if False:
            i = 10
            return i + 15
        from collections import namedtuple
        get = namedtuple('get', ['siteName', 'domain'])
        get.siteName = sitename
        get.domain = sitename
        cert_info = self.__check_cert_dir(get)
        if cert_info:
            return self.copy_cert(cert_info)
        else:
            from panelSite import panelSite
            p_s = panelSite().GetSSL(get)
            if 'msg' in p_s:
                return False
            if not p_s['cert_data']:
                return False
            for i in p_s['cert_data']['dns']:
                if i == sitename:
                    cert_info = {'issuer': p_s['cert_data']['issuer'], 'dns': p_s['cert_data']['dns'], 'notAfter': p_s['cert_data']['notAfter'], 'notBefore': p_s['cert_data']['notBefore'], 'subject': p_s['cert_data']['subject']}
                    return self.copy_cert(cert_info)

    def copy_cert(self, domain_cert):
        if False:
            for i in range(10):
                print('nop')
        res = self.__read_site_cert(domain_cert)
        if res:
            return res
        panel_cert_data = self.__check_panel_cert()
        if not panel_cert_data:
            self.__write_panel_cert()
            return public.returnMsg(True, '1')
        if panel_cert_data['key'] != self.__tmp_key and panel_cert_data['cert'] != self.__tmp_cert:
            self.__write_panel_cert()
            return public.returnMsg(True, '1')
        return public.returnMsg(True, '')

    def set_lets(self, get):
        if False:
            while True:
                i = 10
        '\n        传入参数\n        get.domain  面板域名\n        get.email   管理员email\n        '
        create_site = ''
        domain = self.__check_panel_domain()
        get.domain = domain
        if not domain:
            return public.returnMsg(False, "需要为面板绑定域名后才能申请 Let's Encrypt 证书")
        if not self.__check_host_name(domain):
            create_site = self.__create_site_of_panel_lets(get)
        domain_cert = self.__check_cert_dir(get)
        if domain_cert:
            res = self.copy_cert(domain_cert)
            if not res['status']:
                return res
            public.writeFile('/www/server/panel/data/ssl.pl', 'True')
            self.__save_cert_source(domain, get.email)
            return public.returnMsg(True, '面板lets https设置成功')
        if not create_site:
            create_lets = self.__create_lets(get)
            if not create_lets['status']:
                return create_lets
            if create_lets['msg']:
                domain_cert = self.__check_cert_dir(get)
                self.copy_cert(domain_cert)
                public.writeFile('/www/server/panel/data/ssl.pl', 'True')
                self.__save_cert_source(domain, get.email)
                return public.returnMsg(True, '面板lets https设置成功')
            else:
                return public.returnMsg(False, create_lets)
        else:
            return public.returnMsg(False, create_site)