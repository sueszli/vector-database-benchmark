import os, public, json, re, sys, socket, shutil
os.chdir('/www/server/panel')

class panelRedirect:
    setupPath = '/www/server'
    __redirectfile = '/www/server/panel/data/redirect.conf'

    def GetToDomain(self, tourl):
        if False:
            for i in range(10):
                print('nop')
        if tourl:
            rep = 'https?://([\\w\\-\\.]+)'
            tu = re.search(rep, tourl)
            return tu.group(1)

    def GetAllDomain(self, sitename):
        if False:
            for i in range(10):
                print('nop')
        domains = []
        id = public.M('sites').where('name=?', (sitename,)).getField('id')
        tmp = public.M('domain').where('pid=?', (id,)).field('name').select()
        for key in tmp:
            domains.append(key['name'])
        return domains

    def __CheckRepeatDomain(self, get, action):
        if False:
            return 10
        conf_data = self.__read_config(self.__redirectfile)
        repeat = []
        for conf in conf_data:
            if conf['sitename'] == get.sitename:
                if action == 'create':
                    if conf['redirectname'] == get.redirectname:
                        repeat += list(set(conf['redirectdomain']).intersection(set(get.redirectdomain)))
                elif conf['redirectname'] != get.redirectname:
                    repeat += list(set(conf['redirectdomain']).intersection(set(get.redirectdomain)))
        if list(set(repeat)):
            return list(set(repeat))

    def __CheckRepeatPath(self, get):
        if False:
            while True:
                i = 10
        conf_data = self.__read_config(self.__redirectfile)
        repeat = []
        for conf in conf_data:
            if conf['sitename'] == get.sitename and get.redirectpath != '':
                if conf['redirectname'] != get.redirectname and conf['redirectpath'] == get.redirectpath:
                    repeat.append(get.redirectpath)
        if repeat:
            return repeat

    def __CheckRedirectUrl(self, get):
        if False:
            i = 10
            return i + 15
        sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sk.settimeout(0.5)
        rep = '(https?)://([\\w\\.]+):?([\\d]+)?'
        h = re.search(rep, get.tourl).group(1)
        d = re.search(rep, get.tourl).group(2)
        try:
            p = re.search(rep, get.tourl).group(3)
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

    def __calc_md5(self, redirectname):
        if False:
            return 10
        import hashlib
        md5 = hashlib.md5()
        md5.update(redirectname.encode('utf-8'))
        return md5.hexdigest()

    def SetRedirectNginx(self, get):
        if False:
            while True:
                i = 10
        ng_redirectfile = '%s/panel/vhost/nginx/redirect/%s/*.conf' % (self.setupPath, get.sitename)
        ng_file = self.setupPath + '/panel/vhost/nginx/' + get.sitename + '.conf'
        p_conf = self.__read_config(self.__redirectfile)
        if public.get_webserver() == 'nginx':
            shutil.copyfile(ng_file, '/tmp/ng_file_bk.conf')
        if os.path.exists(ng_file):
            ng_conf = public.readFile(ng_file)
            if not p_conf:
                rep = '#SSL-END(\n|.)*\\/redirect\\/.*\\*.conf;'
                ng_conf = re.sub(rep, '#SSL-END', ng_conf)
                public.writeFile(ng_file, ng_conf)
                return
            sitenamelist = []
            for i in p_conf:
                sitenamelist.append(i['sitename'])
            if get.sitename in sitenamelist:
                rep = 'include.*\\/redirect\\/.*\\*.conf;'
                if not re.search(rep, ng_conf):
                    ng_conf = ng_conf.replace('#SSL-END', '#SSL-END\n\t#引用重定向规则，注释后配置的重定向代理将无效\n\t' + 'include ' + ng_redirectfile + ';')
                    public.writeFile(ng_file, ng_conf)
            else:
                rep = '#SSL-END(\n|.)*\\/redirect\\/.*\\*.conf;'
                ng_conf = re.sub(rep, '#SSL-END', ng_conf)
                public.writeFile(ng_file, ng_conf)

    def SetRedirectApache(self, sitename):
        if False:
            print('Hello World!')
        ap_redirectfile = '%s/panel/vhost/apache/redirect/%s/*.conf' % (self.setupPath, sitename)
        ap_file = self.setupPath + '/panel/vhost/apache/' + sitename + '.conf'
        p_conf = public.readFile(self.__redirectfile)
        if public.get_webserver() == 'apache':
            shutil.copyfile(ap_file, '/tmp/ap_file_bk.conf')
        if os.path.exists(ap_file):
            ap_conf = public.readFile(ap_file)
            if p_conf == '[]':
                rep = '\n*#引用重定向规则，注释后配置的重定向代理将无效\n+\\s+IncludeOptiona[\\s\\w\\/\\.\\*]+'
                ap_conf = re.sub(rep, '', ap_conf)
                public.writeFile(ap_file, ap_conf)
                return
            if sitename in p_conf:
                rep = '#引用重定向(\n|.)+IncludeOptional.*\\/redirect\\/.*conf'
                rep1 = 'combined'
                if not re.search(rep, ap_conf):
                    ap_conf = ap_conf.replace(rep1, rep1 + '\n\t#引用重定向规则，注释后配置的重定向代理将无效' + '\n\tIncludeOptional ' + ap_redirectfile)
                    public.writeFile(ap_file, ap_conf)
            else:
                rep = '\n*#引用重定向规则，注释后配置的重定向代理将无效\n+\\s+IncludeOptiona[\\s\\w\\/\\.\\*]+'
                ap_conf = re.sub(rep, '', ap_conf)
                public.writeFile(ap_file, ap_conf)

    def __CheckRedirectStart(self, get, action=''):
        if False:
            print('Hello World!')
        isError = public.checkWebConfig()
        if isError != True:
            return public.returnMsg(False, '配置文件出错请先排查配置')
        if action == 'create':
            if sys.version_info.major < 3:
                if len(get.redirectname) < 3 or len(get.redirectname) > 15:
                    return public.returnMsg(False, '名称必须大于3小于15个字符串')
            elif len(get.redirectname.encode('utf-8')) < 3 or len(get.redirectname.encode('utf-8')) > 15:
                return public.returnMsg(False, '名称必须大于3小于15个字符串')
            if self.__CheckRedirect(get.sitename, get.redirectname):
                return public.returnMsg(False, '指定重定向名称已存在')
        if get.domainorpath == 'domain':
            if not json.loads(get.redirectdomain):
                return public.returnMsg(False, '请选择重定向域名')
        else:
            if not get.redirectpath:
                return public.returnMsg(False, '请输入重定向路径')
            if '/' not in get.redirectpath:
                return public.returnMsg(False, '路径格式不正确，格式为/xxx')
        repeatdomain = self.__CheckRepeatDomain(get, action)
        if repeatdomain:
            return public.returnMsg(False, '重定向域名重复 %s' % repeatdomain)
        repeatpath = self.__CheckRepeatPath(get)
        if repeatpath:
            return public.returnMsg(False, '重定向路径重复 %s' % repeatpath)
        rep = 'http(s)?\\:\\/\\/([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\\.)+([a-zA-Z0-9][a-zA-Z0-9]{0,62})+.?'
        if not re.match(rep, get.tourl):
            return public.returnMsg(False, '目标URL格式不对 %s' + get.tourl)
        if get.domainorpath == 'domain':
            for d in json.loads(get.redirectdomain):
                tu = self.GetToDomain(get.tourl)
                if d == tu:
                    return public.returnMsg(False, '域名 "%s" 和目标域名一致请取消选择' % d)
        if get.domainorpath == 'path':
            domains = self.GetAllDomain(get.sitename)
            rep = 'https?://(.*)'
            tu = re.search(rep, get.tourl).group(1)
            for d in domains:
                ad = '%s%s' % (d, get.redirectpath)
                if tu == ad:
                    return public.returnMsg(False, '"%s" ，目标URL和被重定向路径一致会导致无限重定向！请不要花样作死' % tu)

    def CreateRedirect(self, get):
        if False:
            for i in range(10):
                print('nop')
        if self.__CheckRedirectStart(get, 'create'):
            return self.__CheckRedirectStart(get, 'create')
        redirectconf = self.__read_config(self.__redirectfile)
        redirectconf.append({'sitename': get.sitename, 'redirectname': get.redirectname, 'tourl': get.tourl, 'redirectdomain': json.loads(get.redirectdomain), 'redirectpath': get.redirectpath, 'redirecttype': get.redirecttype, 'type': int(get.type), 'domainorpath': get.domainorpath, 'holdpath': int(get.holdpath)})
        self.__write_config(self.__redirectfile, redirectconf)
        self.SetRedirectNginx(get)
        self.SetRedirectApache(get.sitename)
        self.SetRedirect(get)
        public.serviceReload()
        return public.returnMsg(True, '创建成功')

    def SetRedirect(self, get):
        if False:
            i = 10
            return i + 15
        ng_file = self.setupPath + '/panel/vhost/nginx/' + get.sitename + '.conf'
        ap_file = self.setupPath + '/panel/vhost/apache/' + get.sitename + '.conf'
        p_conf = self.__read_config(self.__redirectfile)
        if int(get.type) == 1:
            domainstr = "\n        if ($host ~ '^%s'){\n            return %s %s%s;\n        }\n"
            pathstr = '\n        rewrite ^%s(.*) %s%s %s;\n'
            rconf = '#REWRITE-START'
            tourl = get.tourl
            if get.domainorpath == 'domain':
                domains = json.loads(get.redirectdomain)
                holdpath = int(get.holdpath)
                if holdpath == 1:
                    for sd in domains:
                        rconf += domainstr % (sd, get.redirecttype, tourl, '$request_uri')
                else:
                    for sd in domains:
                        rconf += domainstr % (sd, get.redirecttype, tourl, '')
            if get.domainorpath == 'path':
                redirectpath = get.redirectpath
                if get.redirecttype == '301':
                    redirecttype = 'permanent'
                else:
                    redirecttype = 'redirect'
                if int(get.holdpath) == 1 and redirecttype == 'permanent':
                    rconf += pathstr % (redirectpath, tourl, '$1', redirecttype)
                elif int(get.holdpath) == 0 and redirecttype == 'permanent':
                    rconf += pathstr % (redirectpath, tourl, '', redirecttype)
                elif int(get.holdpath) == 1 and redirecttype == 'redirect':
                    rconf += pathstr % (redirectpath, tourl, '$1', redirecttype)
                elif int(get.holdpath) == 0 and redirecttype == 'redirect':
                    rconf += pathstr % (redirectpath, tourl, '', redirecttype)
            rconf += '#REWRITE-END'
            nginxrconf = rconf
            domainstr = '\n\t<IfModule mod_rewrite.c>\n\t\tRewriteEngine on\n\t\tRewriteCond %s{HTTP_HOST} ^%s [NC]\n\t\tRewriteRule ^(.*) %s%s [L,R=%s]\n\t</IfModule>\n'
            pathstr = '\n\t<IfModule mod_rewrite.c>\n\t\tRewriteEngine on\n\t\tRewriteRule ^%s(.*) %s%s [L,R=%s]\n\t</IfModule>\n'
            rconf = '#REWRITE-START'
            if get.domainorpath == 'domain':
                domains = json.loads(get.redirectdomain)
                holdpath = int(get.holdpath)
                if holdpath == 1:
                    for sd in domains:
                        rconf += domainstr % ('%', sd, tourl, '$1', get.redirecttype)
                else:
                    for sd in domains:
                        rconf += domainstr % ('%', sd, tourl, '', get.redirecttype)
            if get.domainorpath == 'path':
                holdpath = int(get.holdpath)
                if holdpath == 1:
                    rconf += pathstr % (get.redirectpath, tourl, '$1', get.redirecttype)
                else:
                    rconf += pathstr % (get.redirectpath, tourl, '', get.redirecttype)
            rconf += '#REWRITE-END'
            apacherconf = rconf
            redirectname_md5 = self.__calc_md5(get.redirectname)
            for w in ['nginx', 'apache']:
                redirectfile = '%s/panel/vhost/%s/redirect/%s/%s_%s.conf' % (self.setupPath, w, get.sitename, redirectname_md5, get.sitename)
                redirectdir = '%s/panel/vhost/%s/redirect/%s' % (self.setupPath, w, get.sitename)
                if not os.path.exists(redirectdir):
                    public.ExecShell('mkdir -p %s' % redirectdir)
                if w == 'nginx':
                    public.writeFile(redirectfile, nginxrconf)
                else:
                    public.writeFile(redirectfile, apacherconf)
            isError = public.checkWebConfig()
            if isError != True:
                if public.get_webserver() == 'nginx':
                    shutil.copyfile('/tmp/ng_file_bk.conf', ng_file)
                else:
                    shutil.copyfile('/tmp/ap_file_bk.conf', ap_file)
                for i in range(len(p_conf) - 1, -1, -1):
                    if get.sitename == p_conf[i]['sitename'] and p_conf[i]['redirectname']:
                        del p_conf[i]
                return public.returnMsg(False, 'ERROR: 配置出错<br><a style="color:red;">' + isError.replace('\n', '<br>') + '</a>')
        else:
            redirectname_md5 = self.__calc_md5(get.redirectname)
            redirectfile = '%s/panel/vhost/%s/redirect/%s/%s_%s.conf'
            for w in ['apache', 'nginx']:
                rf = redirectfile % (self.setupPath, w, get.sitename, redirectname_md5, get.sitename)
                if os.path.exists(rf):
                    os.remove(rf)

    def ModifyRedirect(self, get):
        if False:
            for i in range(10):
                print('nop')
        if self.__CheckRedirectStart(get):
            return self.__CheckRedirectStart(get)
        redirectconf = self.__read_config(self.__redirectfile)
        for i in range(len(redirectconf)):
            if redirectconf[i]['redirectname'] == get.redirectname and redirectconf[i]['sitename'] == get.sitename:
                redirectconf[i]['tourl'] = get.tourl
                redirectconf[i]['redirectdomain'] = json.loads(get.redirectdomain)
                redirectconf[i]['redirectpath'] = get.redirectpath
                redirectconf[i]['redirecttype'] = get.redirecttype
                redirectconf[i]['type'] = int(get.type)
                redirectconf[i]['domainorpath'] = get.domainorpath
                redirectconf[i]['holdpath'] = int(get.holdpath)
        self.__write_config(self.__redirectfile, redirectconf)
        self.SetRedirect(get)
        self.SetRedirectNginx(get)
        self.SetRedirectApache(get.sitename)
        public.serviceReload()
        print('修改成功')
        return public.returnMsg(True, '修改成功')

    def del_redirect_multiple(self, get):
        if False:
            print('Hello World!')
        '\n            @name 批量删除重定向\n            @author zhwen<2020-11-21>\n            @param site_id 1\n            @param redirectnames test,baohu\n        '
        redirectnames = get.redirectnames.split(',')
        del_successfully = []
        del_failed = {}
        get.sitename = public.M('sites').where('id=?', (get.site_id,)).getField('name')
        for redirectname in redirectnames:
            get.redirectname = redirectname
            try:
                get.multiple = 1
                result = self.DeleteRedirect(get, multiple=1)
                if not result['status']:
                    del_failed[redirectname] = result['msg']
                    continue
                del_successfully.append(redirectname)
            except:
                del_failed[redirectname] = '删除时出错了，请再试一次'
        public.serviceReload()
        return {'status': True, 'msg': '删除重定向 [ {} ] 成功'.format(','.join(del_successfully)), 'error': del_failed, 'success': del_successfully}

    def DeleteRedirect(self, get, multiple=None):
        if False:
            while True:
                i = 10
        redirectconf = self.__read_config(self.__redirectfile)
        sitename = get.sitename
        redirectname = get.redirectname
        for i in range(len(redirectconf)):
            if redirectconf[i]['sitename'] == sitename and redirectconf[i]['redirectname'] == redirectname:
                proxyname_md5 = self.__calc_md5(redirectconf[i]['redirectname'])
                public.ExecShell('rm -f %s/panel/vhost/nginx/redirect/%s/%s_%s.conf' % (self.setupPath, redirectconf[i]['sitename'], proxyname_md5, redirectconf[i]['sitename']))
                public.ExecShell('rm -f %s/panel/vhost/apache/redirect/%s/%s_%s.conf' % (self.setupPath, redirectconf[i]['sitename'], proxyname_md5, redirectconf[i]['sitename']))
                del redirectconf[i]
                self.__write_config(self.__redirectfile, redirectconf)
                self.SetRedirectNginx(get)
                self.SetRedirectApache(get.sitename)
                if not multiple:
                    public.serviceReload()
                return public.returnMsg(True, '删除成功')

    def GetRedirectList(self, get):
        if False:
            while True:
                i = 10
        redirectconf = self.__read_config(self.__redirectfile)
        sitename = get.sitename
        redirectlist = []
        for i in redirectconf:
            if i['sitename'] == sitename:
                redirectlist.append(i)
        print(redirectlist)
        return redirectlist

    def ClearOldRedirect(self, get):
        if False:
            for i in range(10):
                print('nop')
        for i in ['apache', 'nginx']:
            conf_path = '%s/panel/vhost/%s/%s.conf' % (self.setupPath, i, get.sitename)
            old_conf = public.readFile(conf_path)
            rep = ''
            if i == 'nginx':
                rep += '#301-START\n+[\\s\\w\\:\\/\\.\\;\\$]+#301-END'
            if i == 'apache':
                rep += '#301-START[\n\\<\\>\\w\\.\\s\\^\\*\\$\\/\\[\\]\\(\\)\\:\\,\\=]+#301-END'
            conf = re.sub(rep, '', old_conf)
            public.writeFile(conf_path, conf)
        public.serviceReload()
        return public.returnMsg(False, '旧版本重定向已经清理')

    def GetRedirectFile(self, get):
        if False:
            return 10
        import files
        conf = self.__read_config(self.__redirectfile)
        sitename = get.sitename
        redirectname = get.redirectname
        proxyname_md5 = self.__calc_md5(redirectname)
        if get.webserver == 'openlitespeed':
            get.webserver = 'apache'
        get.path = '%s/panel/vhost/%s/redirect/%s/%s_%s.conf' % (self.setupPath, get.webserver, sitename, proxyname_md5, sitename)
        for i in conf:
            if redirectname == i['redirectname'] and sitename == i['sitename'] and (i['type'] != 1):
                return public.returnMsg(False, '重定向已暂停')
        f = files.files()
        return (f.GetFileBody(get), get.path)

    def SaveRedirectFile(self, get):
        if False:
            return 10
        import files
        f = files.files()
        return f.SaveFileBody(get)

    def __CheckRedirect(self, sitename, redirectname):
        if False:
            while True:
                i = 10
        conf_data = self.__read_config(self.__redirectfile)
        for i in conf_data:
            if i['sitename'] == sitename:
                if i['redirectname'] == redirectname:
                    return i

    def __read_config(self, path):
        if False:
            i = 10
            return i + 15
        if not os.path.exists(path):
            public.writeFile(path, '[]')
        upBody = public.readFile(path)
        if not upBody:
            upBody = '[]'
        return json.loads(upBody)

    def __write_config(self, path, data):
        if False:
            print('Hello World!')
        return public.writeFile(path, json.dumps(data))