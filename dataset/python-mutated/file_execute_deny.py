import public, re, os, json, shutil

class FileExecuteDeny:

    def _init_conf(self, website):
        if False:
            i = 10
            return i + 15
        self.ng_website_conf = '/www/server/panel/vhost/nginx/{}.conf'.format(website)
        self.ap_website_conf = '/www/server/panel/vhost/apache/{}.conf'.format(website)
        self.ols_website_conf = '/www/server/panel/vhost/openlitespeed/detail/{}.conf'.format(website)
        self.webserver = public.get_webserver()

    def get_file_deny(self, args):
        if False:
            while True:
                i = 10
        '\n        # 添加某个网站禁止运行PHP\n        author: zhwen<zhw@bt.cn>\n        :param args: website 网站名 str\n        :return:\n        '
        self._init_conf(args.website)
        if self.webserver == 'nginx':
            data = self._get_nginx_file_deny()
        elif self.webserver == 'apache':
            data = self._get_apache_file_deny()
        else:
            data = self._get_ols_file_deny()
        return data

    def _get_nginx_file_deny(self):
        if False:
            i = 10
            return i + 15
        conf = public.readFile(self.ng_website_conf)
        if not conf:
            return False
        data = re.findall('BEGIN_DENY_.*', conf)
        deny_name = []
        for i in data:
            tmp = i.split('_')
            if len(tmp) > 2:
                deny_name.append('_'.join(tmp[2:]))
            else:
                deny_name.append(tmp[-1])
        result = []
        for i in deny_name:
            reg = '#BEGIN_DENY_{}\n\\s*location\\s*\\~\\*\\s*\\^(.*)\\.\\*.*\\((.*)\\)\\$'.format(i.replace('|', '\\|'))
            deny_directory = re.search(reg, conf).groups()[0]
            deny_suffix = re.search(reg, conf).groups()[1]
            result.append({'name': i, 'dir': deny_directory, 'suffix': deny_suffix})
        return result

    def _get_apache_file_deny(self):
        if False:
            i = 10
            return i + 15
        conf = public.readFile(self.ap_website_conf)
        if not conf:
            return False
        data = re.findall('BEGIN_DENY_.*', conf)
        deny_name = []
        for i in data:
            tmp = i.split('_')
            if len(tmp) > 2:
                deny_name.append('_'.join(tmp[2:]))
            else:
                deny_name.append(tmp[-1])
        result = []
        for i in deny_name:
            reg = '#BEGIN_DENY_{}\n\\s*<Directory\\s*\\~\\s*"(.*)\\.\\*.*\\((.*)\\)\\$'.format(i.replace('|', '\\|'))
            deny_directory = re.search(reg, conf).groups()[0]
            deny_suffix = re.search(reg, conf).groups()[1]
            result.append({'name': i, 'dir': deny_directory, 'suffix': deny_suffix})
        return result

    def _get_ols_file_deny(self):
        if False:
            i = 10
            return i + 15
        conf = public.readFile(self.ols_website_conf)
        if not conf:
            return False
        data = re.findall('BEGIN_DENY_.*', conf)
        deny_name = []
        for i in data:
            tmp = i.split('_')
            if len(tmp) > 2:
                deny_name.append('_'.join(tmp[2:]))
            else:
                deny_name.append(tmp[-1])
        result = []
        for i in deny_name:
            reg = '#BEGIN_DENY_{}\n\\s*rules\\s*RewriteRule\\s*\\^(.*)\\.\\*.*\\((.*)\\)\\$'.format(i.replace('|', '\\|'))
            deny_directory = re.search(reg, conf).groups()[0]
            deny_suffix = re.search(reg, conf).groups()[1]
            result.append({'name': i, 'dir': deny_directory, 'suffix': deny_suffix})
        return result

    def set_file_deny(self, args):
        if False:
            for i in range(10):
                print('nop')
        '\n        # 添加某个网站禁止运行PHP\n        author: zhwen<zhw@bt.cn>\n        :param args: website 网站名 str\n        :param args: deny_name 规则名称 str\n        :param args: suffix 禁止访问的后续名 str\n        :param args: dir 禁止访问的目录 str\n        :param args: deny_name 规则名称\n        :param args: act 操作方法\n        :return:\n        '
        tmp = self._check_args(args)
        if tmp:
            return tmp
        deny_name = args.deny_name
        dir = args.dir
        suffix = args.suffix
        website = args.website
        self._init_conf(website)
        conf = public.readFile(self.ng_website_conf)
        if not conf:
            return False
        data = re.findall('BEGIN_DENY_.*', conf)
        exist_deny_name = [i.split('_')[-1] for i in data]
        if args.act == 'edit':
            if deny_name not in exist_deny_name:
                return public.returnMsg(False, '指定的规则名不存在! [ {} ]'.format(deny_name))
            self.del_file_deny(args)
        elif deny_name in exist_deny_name:
            return public.returnMsg(False, '指定的规则名不存在! [ {} ]'.format(deny_name))
        self._set_nginx_file_deny(deny_name, dir, suffix)
        self._set_apache_file_deny(deny_name, dir, suffix)
        self._set_ols_file_deny(deny_name, dir, suffix)
        public.serviceReload()
        return public.returnMsg(True, '添加成功')

    def _set_nginx_file_deny(self, name, dir=None, suffix=None):
        if False:
            print('Hello World!')
        conf = public.readFile(self.ng_website_conf)
        if not conf:
            return False
        if not dir and (not suffix):
            reg = '\\s*#BEGIN_DENY_{n}\n(.|\n)*#END_DENY_{n}\n'.format(n=name)
            conf = re.sub(reg, '', conf)
        else:
            new = '\n    #BEGIN_DENY_%s\n    location ~* ^%s.*.(%s)$ {\n        deny all;\n    }\n    #END_DENY_%s\n' % (name, dir, suffix, name)
            if '#BEGIN_DENY_{}\n'.format(name) in conf:
                return True
            conf = re.sub('#ERROR-PAGE-END', '#ERROR-PAGE-END' + new, conf)
        public.writeFile(self.ng_website_conf, conf)
        return True

    def _set_apache_file_deny(self, name, dir=None, suffix=None):
        if False:
            while True:
                i = 10
        conf = public.readFile(self.ap_website_conf)
        if not conf:
            return False
        if not dir and (not suffix):
            reg = '\\s*#BEGIN_DENY_{n}\n(.|\n)*#END_DENY_{n}'.format(n=name)
            conf = re.sub(reg, '', conf)
        else:
            new = '\n    #BEGIN_DENY_{n}\n        <Directory ~ "{d}.*\\.({s})$">\n          Order allow,deny\n          Deny from all\n        </Directory>\n    #END_DENY_{n}\n'.format(n=name, d=dir, s=suffix)
            if '#BEGIN_DENY_{}'.format(name) in conf:
                return True
            conf = re.sub('#DENY\\s*FILES', new + '\n    #DENY FILES', conf)
        public.writeFile(self.ap_website_conf, conf)
        return True

    def _set_ols_file_deny(self, name, dir=None, suffix=None):
        if False:
            return 10
        conf = public.readFile(self.ols_website_conf)
        if not conf:
            return False
        if not dir and (not suffix):
            reg = '#BEGIN_DENY_{n}\n(.|\n)*#END_DENY_{n}\\s*'.format(n=name)
            conf = re.sub(reg, '', conf)
        else:
            new = '\n  #BEGIN_DENY_{n}\n    rules                   RewriteRule ^{d}.*\\.({s})$ - [F,L]\n  #END_DENY_{n}\n'.format(n=name, d=dir, s=suffix)
            if '#BEGIN_DENY_{}'.format(name) in conf:
                return True
            conf = re.sub('autoLoadHtaccess\\s*1', 'autoLoadHtaccess        1' + new, conf)
        public.writeFile(self.ols_website_conf, conf)
        return True

    def del_file_deny(self, args):
        if False:
            return 10
        '\n        # 添加某个网站禁止运行PHP\n        author: zhwen<zhw@bt.cn>\n        :param args: website 网站名 str\n        :param args: deny_name 规则名称 str\n        :return:\n        '
        self._init_conf(args.website)
        deny_name = args.deny_name
        self._set_nginx_file_deny(deny_name)
        self._set_apache_file_deny(deny_name)
        self._set_ols_file_deny(deny_name)
        public.serviceReload()
        return public.returnMsg(True, '删除成功')

    def _check_args(self, args):
        if False:
            return 10
        if hasattr(args, 'deny_name'):
            if len(args.deny_name) < 3:
                return public.returnMsg(False, '规则名最少需要输入3个字符串！')
        if hasattr(args, 'suffix'):
            if not args.suffix:
                return public.returnMsg(False, '文件扩展名不可为空！')
        if hasattr(args, 'dir'):
            if not args.dir:
                return public.returnMsg(False, '目录不可为空！')