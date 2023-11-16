import os, sys, re, public
_title = '网站证书(SSL)'
_version = 1.0
_ps = '检测所有网站是否部署安全证书'
_level = 1
_date = '2020-08-04'
_ignore = os.path.exists('data/warning/ignore/sw_site_ssl.pl')
_tips = ['请考虑为您的网站部署SSL证书，以提升网站的安全性']
_help = ''

def check_run():
    if False:
        while True:
            i = 10
    '\n        @name 开始检测\n        @author hwliang<2020-08-04>\n        @return tuple (status<bool>,msg<string>)\n    '
    site_list = public.M('sites').field('id,name').select()
    not_ssl_list = []
    for site_info in site_list:
        ng_conf_file = '/www/server/panel/vhost/nginx/' + site_info['name'] + '.conf'
        if not os.path.exists(ng_conf_file):
            continue
        s_body = public.readFile(ng_conf_file)
        if not s_body:
            continue
        if s_body.find('ssl_certificate') == -1:
            not_ssl_list.append(site_info['name'])
    if not_ssl_list:
        return (False, '以下站点未部署SSL证书: <br />' + '<br />'.join(not_ssl_list))
    return (True, '无风险')