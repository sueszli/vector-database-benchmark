import os, sys, re, public
_title = '安全入口检测'
_version = 1.0
_ps = '检测当前面板安全入口是否安全'
_level = 3
_date = '2020-08-04'
_ignore = os.path.exists('data/warning/ignore/sw_panel_path.pl')
_tips = ['请在【设置】页面修改安全入口', '或在【设置】页面设置绑定域名，或设置授权IP限制', '注意：请不要设置过于简单的安全入口，这可能导致安全隐患']
_help = ''

def check_run():
    if False:
        print('Hello World!')
    '\n        @name 开始检测\n        @author hwliang<2020-08-03>\n        @return tuple (status<bool>,msg<string>)\n    '
    p_file = '/www/server/panel/data/domain.conf'
    if public.readFile(p_file):
        return (True, '无风险')
    p_file = '/www/server/panel/data/limitip.conf'
    if public.readFile(p_file):
        return (True, '无风险')
    p_file = '/www/server/panel/data/admin_path.pl'
    p_body = public.readFile(p_file)
    if not p_body:
        return (False, '当前未设置安全入口，面板有被扫描的风险')
    p_body = p_body.strip('/').lower()
    if p_body == '':
        return (False, '当前未设置安全入口，面板有被扫描的风险')
    lower_path = ['root', 'admin', '123456', '123', '12', '1234567', '12345', '1234', '12345678', '123456789', 'abc', 'bt']
    if p_body in lower_path:
        return (False, '当前安全入口为：{}，过于简单，存在安全隐患'.format(p_body))
    lower_rule = 'qwertyuiopasdfghjklzxcvbnm1234567890'
    for s in lower_rule:
        for i in range(12):
            if not i:
                continue
            lp = s * i
            if p_body == lp:
                return (False, '当前安全入口为：{}，过于简单，存在安全隐患'.format(p_body))
    return (True, '无风险')