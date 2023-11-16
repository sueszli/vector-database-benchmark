import os, sys, re, public
_title = '面板端口安全'
_version = 1.0
_ps = '检测当前面板端口是否安全'
_level = 2
_date = '2020-08-03'
_ignore = os.path.exists('data/warning/ignore/sw_panel_port.pl')
_tips = ['请在【设置】页面修改默认面板端口', '注意：有【安全组】的服务器应在【安全组】中提前放行新端口，以防新端口无法打开']
_help = ''

def check_run():
    if False:
        i = 10
        return i + 15
    '\n        @name 开始检测\n        @author hwliang<2020-08-03>\n        @return tuple (status<bool>,msg<string>)\n    '
    port_file = '/www/server/panel/data/port.pl'
    port = public.readFile(port_file)
    if not port:
        return (True, '无安全风险')
    port = int(port)
    if port != 8888:
        return (True, '无安全风险')
    return (False, '面板端口为默认端口({}), 这可能造成不必要的安全风险'.format(port))