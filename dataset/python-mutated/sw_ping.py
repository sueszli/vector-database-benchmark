import os, sys, re, public
_title = 'ICMP检测'
_version = 1.0
_ps = '检测是否禁止ICMP协议访问服务器(禁Ping)'
_level = 0
_date = '2020-08-05'
_ignore = os.path.exists('data/warning/ignore/sw_ping.pl')
_tips = ['在【安全】页面中开启【禁Ping】功能', '注意：开启后无法通过ping通服务器IP或域名，请根据实际需求设置']
_help = ''

def check_run():
    if False:
        print('Hello World!')
    '\n        @name 开始检测\n        @author hwliang<2020-08-05>\n        @return tuple (status<bool>,msg<string>)\n    '
    cfile = '/etc/sysctl.conf'
    conf = public.readFile(cfile)
    rep = '#*net\\.ipv4\\.icmp_echo_ignore_all\\s*=\\s*([0-9]+)'
    tmp = re.search(rep, conf)
    if tmp:
        if tmp.groups(0)[0] == '1':
            return (True, '无风险')
    return (False, '当前未开启【禁Ping】功能，存在服务器被ICMP攻击或被扫的风险')