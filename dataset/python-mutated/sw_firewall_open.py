import os, sys, re, public
_title = '系统防火墙检测'
_version = 1.0
_ps = '检测是否开启系统防火墙'
_level = 2
_date = '2020-08-05'
_ignore = os.path.exists('data/warning/ignore/sw_firewall_open.pl')
_tips = ['建议开启系统防火墙，以避免所有服务器端口暴露在互联网上，如服务器有【安全组】功能，请忽略此提示', '注意：开启系统防火墙需提前将需要开放的端口，特别是SSH和面板端口加入放行列表，否则可能导致服务器无法访问']
_help = ''

def check_run():
    if False:
        while True:
            i = 10
    '\n        @name 开始检测\n        @author hwliang<2020-08-04>\n        @return tuple (status<bool>,msg<string>)\n    '
    if os.path.exists('/usr/sbin/firewalld'):
        if public.ExecShell("systemctl status firewalld|grep 'active (running)'")[0]:
            return (True, '无风险')
    elif os.path.exists('/usr/sbin/ufw'):
        if public.ExecShell("ufw status|grep 'Status: active'")[0]:
            return (True, '无风险')
    elif public.ExecShell("service iptables status|grep 'Table: filter'")[0]:
        return (True, '无风险')
    return (False, '未开启系统防火墙，存在安全风险')