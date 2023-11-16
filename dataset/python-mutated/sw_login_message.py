import os, sys, re, public
_title = 'SSH用户登录通知'
_version = 1.0
_ps = '检测是否开启SSH用户登录通知'
_level = 0
_date = '2020-08-05'
_ignore = os.path.exists('data/warning/ignore/sw_login_message.pl')
_tips = ['在【安全】页面，【SSH安全管理】 - 【登录报警】中开启【监控root用户登陆】功能']
_help = ''

def return_bashrc():
    if False:
        print('Hello World!')
    if os.path.exists('/root/.bashrc'):
        return '/root/.bashrc'
    if os.path.exists('/etc/bashrc'):
        return '/etc/bashrc'
    if os.path.exists('/etc/bash.bashrc'):
        return '/etc/bash.bashrc'
    return '/root/.bashrc'

def check_run():
    if False:
        print('Hello World!')
    '\n        @name 开始检测\n        @author hwliang<2020-08-04>\n        @return tuple (status<bool>,msg<string>)\n    '
    data = public.ReadFile(return_bashrc())
    if not data:
        return (True, '无风险')
    if re.search('ssh_security.py login', data):
        return (True, '无风险')
    else:
        return (False, '未配置SSH用户登录通知，无法在第一时间获知服务器是否被非法登录')