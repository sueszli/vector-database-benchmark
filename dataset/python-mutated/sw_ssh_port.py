import os, sys, re, public, json
_title = 'SSH端口安全'
_version = 1.0
_ps = '检测当前服务器的SSH端口是否安全'
_level = 1
_date = '2020-08-04'
_ignore = os.path.exists('data/warning/ignore/sw_ssh_port.pl')
_tips = ['在【安全】页面修改SSH端口，并考虑在【SSH安全管理】中关闭【SSH密码登录】，开启【SSH密钥登录】', '若不需要SSH连接服务，建议在【安全】页面关闭SSH服务', '通过【系统防火墙】插件或在【安全组】修改SSH端口的放行为限定IP，以增强安全性', '使用【Fail2ban防爆破】插件对SSH服务进行保护']
_help = ''

def check_run():
    if False:
        i = 10
        return i + 15
    "\n        @name 开始检测\n        @author hwliang<2020-08-03>\n        @return tuple (status<bool>,msg<string>)\n\n        @example   \n            status, msg = check_run()\n            if status:\n                print('OK')\n            else:\n                print('Warning: {}'.format(msg))\n        \n    "
    file = '/etc/ssh/sshd_config'
    conf = public.readFile(file)
    if not conf:
        conf = ''
    rep = '#*Port\\s+([0-9]+)\\s*\\n'
    tmp1 = re.search(rep, conf)
    port = '22'
    if tmp1:
        port = tmp1.groups(0)[0]
    version = public.readFile('/etc/redhat-release')
    if not version:
        version = public.readFile('/etc/issue').strip().split('\n')[0].replace('\\n', '').replace('\\l', '').strip()
    else:
        version = version.replace('release ', '').replace('Linux', '').replace('(Core)', '').strip()
    if os.path.exists('/usr/bin/apt-get'):
        if os.path.exists('/etc/init.d/sshd'):
            status = public.ExecShell("service sshd status | grep -P '(dead|stop)'|grep -v grep")
        else:
            status = public.ExecShell("service ssh status | grep -P '(dead|stop)'|grep -v grep")
    elif version.find(' 7.') != -1 or version.find(' 8.') != -1 or version.find('Fedora') != -1:
        status = public.ExecShell("systemctl status sshd.service | grep 'dead'|grep -v grep")
    else:
        status = public.ExecShell("/etc/init.d/sshd status | grep -e 'stopped' -e '已停'|grep -v grep")
    fail2ban_file = '/www/server/panel/plugin/fail2ban/config.json'
    if os.path.exists(fail2ban_file):
        try:
            fail2ban_config = json.loads(public.readFile(fail2ban_file))
            if 'sshd' in fail2ban_config.keys():
                if fail2ban_config['sshd']['act'] == 'true':
                    return (True, '已开启Fail2ban防爆破')
        except:
            pass
    if len(status[0]) > 3:
        status = False
    else:
        status = True
    if not status:
        return (True, '未开启SSH服务')
    if port != '22':
        return (True, '已修改默认SSH端口')
    result = public.check_port_stat(int(port), public.GetLocalIp())
    if result == 0:
        return (True, '无风险')
    return (False, '默认SSH端口({})未修改，且未做访问IP限定配置，有SSH暴破风险'.format(port))