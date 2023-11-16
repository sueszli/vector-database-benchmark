import os, sys, re, public, json
_title = 'MySQL端口安全'
_version = 1.0
_ps = '检测当前服务器的MySQL端口是否安全'
_level = 2
_date = '2020-08-03'
_ignore = os.path.exists('data/warning/ignore/sw_mysql_port.pl')
_tips = ['若非必要，在【安全】页面将MySQL端口的放行删除', '通过【系统防火墙】插件修改MySQL端口的放行为限定IP，以增强安全性', '使用【Fail2ban防爆破】插件对MySQL服务进行保护']
_help = ''

def check_run():
    if False:
        i = 10
        return i + 15
    "\n        @name 开始检测\n        @author hwliang<2020-08-03>\n        @return tuple (status<bool>,msg<string>)\n\n        @example   \n            status, msg = check_run()\n            if status:\n                print('OK')\n            else:\n                print('Warning: {}'.format(msg))\n        \n    "
    mycnf_file = '/etc/my.cnf'
    if not os.path.exists(mycnf_file):
        return (True, '未安装MySQL')
    mycnf = public.readFile(mycnf_file)
    port_tmp = re.findall('port\\s*=\\s*(\\d+)', mycnf)
    if not port_tmp:
        return (True, '未安装MySQL')
    if not public.ExecShell('lsof -i :{}'.format(port_tmp[0]))[0]:
        return (True, '未启动MySQL')
    result = public.check_port_stat(int(port_tmp[0]), public.GetLocalIp())
    if result == 0:
        return (True, '无风险')
    fail2ban_file = '/www/server/panel/plugin/fail2ban/config.json'
    if os.path.exists(fail2ban_file):
        try:
            fail2ban_config = json.loads(public.readFile(fail2ban_file))
            if 'mysql' in fail2ban_config.keys():
                if fail2ban_config['mysql']['act'] == 'true':
                    return (True, '已开启Fail2ban防爆破')
        except:
            pass
    return (False, '当前MySQL端口: {}，可被任意服务器访问，这可能导致MySQL被暴力破解，存在安全隐患'.format(port_tmp[0]))