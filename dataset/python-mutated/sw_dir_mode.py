import os, sys, re, public
_title = '关键目录权限检测'
_version = 1.0
_ps = '检测关键目录权限是否正确'
_level = 0
_date = '2020-08-05'
_ignore = os.path.exists('data/warning/ignore/sw_dir_mode.pl')
_tips = ['在【文件】页面，对指定目录或文件设置正确的权限和所有者', '注意1：通过【文件】页面设置目录权限时，请取消【应用到子目录】选项', '注意2：错误的文件权限，不但存在安全风险，还可能导致服务器上的一些软件无法正常工作']
_help = ''

def check_run():
    if False:
        i = 10
        return i + 15
    '\n        @name 开始检测\n        @author hwliang<2020-08-05>\n        @return tuple (status<bool>,msg<string>)\n    '
    dir_list = [['/usr', 755, 'root'], ['/usr/bin', 555, 'root'], ['/usr/sbin', 555, 'root'], ['/usr/lib', 555, 'root'], ['/usr/lib64', 555, 'root'], ['/usr/local', 755, 'root'], ['/etc', 755, 'root'], ['/etc/passwd', 644, 'root'], ['/etc/shadow', 600, 'root'], ['/etc/gshadow', 600, 'root'], ['/etc/cron.deny', 600, 'root'], ['/etc/anacrontab', 600, 'root'], ['/var', 755, 'root'], ['/var/spool', 755, 'root'], ['/var/spool/cron', 700, 'root'], ['/var/spool/cron/root', 600, 'root'], ['/var/spool/cron/crontabs/root', 600, 'root'], ['/www', 755, 'root'], ['/www/server', 755, 'root'], ['/www/wwwroot', 755, 'root'], ['/root', 550, 'root'], ['/mnt', 755, 'root'], ['/home', 755, 'root'], ['/dev', 755, 'root'], ['/opt', 755, 'root'], ['/sys', 555, 'root'], ['/run', 755, 'root'], ['/tmp', 777, 'root']]
    not_mode_list = []
    return (True, '无风险')