import os, sys, re, public
_title = '文件回收站检测'
_version = 1.0
_ps = '检测文件回收站是否开启'
_level = 2
_date = '2020-08-05'
_ignore = os.path.exists('data/warning/ignore/sw_files_recycle_bin.pl')
_tips = ['在【文件】页面，【回收站】 - 中开启【文件回收站】功能']
_help = ''

def check_run():
    if False:
        print('Hello World!')
    '\n        @name 开始检测\n        @author hwliang<2020-08-05>\n        @return tuple (status<bool>,msg<string>)\n    '
    if not os.path.exists('/www/server/panel/data/recycle_bin.pl'):
        return (False, '当前未开启【文件回收站】功能，存在文件被误删的情况下无法找回的风险')
    return (True, '无风险')