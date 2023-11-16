import os, re
_title = 'FTP服务弱口令检测'
_version = 1.0
_ps = '检测已启用的FTP服务弱口令'
_level = 2
_date = '2020-09-19'
_ignore = os.path.exists('data/warning/ignore/sw_ftp_pass.pl')
_tips = ['请到【FTP】页面修改FTP密码', '注意：请不要使用过于简单的帐号密码，以免造成安全隐患', '推荐使用高安全强度的密码：分别包含数字、大小写、特殊字符混合，且长度不少于7位。', '使用【Fail2ban防爆破】插件对FTP服务进行保护']
_help = ''
_topic = 'ftp'

def check_run():
    if False:
        return 10
    '检测FTP弱口令\n\n        @author linxiao<2020-9-19>\n        @return (bool, msg)\n    '
    ftp_list = public.M('ftps').field('name,password,status').select()
    if not ftp_list:
        return (True, '无风险')
    weak_pass_ftp = []
    for ftp_info in ftp_list:
        status = ftp_info['status']
        if status == '0' or status == 0:
            continue
        login_name = ftp_info['name']
        login_pass = ftp_info['password']
        if not is_strong_password(login_pass):
            weak_pass_ftp.append(login_name)
    if weak_pass_ftp:
        return (False, '以下FTP服务密码设置过于简单，存在安全隐患：<br />' + '<br />'.join(weak_pass_ftp))
    return (True, '无风险')

def is_strong_password(password):
    if False:
        print('Hello World!')
    '判断密码复杂度是否安全\n\n    非弱口令标准：长度大于等于7，分别包含数字、小写、大写、特殊字符。\n    @password: 密码文本\n    @return: True/False\n    @author: linxiao<2020-9-19>\n    '
    if len(password) < 7:
        return False
    import re
    digit_reg = '[0-9]'
    lower_case_letters_reg = '[a-z]'
    upper_case_letters_reg = '[A-Z]'
    special_characters_reg = '((?=[\\x21-\\x7e]+)[^A-Za-z0-9])'
    regs = [digit_reg, lower_case_letters_reg, upper_case_letters_reg, special_characters_reg]
    grade = 0
    for reg in regs:
        if re.search(reg, password):
            grade += 1
    if grade == 4 or (grade >= 2 and len(password) >= 9):
        return True
    return False