from decrypt import hash33_token
from decrypt import hash33_bkn
from decrypt import get_sck
from url_request import get_html
from url_request import post_html
from decrypt import get_csrf_token
import re
import time
from requests import get
from requests import post
from requests.packages import urllib3
from requests.utils import dict_from_cookiejar
from json import loads
import PIL.Image
import PIL.ImageTk
from io import BytesIO
from tkinter_gui import *

class Bot(object):
    """
    QQ机器人对象，用于获取指定QQ号的群信息及群成员信息，
    同时，该接口可获取指定QQ的所有好友分组，但是获取的好友数据仅包含备注名和QQ号
    """

    def __init__(self):
        if False:
            return 10
        self.is_login = False
        self.cookies_merge_dict_in_id_qq_com = {}
        self.cookies_merge_dict_in_qun_qq_com = {}
        self.cookies_merge_dict_in_qzone_qq_com = {}
        self.qq_number = ''
        self.login_id_qq_com()
        self.login_qun_qq_com()
        self.login_qzone_qq_com()
        picture = self.get_profile_picture(self.qq_number, 140)
        BytesIOObj = BytesIO()
        BytesIOObj.write(picture)
        qr_code = PIL.Image.open(BytesIOObj)
        image = PIL.ImageTk.PhotoImage(qr_code)
        image_label['image'] = image

    def login_qun_qq_com(self):
        if False:
            return 10
        login_url = 'https://xui.ptlogin2.qq.com/cgi-bin/xlogin?pt_disable_pwd=1&appid=715030901&daid=73&hide_close_icon=1&pt_no_auth=1&s_url=https%3A%2F%2Fqun.qq.com%2Fmember.html%23'
        html = get_html(login_url, '')
        cookies_back_dict = dict_from_cookiejar(html.cookies)
        pt_login_sig = cookies_back_dict['pt_login_sig']
        self.cookies_merge_dict_in_qun_qq_com.update(cookies_back_dict)
        qrcode_url = 'https://ssl.ptlogin2.qq.com/ptqrshow?appid=715030901&e=2&l=M&s=3&d=72&v=4&t=0.055986512113441966&daid=73&pt_3rd_aid=0'
        html = get_html(qrcode_url, '')
        cookies_back_dict = dict_from_cookiejar(html.cookies)
        qrsig = cookies_back_dict['qrsig']
        ptqrtoken = hash33_token(qrsig)
        self.cookies_merge_dict_in_qun_qq_com.update(cookies_back_dict)
        BytesIOObj = BytesIO()
        BytesIOObj.write(html.content)
        qr_code = PIL.Image.open(BytesIOObj)
        image = PIL.ImageTk.PhotoImage(qr_code)
        image_label['image'] = image
        while True:
            target_url = 'https://ssl.ptlogin2.qq.com/ptqrlogin?u1=https%3A%2F%2Fqun.qq.com%2Fmember.html%23&ptqrtoken={}&ptredirect=1&h=1&t=1&g=1&from_ui=1&ptlang=2052&action=1-5-1647508149682&js_ver=22030810&js_type=1&login_sig={}&pt_uistyle=40&aid=715030901&daid=73&ptdrvs=zE3Wcrd07f4ZnrPfNLLsdLfVh-6bEh5vNlgIR2dcTfep1pxoywDr2Yw03vvhZswWdGxT2OCdJkA_&sid=2353889993957767859&'.format(ptqrtoken, pt_login_sig)
            html = get_html(target_url, self.cookies_merge_dict_in_qun_qq_com)
            if html.status_code:
                if '二维码未失效' in html.text:
                    custom_print(u'(2/3)登录qun.qq.com中，当前二维码未失效，请你扫描二维码进行登录')
                elif '二维码认证' in html.text:
                    custom_print(u'(2/3)登录qun.qq.com中，扫描成功，正在认证中')
                elif '登录成功' in html.text:
                    self.is_login = True
                    custom_print(u'(2/3)登录qun.qq.com中，登录成功')
                    break
                if '二维码已经失效' in html.text:
                    custom_print(u'(2/3)登录qun.qq.com中，当前二维码已失效，请重启本软件')
                    exit()
            time.sleep(2)
        cookies_back_dict = dict_from_cookiejar(html.cookies)
        self.cookies_merge_dict_in_qun_qq_com.update(cookies_back_dict)
        qq_list = re.findall('&uin=(.+?)&service', html.text)
        self.qq_number = qq_list[0]
        startIndex = html.text.find('https')
        endIndex = html.text.find('pt_3rd_aid=0')
        url = html.text[startIndex:endIndex] + 'pt_3rd_aid=0'
        html = get(url, cookies=self.cookies_merge_dict_in_qun_qq_com, allow_redirects=False, verify=False)
        cookies_back_dict = dict_from_cookiejar(html.cookies)
        self.cookies_merge_dict_in_qun_qq_com.update(cookies_back_dict)

    def login_qzone_qq_com(self):
        if False:
            while True:
                i = 10
        login_url = 'https://xui.ptlogin2.qq.com/cgi-bin/xlogin?proxy_url=https://qzs.qq.com/qzone/v6/portal/proxy.html&daid=5&&hide_title_bar=1&low_login=0&qlogin_auto_login=1&no_verifyimg=1&link_target=blank&appid=549000912&style=22&target=self&s_url=https://qzs.qq.com/qzone/v5/loginsucc.html?para=izone&pt_qr_app=手机QQ空间&pt_qr_link=https://z.qzone.com/download.html&self_regurl=https://qzs.qq.com/qzone/v6/reg/index.html&pt_qr_help_link=https://z.qzone.com/download.html&pt_no_auth=0'
        html = get_html(login_url, '')
        cookies_back_dict = dict_from_cookiejar(html.cookies)
        pt_login_sig = cookies_back_dict['pt_login_sig']
        self.cookies_merge_dict_in_qzone_qq_com.update(cookies_back_dict)
        qrcode_url = 'https://ssl.ptlogin2.qq.com/ptqrshow?appid=549000912&e=2&l=M&s=4&d=72&v=4&t=0.0010498811219192827&daid=5&pt_3rd_aid=0'
        html = get_html(qrcode_url, '')
        cookies_back_dict = dict_from_cookiejar(html.cookies)
        qrsig = cookies_back_dict['qrsig']
        ptqrtoken = hash33_token(qrsig)
        self.cookies_merge_dict_in_qzone_qq_com.update(cookies_back_dict)
        BytesIOObj = BytesIO()
        BytesIOObj.write(html.content)
        qr_code = PIL.Image.open(BytesIOObj)
        image = PIL.ImageTk.PhotoImage(qr_code)
        image_label['image'] = image
        while True:
            target_url = 'https://ssl.ptlogin2.qq.com/ptqrlogin?u1=https://qzs.qq.com/qzone/v5/loginsucc.html?para=izone&ptqrtoken=' + str(ptqrtoken) + '&ptredirect=0&h=1&t=1&g=1&from_ui=1&ptlang=2052&action=0-0-1558286321351&js_ver=19042519&js_type=1&login_sig=' + str(pt_login_sig) + '&pt_uistyle=40&aid=549000912&daid=5&'
            html = get_html(target_url, self.cookies_merge_dict_in_qzone_qq_com)
            if html.status_code:
                if '二维码未失效' in html.text:
                    custom_print(u'(3/3)登录qzone.qq.com中，当前二维码未失效，请你扫描二维码进行登录')
                elif '二维码认证' in html.text:
                    custom_print(u'(3/3)登录qzone.qq.com中，扫描成功，正在认证中')
                elif '登录成功' in html.text:
                    self.is_login = True
                    custom_print(u'(3/3)登录qzone.qq.com中，登录成功')
                    break
                if '二维码已经失效' in html.text:
                    custom_print(u'(3/3)登录qzone.qq.com中，当前二维码已失效，请重启本软件')
                    exit()
            time.sleep(2)
        cookies_back_dict = dict_from_cookiejar(html.cookies)
        self.cookies_merge_dict_in_qzone_qq_com.update(cookies_back_dict)
        qq_list = re.findall('&uin=(.+?)&service', html.text)
        self.qq_number = qq_list[0]
        startIndex = html.text.find('http')
        endIndex = html.text.find('pt_3rd_aid=0')
        url = html.text[startIndex:endIndex] + 'pt_3rd_aid=0'
        urllib3.disable_warnings()
        html = get(url, cookies=self.cookies_merge_dict_in_qzone_qq_com, allow_redirects=False, verify=False)
        cookies_back_dict = dict_from_cookiejar(html.cookies)
        self.cookies_merge_dict_in_qzone_qq_com.update(cookies_back_dict)

    def login_id_qq_com(self):
        if False:
            i = 10
            return i + 15
        login_url = 'https://xui.ptlogin2.qq.com/cgi-bin/xlogin?pt_disable_pwd=1&appid=1006102&daid=1&style=23&hide_border=1&proxy_url=https://id.qq.com/login/proxy.html&s_url=https://id.qq.com/index.html'
        html = get_html(login_url, '')
        cookies_back_dict = dict_from_cookiejar(html.cookies)
        pt_login_sig = cookies_back_dict['pt_login_sig']
        self.cookies_merge_dict_in_id_qq_com.update(cookies_back_dict)
        qrcode_url = 'https://ssl.ptlogin2.qq.com/ptqrshow?appid=1006102&e=2&l=M&s=4&d=72&v=4&t=0.10239549811477189&daid=1&pt_3rd_aid=0'
        html = get_html(qrcode_url, '')
        cookies_back_dict = dict_from_cookiejar(html.cookies)
        qrsig = cookies_back_dict['qrsig']
        ptqrtoken = hash33_token(qrsig)
        self.cookies_merge_dict_in_id_qq_com.update(cookies_back_dict)
        BytesIOObj = BytesIO()
        BytesIOObj.write(html.content)
        qr_code = PIL.Image.open(BytesIOObj)
        image = PIL.ImageTk.PhotoImage(qr_code)
        image_label['image'] = image
        while True:
            target_url = 'https://ssl.ptlogin2.qq.com/ptqrlogin?u1=https://id.qq.com/index.html&ptqrtoken=' + str(ptqrtoken) + '&ptredirect=1&h=1&t=1&g=1&from_ui=1&ptlang=2052&action=0-0-1556812236254&js_ver=19042519&js_type=1&login_sig=' + str(pt_login_sig) + '&pt_uistyle=40&aid=1006102&daid=1&'
            html = get_html(target_url, self.cookies_merge_dict_in_id_qq_com)
            if html.status_code:
                if '二维码未失效' in html.text:
                    custom_print(u'(1/3)登录id.qq.com中，当前二维码未失效，请你扫描二维码进行登录')
                elif '二维码认证' in html.text:
                    custom_print(u'(1/3)登录id.qq.com中，扫描成功，正在认证中')
                elif '登录成功' in html.text:
                    self.is_login = True
                    custom_print(u'(1/3)登录id.qq.com中，登录成功')
                    break
                if '二维码已经失效' in html.text:
                    custom_print(u'(1/3)登录id.qq.com中，当前二维码已失效，请重启本软件')
                    exit()
            time.sleep(2)
        self.cookies_merge_dict_in_id_qq_com = dict_from_cookiejar(html.cookies)
        self.cookies_merge_dict_in_id_qq_com.update(cookies_back_dict)
        qq_list = re.findall('&uin=(.+?)&service', html.text)
        self.qq_number = qq_list[0]
        startIndex = html.text.find('http')
        endIndex = html.text.find('pt_3rd_aid=0')
        url = html.text[startIndex:endIndex] + 'pt_3rd_aid=0'
        urllib3.disable_warnings()
        html = get(url, cookies=self.cookies_merge_dict_in_id_qq_com, allow_redirects=False, verify=False)
        cookies_back_dict = dict_from_cookiejar(html.cookies)
        self.cookies_merge_dict_in_id_qq_com.update(cookies_back_dict)

    def get_group(self):
        if False:
            return 10
        bkn = hash33_bkn(self.cookies_merge_dict_in_qun_qq_com['skey'])
        submit_data = {'bkn': bkn}
        html = post_html('https://qun.qq.com/cgi-bin/qun_mgr/get_group_list', self.cookies_merge_dict_in_qun_qq_com, submit_data)
        group_info = loads(html.text)
        return group_info['join']

    def get_members_in_group(self, group_number):
        if False:
            i = 10
            return i + 15
        bkn = hash33_bkn(self.cookies_merge_dict_in_qun_qq_com['skey'])
        url = 'http://qinfo.clt.qq.com/cgi-bin/qun_info/get_members_info_v1?friends=1&name=1&gc=' + str(group_number) + '&bkn=' + str(bkn) + '&src=qinfo_v3'
        html = get_html(url, self.cookies_merge_dict_in_qun_qq_com)
        group_member = loads(html.text)
        return group_member

    def get_all_friends_in_qq(self):
        if False:
            print('Hello World!')
        "\n        # 获取所有qq好友基本信息\n        # bkn由参数skey通过另一个加密函数得到\n        bkn = hash33_bkn(self.cookies_merge_dict_in_qun_qq_com['skey'])\n        submit_data = {'bkn': bkn}\n        html = post_html('https://qun.qq.com/cgi-bin/qun_mgr/get_friend_list', self.cookies_merge_dict_in_qun_qq_com, submit_data)\n        friend_info = loads(html.text)\n        return friend_info['result']\n        "
        return None

    def get_info_in_qq_friend(self, qq_number):
        if False:
            for i in range(10):
                print('nop')
        bkn = hash33_bkn(self.cookies_merge_dict_in_qun_qq_com['skey'])
        submit_data = {'keyword': str(qq_number), 'ldw': str(bkn), 'num': '20', 'page': '0', 'sessionid': '0', 'agerg': '0', 'sex': '0', 'firston': '0', 'video': '0', 'country': '1', 'province': '65535', 'city': '0', 'district': '0', 'hcountry': '1', 'hprovince': '0', 'hcity': '0', 'hdistrict': '0', 'online': '0'}
        header = {'Accept': 'application/json, text/javascript, */*; q=0.01', 'Origin': 'http://find.qq.com', 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36', 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8', 'Referer': 'http://find.qq.com/'}
        urllib3.disable_warnings()
        html = post('http://cgi.find.qq.com/qqfind/buddy/search_v3', data=submit_data, cookies=self.cookies_merge_dict_in_qun_qq_com, headers=header, verify=False)
        friend_info = loads(html.text)
        return friend_info['result']['buddy']['info_list'][0]

    def get_profile_picture(self, qq_number, size=100):
        if False:
            i = 10
            return i + 15
        urllib3.disable_warnings()
        header = {'Accept': 'application/json, text/javascript, */*; q=0.01', 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36', 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8', 'Referer': 'http://find.qq.com/'}
        html = get('http://q1.qlogo.cn/g?b=qq&nk=' + str(qq_number) + '&s=' + str(size), headers=header, verify=False)
        return html.content

    def get_quit_of_group(self):
        if False:
            for i in range(10):
                print('nop')
        bkn = hash33_bkn(self.cookies_merge_dict_in_qun_qq_com['skey'])
        submit_data = {'bkn': str(bkn)}
        header = {'Accept': 'application/json, text/javascript, */*; q=0.01', 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36', 'Content-Type': 'text/plain', 'origin': 'https://huifu.qq.com', 'referer': 'https://huifu.qq.com/recovery/index.html?frag=0'}
        urllib3.disable_warnings()
        html = post('https://huifu.qq.com/cgi-bin/gr_grouplist', data=submit_data, cookies=self.cookies_merge_dict_in_qun_qq_com, headers=header, verify=False)
        result = loads(html.text)
        return result

    def get_delete_friend_in_360day(self):
        if False:
            while True:
                i = 10
        bkn = hash33_bkn(self.cookies_merge_dict_in_qun_qq_com['skey'])
        qq_number = str(self.qq_number)
        skey = str(self.cookies_merge_dict_in_qun_qq_com['skey'])
        url = 'https://proxy.vip.qq.com/cgi-bin/srfentry.fcgi?bkn=' + str(bkn) + '&ts=&g_tk=' + str(bkn) + '&data={"11053":{"iAppId":1,"iKeyType":1,"sClientIp":"","sSessionKey":"' + skey + '","sUin":"' + qq_number + '"}}'
        header = {'Accept': 'application/json, text/javascript, */*; q=0.01', 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36', 'Accept-Encoding': 'gzip, deflate', 'Referer': 'https://huifu.qq.com/recovery/index.html?frag=1', 'Origin': 'https://huifu.qq.com', 'Connection': 'close'}
        urllib3.disable_warnings()
        html = get(url, cookies=self.cookies_merge_dict_in_qun_qq_com, headers=header, verify=False)
        result = loads(html.text)
        delFriendList = result['11053']['data']['delFriendList']
        if len(delFriendList) == 0:
            return []
        qq_number_list = delFriendList['364']['vecUin']
        return qq_number_list

    def is_vip_svip(self):
        if False:
            i = 10
            return i + 15
        bkn = hash33_bkn(self.cookies_merge_dict_in_qun_qq_com['skey'])
        qq_number = str(self.qq_number)
        skey = str(self.cookies_merge_dict_in_qun_qq_com['skey'])
        url = 'https://proxy.vip.qq.com/cgi-bin/srfentry.fcgi?bkn=' + str(bkn) + '&ts=&g_tk=' + str(bkn) + '&data={"11053":{"iAppId":1,"iKeyType":1,"sClientIp":"","sSessionKey":"' + skey + '","sUin":"' + qq_number + '"}}'
        header = {'Accept': 'application/json, text/javascript, */*; q=0.01', 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36', 'Accept-Encoding': 'gzip, deflate', 'Referer': 'https://huifu.qq.com/recovery/index.html?frag=1', 'Origin': 'https://huifu.qq.com', 'Connection': 'close'}
        urllib3.disable_warnings()
        html = get(url, cookies=self.cookies_merge_dict_in_qun_qq_com, headers=header, verify=False)
        result = loads(html.text)
        isSvip = result['11053']['data']['isSvip']
        isVip = result['11053']['data']['isVip']
        return {'isSvip': isSvip, 'isVip': isVip}

    def get_qb(self):
        if False:
            print('Hello World!')
        qq_number = str(self.qq_number)
        skey = str(self.cookies_merge_dict_in_qun_qq_com['skey'])
        url = 'https://api.unipay.qq.com/v1/r/1450000186/wechat_query?cmd=4&pf=vip_m-pay_html5-html5&pfkey=pfkey&from_h5=1&from_https=1&openid=' + qq_number + '&openkey=' + skey + '&session_id=uin&session_type=skey'
        header = {'Accept': 'application/json, text/javascript, */*; q=0.01', 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36', 'Accept-Encoding': 'gzip, deflate', 'Referer': 'https://my.pay.qq.com/account/index.shtml', 'Origin': 'https://my.pay.qq.com', 'Connection': 'close'}
        urllib3.disable_warnings()
        html = get(url, cookies=self.cookies_merge_dict_in_qun_qq_com, headers=header, verify=False)
        result = loads(html.text)
        qb_value = float(result['qb_balance']) / 10
        return qb_value

    def get_pay_for_another(self):
        if False:
            i = 10
            return i + 15
        skey = str(self.cookies_merge_dict_in_qun_qq_com['skey'])
        url = 'https://pay.qq.com/cgi-bin/personal/account_msg.cgi?p=0.6796416908412624&cmd=1&sck=' + get_sck(skey) + '&type=100&showitem=2&per=100&pageno=1&r=0.3177912609760205'
        header = {'Accept': 'application/json, text/javascript, */*; q=0.01', 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36', 'Accept-Encoding': 'gzip, deflate', 'Referer': 'https://pay.qq.com/infocenter/infocenter.shtml?asktype=100', 'Connection': 'keep-alive'}
        urllib3.disable_warnings()
        html = get(url, cookies=self.cookies_merge_dict_in_qun_qq_com, headers=header, verify=False)
        result = loads(html.text)
        return result['resultinfo']['list']

    def get_detail_information(self):
        if False:
            i = 10
            return i + 15
        result = {}
        bkn = hash33_bkn(self.cookies_merge_dict_in_id_qq_com['skey'])
        url = 'https://id.qq.com/cgi-bin/summary?ldw=' + str(bkn)
        header = {'Accept': '*/*', 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36', 'Accept-Encoding': 'gzip, deflate, br', 'accept-language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7', 'Referer': 'https://id.qq.com/home/home.html?ver=10049&', 'Connection': 'keep-alive'}
        urllib3.disable_warnings()
        html = get(url, cookies=self.cookies_merge_dict_in_id_qq_com, headers=header, verify=False)
        html.encoding = 'utf-8'
        result.update(loads(html.text))
        skey = str(self.cookies_merge_dict_in_id_qq_com['skey'])
        g_tk = str(get_csrf_token(skey))
        url = 'https://cgi.vip.qq.com/querygrow/get?r=0.8102122812749504&g_tk=' + g_tk
        header = {'Accept': '*/*', 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36', 'Accept-Encoding': 'gzip, deflate, br', 'accept-language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7', 'Referer': 'https://id.qq.com/level/mylevel.html?ver=10043&', 'Connection': 'keep-alive'}
        urllib3.disable_warnings()
        html = get(url, cookies=self.cookies_merge_dict_in_id_qq_com, headers=header, verify=False)
        html.encoding = 'utf-8'
        result.update(loads(html.text))
        while True:
            url = 'https://id.qq.com/cgi-bin/userinfo?ldw=' + str(bkn)
            header = {'Accept': '*/*', 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36', 'Accept-Encoding': 'gzip, deflate, br', 'accept-language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7', 'Referer': 'https://id.qq.com/myself/myself.html?ver=10045&', 'Connection': 'keep-alive'}
            urllib3.disable_warnings()
            html = get(url, cookies=self.cookies_merge_dict_in_id_qq_com, headers=header, verify=False)
            html.encoding = 'utf-8'
            if html.text != '':
                result.update(loads(html.text))
                break
        data = {}
        data.update({'bind_email': result['bind_email']})
        data.update({'nickname': result['nick']})
        data.update({'age': result['age']})
        data.update({'birthday': str(result['bir_y']) + '/' + str(result['bir_m']) + '/' + str(result['bir_d'])})
        data.update({'last_contact_friend_count': result['chat_count']})
        data.update({'friend_count': result['friend_count']})
        data.update({'group_count': result['group_count']})
        data.update({'remark_friend_count': result['remark_count']})
        data.update({'odd_friend_count': result['odd_count']})
        data.update({'qq_level': result['level']})
        data.update({'qq_level_rank': str(result['level_rank']) + '/' + str(result['friend_count'])})
        data.update({'qq_age': result['qq_age']})
        data.update({'mobile_qq_online_hour': result['iMobileQQOnlineTime']})
        data.update({'no_hide_online_hour': result['iNoHideOnlineTime']})
        data.update({'total_active_day': result['iTotalActiveDay']})
        qq_signature = result['ln'].replace('&nbsp;', ' ')
        return data

    def who_care_about_me(self):
        if False:
            return 10
        bkn = hash33_bkn(self.cookies_merge_dict_in_qzone_qq_com['p_skey'])
        target_url = 'https://user.qzone.qq.com/proxy/domain/r.qzone.qq.com/cgi-bin/tfriend/friend_ship_manager.cgi?uin={}&do=2&rd=0.6629930546880991&fupdate=1&clean=1&g_tk={}&g_tk={}'.format(self.qq_number, bkn, bkn)
        urllib3.disable_warnings()
        html = get_html(target_url, self.cookies_merge_dict_in_qzone_qq_com)
        result_data = html.text.replace('_Callback(', '')
        result_data = result_data[:len(result_data) - 2]
        result_data = loads(result_data)
        result_data = result_data['data']['items_list']
        return result_data

    def i_care_about_who(self):
        if False:
            i = 10
            return i + 15
        bkn = hash33_bkn(self.cookies_merge_dict_in_qzone_qq_com['p_skey'])
        target_url = 'https://user.qzone.qq.com/proxy/domain/r.qzone.qq.com/cgi-bin/tfriend/friend_ship_manager.cgi?uin={}&do=1&rd=0.6629930546880991&fupdate=1&clean=1&g_tk={}&g_tk={}'.format(self.qq_number, bkn, bkn)
        urllib3.disable_warnings()
        html = get_html(target_url, self.cookies_merge_dict_in_qzone_qq_com)
        result_data = html.text.replace('_Callback(', '')
        result_data = result_data[:len(result_data) - 2]
        result_data = loads(result_data)
        result_data = result_data['data']['items_list']
        return result_data

    def qzone_friendship(self, number):
        if False:
            print('Hello World!')
        bkn = hash33_bkn(self.cookies_merge_dict_in_qzone_qq_com['p_skey'])
        urllib3.disable_warnings()
        target_url = 'https://user.qzone.qq.com/' + self.qq_number
        html = get_html(target_url, self.cookies_merge_dict_in_qzone_qq_com)
        qzonetoken = re.findall('{ try{return "(.+?)";', html.text)
        qzonetoken = qzonetoken[0]
        target_url = 'https://user.qzone.qq.com/proxy/domain/r.qzone.qq.com/cgi-bin/friendship/cgi_friendship?activeuin=' + self.qq_number + '&passiveuin=' + str(number) + '&situation=1&isCalendar=1&g_tk=' + str(bkn) + '&qzonetoken=' + str(qzonetoken) + '&g_tk=' + str(bkn)
        urllib3.disable_warnings()
        html = get_html(target_url, self.cookies_merge_dict_in_qzone_qq_com)
        result_data = html.text.replace('_Callback(', '')
        result_data = result_data[:len(result_data) - 2]
        result_data = loads(result_data)
        print(result_data)