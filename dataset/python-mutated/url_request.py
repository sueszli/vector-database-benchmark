from requests.packages import urllib3
from requests import get
from requests import post

def get_html(url, submit_cookies):
    if False:
        i = 10
        return i + 15
    header = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36', 'Referer': 'http://ui.ptlogin2.qq.com/cgi-bin/login?appid=549000912&s_url=http://qun.qq.com/member.html'}
    urllib3.disable_warnings()
    html = get(url, cookies=submit_cookies, headers=header, verify=False)
    return html

def post_html(url, submit_cookies, submit_data):
    if False:
        print('Hello World!')
    header = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36', 'Referer': 'https://qun.qq.com/member.html'}
    urllib3.disable_warnings()
    html = post(url, data=submit_data, cookies=submit_cookies, headers=header, verify=False)
    return html