from bs4 import BeautifulSoup
from selenium import webdriver
import subprocess as sp
from lxml import etree
import requests
import random
import re
'\n函数说明:获取IP代理\nParameters:\n\tpage - 高匿代理页数,默认获取第一页\nReturns:\n\tproxys_list - 代理列表\nModify:\n\t2017-05-27\n'

def get_proxys(page=1):
    if False:
        return 10
    S = requests.Session()
    target_url = 'http://www.xicidaili.com/nn/%d' % page
    target_headers = {'Upgrade-Insecure-Requests': '1', 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36', 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8', 'Referer': 'http://www.xicidaili.com/nn/', 'Accept-Encoding': 'gzip, deflate, sdch', 'Accept-Language': 'zh-CN,zh;q=0.8'}
    target_response = S.get(url=target_url, headers=target_headers)
    target_response.encoding = 'utf-8'
    target_html = target_response.text
    bf1_ip_list = BeautifulSoup(target_html, 'lxml')
    bf2_ip_list = BeautifulSoup(str(bf1_ip_list.find_all(id='ip_list')), 'lxml')
    ip_list_info = bf2_ip_list.table.contents
    proxys_list = []
    for index in range(len(ip_list_info)):
        if index % 2 == 1 and index != 1:
            dom = etree.HTML(str(ip_list_info[index]))
            ip = dom.xpath('//td[2]')
            port = dom.xpath('//td[3]')
            protocol = dom.xpath('//td[6]')
            proxys_list.append(protocol[0].text.lower() + '#' + ip[0].text + '#' + port[0].text)
    return proxys_list
'\n函数说明:检查代理IP的连通性\nParameters:\n\tip - 代理的ip地址\n\tlose_time - 匹配丢包数\n\twaste_time - 匹配平均时间\nReturns:\n\taverage_time - 代理ip平均耗时\nModify:\n\t2017-05-27\n'

def check_ip(ip, lose_time, waste_time):
    if False:
        for i in range(10):
            print('nop')
    cmd = 'ping -n 3 -w 3 %s'
    p = sp.Popen(cmd % ip, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
    out = p.stdout.read().decode('gbk')
    lose_time = lose_time.findall(out)
    if len(lose_time) == 0:
        lose = 3
    else:
        lose = int(lose_time[0])
    if lose > 2:
        return 1000
    else:
        average = waste_time.findall(out)
        if len(average) == 0:
            return 1000
        else:
            average_time = int(average[0])
            return average_time
'\n函数说明:初始化正则表达式\nParameters:\n\t无\nReturns:\n\tlose_time - 匹配丢包数\n\twaste_time - 匹配平均时间\nModify:\n\t2017-05-27\n'

def initpattern():
    if False:
        print('Hello World!')
    lose_time = re.compile(u'丢失 = (\\d+)', re.IGNORECASE)
    waste_time = re.compile(u'平均 = (\\d+)ms', re.IGNORECASE)
    return (lose_time, waste_time)
if __name__ == '__main__':
    (lose_time, waste_time) = initpattern()
    proxys_list = get_proxys(1)
    while True:
        proxy = random.choice(proxys_list)
        split_proxy = proxy.split('#')
        ip = split_proxy[1]
        average_time = check_ip(ip, lose_time, waste_time)
        if average_time > 200:
            proxys_list.remove(proxy)
            print('ip连接超时, 重新获取中!')
        if average_time < 200:
            break
    proxys_list.remove(proxy)
    proxy_dict = {split_proxy[0]: split_proxy[1] + ':' + split_proxy[2]}
    print('使用代理:', proxy_dict)