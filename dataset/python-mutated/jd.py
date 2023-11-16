import os
import re
import sys
import bs4
import json
import math
import time
import math
import argparse
import requests
from contextlib import closing

def search_goods(keyword, pages):
    if False:
        return 10
    '\n\t搜索商品\n\tParameters:\n\t\tkeyword - str 搜索关键词\n\t\tpages - int 搜索页数\n\tReturns:\n\t\tgoods_urls - list 商品链接\n\t'
    sess = requests.Session()
    goods_urls = []
    for page in range(pages):
        search_headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8', 'Accept-Encoding': 'gzip, deflate, br', 'Accept-Language': 'zh-CN,zh;q=0.9', 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.167 Safari/537.36', 'Host': 'search.jd.com'}
        s = page * 28
        if s == 0:
            s = 1
        search_url = 'https://search.jd.com/Search'
        search_params = {'keyword': keyword, 'enc': 'utf-8', 'qrst': '1', 'rt': '1', 'stop': '1', 'vt': '2', 'wq': keyword, 'stock': '1', 'page': page * 2 + 1, 's': s, 'click': '0'}
        search_req = sess.get(url=search_url, params=search_params, headers=search_headers, verify=False)
        search_req.encoding = 'utf-8'
        search_req_bf = bs4.BeautifulSoup(search_req.text, 'lxml')
        for item in search_req_bf.find_all('li', class_='gl-item'):
            item_url = item.div.div.a.get('href')
            if 'ccc-x.jd.com' not in item_url:
                goods_urls.append(item_url)
        log_id = re.findall("log_id:'(.*)',", search_req.text)[0]
        search_more_url = 'https://search.jd.com/s_new.php'
        search_more_headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8', 'Accept-Encoding': 'gzip, deflate, br', 'Accept-Language': 'zh-CN,zh;q=0.9', 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.167 Safari/537.36', 'Host': 'search.jd.com', 'Referer': search_req.url}
        s = (1 + page) * 25
        search_more_params = {'keyword': keyword, 'enc': 'utf-8', 'qrst': '1', 'rt': '1', 'stop': '1', 'vt': '2', 'wq': keyword, 'stock': '1', 'page': (1 + page) * 2, 's': s, 'log_id': log_id, 'scrolling': 'y', 'tpl': '1_M'}
        search_more_req = sess.get(url=search_more_url, params=search_more_params, headers=search_more_headers, verify=False)
        search_more_req.encoding = 'utf-8'
        search_more_req_bf = bs4.BeautifulSoup(search_more_req.text, 'lxml')
        for item in search_more_req_bf.find_all('li', class_='gl-item'):
            item_url = item.div.div.a.get('href')
            if 'ccc-x.jd.com' not in item_url:
                goods_urls.append(item_url)
    goods_urls = list(set(goods_urls))
    goods_urls = list(map(lambda x: 'http:' + x, goods_urls))
    return goods_urls

def goods_images(goods_url):
    if False:
        print('Hello World!')
    '\n\t获得商品晒图\n\tParameters:\n\t\tgoods_url - str 商品链接\n\tReturns:\n\t\timage_urls - list 图片链接\n\t'
    image_urls = []
    productId = goods_url.split('/')[-1].split('.')[0]
    comment_url = 'https://sclub.jd.com/comment/productPageComments.action'
    comment_params = {'productId': productId, 'score': '0', 'sortType': '5', 'page': '0', 'pageSize': '10', 'isShadowSku': '0', 'fold': '1'}
    comment_headers = {'Accept': '*/*', 'Accept-Encoding': 'gzip, deflate, br', 'Accept-Language': 'zh-CN,zh;q=0.9', 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.167 Safari/537.36', 'Referer': goods_url, 'Host': 'sclub.jd.com'}
    comment_req = requests.get(url=comment_url, params=comment_params, headers=comment_headers, verify=False)
    html = json.loads(comment_req.text)
    imageListCount = html['imageListCount']
    pages = math.ceil(imageListCount / 10)
    for page in range(1, pages + 1):
        club_url = 'https://club.jd.com/discussion/getProductPageImageCommentList.action'
        now = time.time()
        now_str = str(now).split('.')
        now = now_str[0] + now_str[-1][:3]
        club_params = {'productId': productId, 'isShadowSku': '0', 'page': page, 'pageSize': '10', '_': now}
        club_headers = comment_headers
        club_req = requests.get(url=club_url, params=club_params, headers=club_headers, verify=False)
        html = json.loads(club_req.text)
        for img in html['imgComments']['imgList']:
            image_urls.append(img['imageUrl'])
    image_urls = list(set(image_urls))
    image_urls = list(map(lambda x: 'http:' + x, image_urls))
    return image_urls

def download_image(path, image_url):
    if False:
        print('Hello World!')
    '\n\t图片下载\n\tParameters:\n\t\tpath - str 图片保存地址\n\t\timage_url - str 图片下载地址\n\tReturns:\n\t\tNone\n\t'
    print(image_url)
    filename = image_url.split('/')[-1]
    image_path = os.path.join(path, filename)
    download_headers = {'Accept': '*/*', 'Accept-Encoding': 'gzip, deflate, br', 'Accept-Language': 'zh-CN,zh;q=0.9', 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.167 Safari/537.36'}
    size = 0
    with closing(requests.get(image_url, headers=download_headers, stream=True)) as response:
        chunk_size = 1024
        content_size = int(response.headers['content-length'])
        if response.status_code == 200:
            sys.stdout.write(filename + '下载中:\n')
            sys.stdout.write('    [文件大小]:%0.2f MB\n' % (content_size / chunk_size / 1024))
            with open(image_path, 'wb') as file:
                for data in response.iter_content(chunk_size=chunk_size):
                    file.write(data)
                    size += len(data)
                    file.flush()
                    sys.stdout.write('    [下载进度]:%.2f%%' % float(size / content_size * 100) + '\r')
                    sys.stdout.flush()

def run(path, keyword, num):
    if False:
        i = 10
        return i + 15
    '\n\t运行函数\n\tParameters:\n\t\tpath - str 图片保存目录\n\t\tkeyword - str 关键词\n\t\tnum - int 下载的商店个数\n\tReturns:\n\t\tNone\n\t'
    flag = False
    pages = 1
    while flag == False:
        goods_urls = search_goods(keyword, pages)
        if len(goods_urls) > num:
            flag = True
        else:
            pages += 1
    if keyword not in os.listdir():
        os.mkdir(keyword)
    path = os.path.join(path, keyword)
    for goods_url in goods_urls[:num]:
        image_urls = goods_images(goods_url)
        for image_url in image_urls:
            download_image(path, image_url)
if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.append('--help')
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', help='store path', type=str, default=os.path.dirname(__file__))
    parser.add_argument('-k', '--keyword', required=True, help='search content')
    parser.add_argument('-n', '--num', help='the number of goods to download images', type=int, default=1)
    args = parser.parse_args()
    run(args.dir, args.keyword, args.num)