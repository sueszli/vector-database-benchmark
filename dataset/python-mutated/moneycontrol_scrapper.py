import re
import json
import requests
import datetime
from tqdm import tqdm
from bs4 import BeautifulSoup
from collections import defaultdict
submission = defaultdict(list)
src_url = 'https://www.moneycontrol.com/news/technical-call-221.html'

def setup(url):
    if False:
        for i in range(10):
            print('nop')
    nextlinks = []
    src_page = requests.get(url).text
    src = BeautifulSoup(src_page, 'lxml')
    anchors = src.find('div', attrs={'class': 'pagenation'}).findAll('a', {'href': re.compile('^((?!void).)*$')})
    nextlinks = [i.attrs['href'] for i in anchors]
    for (idx, link) in enumerate(tqdm(nextlinks)):
        scrap('https://www.moneycontrol.com' + link, idx)

def scrap(url, idx):
    if False:
        while True:
            i = 10
    src_page = requests.get(url).text
    src = BeautifulSoup(src_page, 'lxml')
    span = src.find('ul', {'id': 'cagetory'}).findAll('span')
    img = src.find('ul', {'id': 'cagetory'}).findAll('img')
    imgs = [i.attrs['src'] for i in img]
    titles = [i.attrs['alt'] for i in img]
    date = [i.get_text() for i in span]
    submission[str(idx)].append({'title': titles})
    submission[str(idx)].append({'date': date})
    submission[str(idx)].append({'img_src': imgs})

def json_dump(data):
    if False:
        return 10
    date = datetime.date.today().strftime('%B %d, %Y')
    with open('moneycontrol_' + str(date) + '.json', 'w') as outfile:
        json.dump(submission, outfile)
setup(src_url)
json_dump(submission)