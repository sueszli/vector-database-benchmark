import os
import sys
import requests
import re
from bs4 import BeautifulSoup
os.chdir('\\'.join(__file__.split('/')[:-1]))

def get_page():
    if False:
        return 10
    global url
    url = input('Enter url of a medium article: ')
    if not re.match('https?://medium.com/', url):
        print('Please enter a valid website, or make sure it is a medium article')
        sys.exit(1)
    res = requests.get(url)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, 'html.parser')
    return soup

def purify(text):
    if False:
        for i in range(10):
            print('nop')
    rep = {'<br>': '\n', '<br/>': '\n', '<li>': '\n'}
    rep = dict(((re.escape(k), v) for (k, v) in rep.items()))
    pattern = re.compile('|'.join(rep.keys()))
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
    text = re.sub('\\<(.*?)\\>', '', text)
    return text

def collect_text(soup):
    if False:
        return 10
    fin = f'url: {url}\n\n'
    main = soup.head.title.text.split('|')
    global title
    title = main[0].strip()
    fin += f'Title: {title.upper()}\n{main[1].strip()}'
    header = soup.find_all('h1')
    j = 1
    try:
        fin += '\n\nINTRODUCTION\n'
        for elem in list(header[j].previous_siblings)[::-1]:
            fin += f'\n{purify(str(elem))}'
    except:
        pass
    fin += f'\n\n{header[j].text.upper()}'
    for elem in header[j].next_siblings:
        if elem.name == 'h1':
            j += 1
            fin += f'\n\n{header[j].text.upper()}'
            continue
        fin += f'\n{purify(str(elem))}'
    return fin

def save_file(fin):
    if False:
        print('Hello World!')
    if not os.path.exists('./scraped_articles'):
        os.mkdir('./scraped_articles')
    fname = './scraped_articles/' + '_'.join(title.split()) + '.txt'
    with open(fname, 'w', encoding='utf8') as outfile:
        outfile.write(fin)
    print(f'File saved in directory {fname}')
if __name__ == '__main__':
    fin = collect_text(get_page())
    save_file(fin)