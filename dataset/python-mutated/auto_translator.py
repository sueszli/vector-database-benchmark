import os
import re
import html2text
import markdown
import requests
import requests_cache
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from markdownify import markdownify as md

def get_description(url):
    if False:
        while True:
            i = 10
    requests_cache.install_cache(expire_after=21600)
    proxy = {'http': os.getenv('HTTP_PROXY')}
    access_token = os.getenv('GITHUB_ACCESS_KEY')
    headers = {'Authorization': access_token}
    api_url = url.replace('github.com', 'api.github.com/repos')
    response = requests.get(api_url, proxies=proxy, headers=headers)
    if response.ok:
        description = response.json()['description']
        return description
    else:
        return None

def do_auto_update_star():
    if False:
        while True:
            i = 10
    with open('./README_en.md', 'r', encoding='utf-8') as f:
        content = f.read()
    html = markdown.markdown(content, extensions=['markdown.extensions.tables', 'markdown.extensions.toc'])
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all('table')
    for table in tables:
        header_row = table.find('tr')
        cells = header_row.find_all('th')
        last_name_column_index = None
        for (i, cell) in enumerate(cells):
            if cell.text == 'introduction':
                last_name_column_index = i
                break
        data_rows = table.find_all('tr')[1:]
        for row in data_rows:
            match = re.search('<a href="(.*?)">', str(row))
            if match:
                new_data_cell = soup.new_tag('td')
                url = match.group(1)
                new_data_cell.string = get_description(url) if get_description(url) else ''
                cells_td = row.find_all('td')
                update_row = cells_td[last_name_column_index]
                if len(new_data_cell.string) != 0:
                    update_row.string = new_data_cell.string
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.body_width = 0
    h.ignore_emphasis = True
    h.unicode_snob = True
    h.wrap_links = True
    h.single_line_break = True
    markdown_text = md(str(soup))
    with open('README_en.md', 'w') as f:
        f.write(markdown_text)
if __name__ == '__main__':
    load_dotenv()
    do_auto_update_star()