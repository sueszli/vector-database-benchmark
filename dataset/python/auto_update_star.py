import os
import re

import html2text
import markdown
import requests
import requests_cache
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from markdownify import markdownify as md


# 获取给定URL的Star数
def get_star_count(url):
    # 启用缓存，并设置过期时间为 6 小时
    requests_cache.install_cache(expire_after=21600)
    # 设置代理
    proxy = {
        "http":os.getenv("HTTP_PROXY")
    }
    access_token = os.getenv("GITHUB_ACCESS_KEY")
    headers = {'Authorization': access_token}
    api_url = url.replace("github.com", "api.github.com/repos")
    response = requests.get(api_url, proxies=proxy, headers=headers)
    if response.ok:
        star_count = response.json()["stargazers_count"]
        if star_count >= 1000:
            star_count_str = f'{star_count / 1000:.1f}k'
        else:
            star_count_str = str(star_count)
        return star_count_str
    else:
        return None

def do_auto_update_star():
    # 读取md文件的内容
    with open("./README.md", 'r', encoding='utf-8') as f:
        content = f.read()

    html = markdown.markdown(content, extensions=['markdown.extensions.tables', 'markdown.extensions.toc'])

    soup = BeautifulSoup(html, "html.parser")

    # 提取所有表格
    tables = soup.find_all("table")

    # 处理每个表格
    for table in tables:
        # 增加一列
        header_row = table.find('tr')
        # 找到所有的单元格
        cells = header_row.find_all('th')
        # 找到 "Last Name" 所在的列
        last_name_column_index = None
        for i, cell in enumerate(cells):
            if cell.text == 'github地址':
                last_name_column_index = i
                break

        column_names = [th.text for th in header_row.find_all('th')]
        if '点赞数' not in column_names:
            new_header_cell = soup.new_tag('th')
            new_header_cell.string = '点赞数'
            header_row.insert(last_name_column_index + 3, new_header_cell)

        # 添加列数据
        data_rows = table.find_all('tr')[1:]
        for row in data_rows:
            match = re.search(r'<a href="(.*?)">', str(row))
            if match:
                new_data_cell = soup.new_tag('td')
                url = match.group(1)
                new_data_cell.string = get_star_count(url) if get_star_count(url) else ""
                if '点赞数' not in column_names:
                    row.insert(last_name_column_index + 3, new_data_cell)
                else:
                    cells_td = row.find_all('td')
                    update_row = cells_td[last_name_column_index + 1]
                    if len(new_data_cell.string) != 0:
                        update_row.string = new_data_cell.string

    # 将HTML保存回Markdown文件
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.body_width = 0
    h.ignore_emphasis = True
    h.unicode_snob = True  # 设置为 True 可以避免一些编码问题
    h.wrap_links = True
    h.single_line_break = True
    markdown_text = md(str(soup))

    with open("README_new.md", "w") as f:
        f.write(markdown_text)


if __name__ == '__main__':
    # 加载.env文件
    load_dotenv()
    do_auto_update_star()