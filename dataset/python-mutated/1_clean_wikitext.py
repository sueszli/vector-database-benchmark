import os
import re
import time
import timeit
import pandas as pd
import psutil
from datasets import load_dataset
from tqdm import tqdm

def memory_and_speed_test():
    if False:
        print('Hello World!')
    mem_before = psutil.Process(os.getpid()).memory_info().rss >> 20
    wiki = load_dataset('wikipedia', '20220301.en', split='train')
    mem_after = psutil.Process(os.getpid()).memory_info().rss >> 20
    print(f'RAM memory used: {mem_after - mem_before} MB')
    s = 'batch_size = 1000\n    for i in range(0, len(wiki), batch_size):\n        batch = wiki[i:i + batch_size]\n    '
    time = timeit.timeit(stmt=s, number=1, globals=globals())
    size = wiki.dataset_size / 2 ** 30
    print(f'Iterated over the {size:.1f} GB dataset in {time:.1f} s, i.e. {size * 8 / time:.1f} Gbit/s')

def remove_empty_lines(article: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    return article.replace('\n\n', '\n')

def extract_main_content(article: str) -> (str, int):
    if False:
        i = 10
        return i + 15
    lines = []
    word_num = 0
    is_first_line = True
    for line in article.splitlines():
        if (len(line.split(' ')) <= 5 or word_num >= 500) and (not is_first_line):
            return ('\n'.join(lines), word_num)
        is_first_line = False
        word_num += len(line.split(' '))
        lines.append(line)
    return ('\n'.join(lines), word_num)

def remove_all_parentesis(article: str) -> str:
    if False:
        i = 10
        return i + 15
    return re.sub('\\([^)]*\\)', '', article)
if __name__ == '__main__':
    wiki_dataset = load_dataset('wikipedia', '20220301.en', split='train')
    count = 0
    (id_list, url_list, text_list, title_list, word_num_list) = ([], [], [], [], [])
    for page in tqdm(wiki_dataset):
        count += 1
        if count % 1000 == 1:
            date = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            print(f'[{date}] count: {count}')
        (id, url, text, title) = (page['id'], page['url'], page['text'], page['title'])
        text = remove_empty_lines(text)
        (text, word_num) = extract_main_content(text)
        text = remove_all_parentesis(text)
        id_list.append(id)
        url_list.append(url)
        text_list.append(text)
        title_list.append(title)
        word_num_list.append(word_num)
    df = pd.DataFrame({'id': id_list, 'url': url_list, 'text': text_list, 'title': title_list, 'word_num': word_num_list})
    df.to_parquet('wiki_trimmed.parquet')