import json
import os
import requests
from dagster import asset

@asset
def topstory_ids() -> None:
    if False:
        print('Hello World!')
    newstories_url = 'https://hacker-news.firebaseio.com/v0/topstories.json'
    top_new_story_ids = requests.get(newstories_url).json()[:100]
    os.makedirs('data', exist_ok=True)
    with open('data/topstory_ids.json', 'w') as f:
        json.dump(top_new_story_ids, f)
import json
import os
import pandas as pd
import requests
from dagster import asset

@asset(deps=[topstory_ids])
def topstories() -> None:
    if False:
        i = 10
        return i + 15
    with open('data/topstory_ids.json', 'r') as f:
        topstory_ids = json.load(f)
    results = []
    for item_id in topstory_ids:
        item = requests.get(f'https://hacker-news.firebaseio.com/v0/item/{item_id}.json').json()
        results.append(item)
        if len(results) % 20 == 0:
            print(f'Got {len(results)} items so far.')
    df = pd.DataFrame(results)
    df.to_csv('data/topstories.csv')

@asset(deps=[topstories])
def most_frequent_words() -> None:
    if False:
        i = 10
        return i + 15
    stopwords = ['a', 'the', 'an', 'of', 'to', 'in', 'for', 'and', 'with', 'on', 'is']
    topstories = pd.read_csv('data/topstories.csv')
    word_counts = {}
    for raw_title in topstories['title']:
        title = raw_title.lower()
        for word in title.split():
            cleaned_word = word.strip('.,-!?:;()[]\'"-')
            if cleaned_word not in stopwords and len(cleaned_word) > 0:
                word_counts[cleaned_word] = word_counts.get(cleaned_word, 0) + 1
    top_words = {pair[0]: pair[1] for pair in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:25]}
    with open('data/most_frequent_words.json', 'w') as f:
        json.dump(top_words, f)