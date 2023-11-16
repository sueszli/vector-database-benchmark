from .assets_initial_state import topstory_ids
import json
import requests
import pandas as pd
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from dagster import AssetExecutionContext, MetadataValue, asset, MaterializeResult

@asset(deps=[topstory_ids])
def topstories(context: AssetExecutionContext) -> MaterializeResult:
    if False:
        return 10
    with open('data/topstory_ids.json', 'r') as f:
        topstory_ids = json.load(f)
    results = []
    for item_id in topstory_ids:
        item = requests.get(f'https://hacker-news.firebaseio.com/v0/item/{item_id}.json').json()
        results.append(item)
        if len(results) % 20 == 0:
            context.log.info(f'Got {len(results)} items so far.')
    df = pd.DataFrame(results)
    df.to_csv('data/topstories.csv')
    return MaterializeResult(metadata={'num_records': len(df), 'preview': MetadataValue.md(df.head().to_markdown())})

@asset(deps=[topstories])
def most_frequent_words() -> MaterializeResult:
    if False:
        return 10
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
    plt.figure(figsize=(10, 6))
    plt.bar(list(top_words.keys()), list(top_words.values()))
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 25 Words in Hacker News Titles')
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    image_data = base64.b64encode(buffer.getvalue())
    md_content = f'![img](data:image/png;base64,{image_data.decode()})'
    with open('data/most_frequent_words.json', 'w') as f:
        json.dump(top_words, f)
    return MaterializeResult(metadata={'plot': MetadataValue.md(md_content)})