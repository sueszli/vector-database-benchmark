import json
import requests

def load(filename):
    if False:
        i = 10
        return i + 15
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def save(filename, obj):
    if False:
        print('Hello World!')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '), ensure_ascii=False)
    pass

def get(url):
    if False:
        print('Hello World!')
    req = requests.get(url)
    return req.json()