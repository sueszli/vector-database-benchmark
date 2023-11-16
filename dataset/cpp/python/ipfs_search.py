from urllib.parse import urlencode
import json
import re
from datetime import datetime

# config
categories = ['general', 'images', 'music', 'videos']
paging = True
time_range_support = True

# url
base_url = 'https://api.ipfs-search.com/v1/'
search_string = 'search?{query} first-seen:{time_range} metadata.Content-Type:({mime_type})&page={page} '


mime_types_map = {
    'general': "*",
    'images': 'image*',
    'music': 'audio*',
    'videos': 'video*'
}

time_range_map = {'day': '[ now-24h\/h TO *]',
                  'week': '[ now\/h-7d TO *]',
                  'month': '[ now\/d-30d TO *]',
                  'year': '[ now\/d-1y TO *]'}

ipfs_url = 'https://gateway.ipfs.io/ipfs/{hash}'


def request(query, params):
    mime_type = mime_types_map.get(params['category'], '*')
    time_range = time_range_map.get(params['time_range'], '*')
    search_path = search_string.format(
        query=urlencode({'q': query}),
        time_range=time_range,
        page=params['pageno'],
        mime_type=mime_type)

    params['url'] = base_url + search_path

    return params


def clean_html(text):
    if not text:
        return ""
    return str(re.sub(re.compile('<.*?>'), '', text))


def create_base_result(record):
    url = ipfs_url.format(hash=record.get('hash'))
    title = clean_html(record.get('title'))
    published_date = datetime.strptime(record.get('first-seen'), '%Y-%m-%dT%H:%M:%SZ')
    return {'url': url,
            'title': title,
            'publishedDate': published_date}


def create_text_result(record):
    result = create_base_result(record)
    description = clean_html(record.get('description'))
    result['description'] = description
    return result


def create_image_result(record):
    result = create_base_result(record)
    result['img_src'] = result['url']
    result['template'] = 'images.html'
    return result


def create_video_result(record):
    result = create_base_result(record)
    result['thumbnail'] = ''
    result['template'] = 'videos.html'
    return result


def response(resp):
    api_results = json.loads(resp.text)
    results = []
    for result in api_results.get('hits', []):
        mime_type = result.get('mimetype', 'text/plain')

        if mime_type.startswith('image'):
            results.append(create_image_result(result))
        elif mime_type.startswith('video'):
            results.append(create_video_result(result))
        else:
            results.append(create_text_result(result))
    return results
