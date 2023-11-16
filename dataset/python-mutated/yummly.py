"""Yummly
"""
from urllib.parse import urlencode
from flask_babel import gettext
from searx.utils import markdown_to_text
about = {'website': 'https://yummly.com', 'wikidata_id': 'Q8061140', 'official_api_documentation': 'https://developer.yummly.com/documentation.html', 'use_official_api': False, 'require_api_key': False, 'results': 'JSON'}
paging = True
categories = []
api_url = 'https://mapi.yummly.com'
number_of_results = 10
show_pro_recipes = False
base_url = 'https://www.yummly.com'

def request(query, params):
    if False:
        print('Hello World!')
    args = {'q': query, 'start': (params['pageno'] - 1) * number_of_results, 'maxResult': number_of_results}
    params['url'] = f'{api_url}/mapi/v23/content/search?{urlencode(args)}&allowedContent=single_recipe'
    return params

def response(resp):
    if False:
        return 10
    results = []
    json = resp.json()
    for result in json['feed']:
        if not show_pro_recipes and result['proRecipe']:
            continue
        content = result['seo']['web']['meta-tags']['description']
        description = result['content']['description']
        if description is not None:
            content = markdown_to_text(description['text'])
        img_src = None
        if result['display']['images']:
            img_src = result['display']['images'][0]
        elif result['content']['details']['images']:
            img_src = result['content']['details']['images'][0]['resizableImageUrl']
        url = result['display']['source']['sourceRecipeUrl']
        if 'www.yummly.com/private' in url:
            url = base_url + '/' + result['tracking-id']
        results.append({'url': url, 'title': result['display']['displayName'], 'content': content, 'img_src': img_src, 'metadata': f"{gettext('Language')}: {result['locale'].split('-')[0]}"})
    for suggestion in json['relatedPhrases']['keywords']:
        results.append({'suggestion': suggestion})
    return results