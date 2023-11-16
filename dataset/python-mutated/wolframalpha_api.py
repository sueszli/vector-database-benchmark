"""
 Wolfram|Alpha (Science)
"""
from lxml import etree
from urllib.parse import urlencode
about = {'website': 'https://www.wolframalpha.com', 'wikidata_id': 'Q207006', 'official_api_documentation': 'https://products.wolframalpha.com/api/', 'use_official_api': True, 'require_api_key': False, 'results': 'XML'}
search_url = 'https://api.wolframalpha.com/v2/query?appid={api_key}&{query}'
site_url = 'https://www.wolframalpha.com/input/?{query}'
api_key = ''
failure_xpath = '/queryresult[attribute::success="false"]'
input_xpath = '//pod[starts-with(attribute::id, "Input")]/subpod/plaintext'
pods_xpath = '//pod'
subpods_xpath = './subpod'
pod_primary_xpath = './@primary'
pod_id_xpath = './@id'
pod_title_xpath = './@title'
plaintext_xpath = './plaintext'
image_xpath = './img'
img_src_xpath = './@src'
img_alt_xpath = './@alt'
image_pods = {'VisualRepresentation', 'Illustration'}

def request(query, params):
    if False:
        for i in range(10):
            print('nop')
    params['url'] = search_url.format(query=urlencode({'input': query}), api_key=api_key)
    params['headers']['Referer'] = site_url.format(query=urlencode({'i': query}))
    return params

def replace_pua_chars(text):
    if False:
        return 10
    pua_chars = {'\uf522': '→', '\uf7b1': 'ℕ', '\uf7b4': 'ℚ', '\uf7b5': 'ℝ', '\uf7bd': 'ℤ', '\uf74c': 'd', '\uf74d': 'ℯ', '\uf74e': 'i', '\uf7d9': '='}
    for (k, v) in pua_chars.items():
        text = text.replace(k, v)
    return text

def response(resp):
    if False:
        print('Hello World!')
    results = []
    search_results = etree.XML(resp.content)
    if search_results.xpath(failure_xpath):
        return []
    try:
        infobox_title = search_results.xpath(input_xpath)[0].text
    except:
        infobox_title = ''
    pods = search_results.xpath(pods_xpath)
    result_chunks = []
    result_content = ''
    for pod in pods:
        pod_id = pod.xpath(pod_id_xpath)[0]
        pod_title = pod.xpath(pod_title_xpath)[0]
        pod_is_result = pod.xpath(pod_primary_xpath)
        subpods = pod.xpath(subpods_xpath)
        if not subpods:
            continue
        for subpod in subpods:
            content = subpod.xpath(plaintext_xpath)[0].text
            image = subpod.xpath(image_xpath)
            if content and pod_id not in image_pods:
                if pod_is_result or not result_content:
                    if pod_id != 'Input':
                        result_content = '%s: %s' % (pod_title, content)
                if not infobox_title:
                    infobox_title = content
                content = replace_pua_chars(content)
                result_chunks.append({'label': pod_title, 'value': content})
            elif image:
                result_chunks.append({'label': pod_title, 'image': {'src': image[0].xpath(img_src_xpath)[0], 'alt': image[0].xpath(img_alt_xpath)[0]}})
    if not result_chunks:
        return []
    title = 'Wolfram Alpha (%s)' % infobox_title
    results.append({'infobox': infobox_title, 'attributes': result_chunks, 'urls': [{'title': 'Wolfram|Alpha', 'url': resp.request.headers['Referer']}]})
    results.append({'url': resp.request.headers['Referer'], 'title': title, 'content': result_content})
    return results