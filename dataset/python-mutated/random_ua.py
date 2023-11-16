import json
import random
from calibre.utils.resources import get_path as P

def user_agent_data():
    if False:
        for i in range(10):
            print('nop')
    ans = getattr(user_agent_data, 'ans', None)
    if ans is None:
        ans = user_agent_data.ans = json.loads(P('user-agent-data.json', data=True, allow_user_override=False))
    return ans

def common_english_words():
    if False:
        for i in range(10):
            print('nop')
    ans = getattr(common_english_words, 'ans', None)
    if ans is None:
        ans = common_english_words.ans = tuple((x.strip() for x in P('common-english-words.txt', data=True).decode('utf-8').splitlines()))
    return ans

def common_user_agents():
    if False:
        print('Hello World!')
    return user_agent_data()['common_user_agents']

def common_chrome_user_agents():
    if False:
        while True:
            i = 10
    for x in user_agent_data()['common_user_agents']:
        if 'Chrome/' in x:
            yield x

def choose_randomly_by_popularity(ua_list):
    if False:
        print('Hello World!')
    pm = user_agents_popularity_map()
    weights = None
    if pm:
        weights = tuple(map(pm.__getitem__, ua_list))
    return random.choices(ua_list, weights=weights)[0]

def random_common_chrome_user_agent():
    if False:
        return 10
    return choose_randomly_by_popularity(tuple(common_chrome_user_agents()))

def user_agents_popularity_map():
    if False:
        for i in range(10):
            print('nop')
    return user_agent_data().get('user_agents_popularity', {})

def random_desktop_platform():
    if False:
        for i in range(10):
            print('nop')
    return random.choice(user_agent_data()['desktop_platforms'])

def accept_header_for_ua(ua):
    if False:
        while True:
            i = 10
    if 'Firefox/' in ua:
        return 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8'
    return 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8'

def common_english_word_ua():
    if False:
        return 10
    words = common_english_words()
    w1 = random.choice(words)
    w2 = w1
    while w2 == w1:
        w2 = random.choice(words)
    return f'{w1}/{w2}'