import time
from datetime import datetime
from bs4 import BeautifulSoup
from re import findall
from json import loads
import logging as logme
from .tweet import utc_to_local, Tweet_formats

class NoMoreTweetsException(Exception):

    def __init__(self, msg):
        if False:
            return 10
        super().__init__(msg)

def Follow(response):
    if False:
        i = 10
        return i + 15
    logme.debug(__name__ + ':Follow')
    soup = BeautifulSoup(response, 'html.parser')
    follow = soup.find_all('td', 'info fifty screenname')
    cursor = soup.find_all('div', 'w-button-more')
    try:
        cursor = findall('cursor=(.*?)">', str(cursor))[0]
    except IndexError:
        logme.critical(__name__ + ':Follow:IndexError')
    return (follow, cursor)

def Mobile(response):
    if False:
        while True:
            i = 10
    logme.debug(__name__ + ':Mobile')
    soup = BeautifulSoup(response, 'html.parser')
    tweets = soup.find_all('span', 'metadata')
    max_id = soup.find_all('div', 'w-button-more')
    try:
        max_id = findall('max_id=(.*?)">', str(max_id))[0]
    except Exception as e:
        logme.critical(__name__ + ':Mobile:' + str(e))
    return (tweets, max_id)

def MobileFav(response):
    if False:
        i = 10
        return i + 15
    soup = BeautifulSoup(response, 'html.parser')
    tweets = soup.find_all('table', 'tweet')
    max_id = soup.find_all('div', 'w-button-more')
    try:
        max_id = findall('max_id=(.*?)">', str(max_id))[0]
    except Exception as e:
        print(str(e) + ' [x] feed.MobileFav')
    return (tweets, max_id)

def _get_cursor(response):
    if False:
        return 10
    try:
        next_cursor = response['timeline']['instructions'][0]['addEntries']['entries'][-1]['content']['operation']['cursor']['value']
    except KeyError:
        next_cursor = response['timeline']['instructions'][-1]['replaceEntry']['entry']['content']['operation']['cursor']['value']
    return next_cursor

def Json(response):
    if False:
        print('Hello World!')
    logme.debug(__name__ + ':Json')
    json_response = loads(response)
    html = json_response['items_html']
    soup = BeautifulSoup(html, 'html.parser')
    feed = soup.find_all('div', 'tweet')
    return (feed, json_response['min_position'])

def parse_tweets(config, response):
    if False:
        return 10
    logme.debug(__name__ + ':parse_tweets')
    response = loads(response)
    if len(response['globalObjects']['tweets']) == 0:
        msg = 'No more data!'
        raise NoMoreTweetsException(msg)
    feed = []
    for timeline_entry in response['timeline']['instructions'][0]['addEntries']['entries']:
        if (config.TwitterSearch or config.Profile) and (timeline_entry['entryId'].startswith('sq-I-t-') or timeline_entry['entryId'].startswith('tweet-')):
            if 'tweet' in timeline_entry['content']['item']['content']:
                _id = timeline_entry['content']['item']['content']['tweet']['id']
                if 'promotedMetadata' in timeline_entry['content']['item']['content']['tweet']:
                    continue
            elif 'tombstone' in timeline_entry['content']['item']['content'] and 'tweet' in timeline_entry['content']['item']['content']['tombstone']:
                _id = timeline_entry['content']['item']['content']['tombstone']['tweet']['id']
            else:
                _id = None
            if _id is None:
                raise ValueError('Unable to find ID of tweet in timeline.')
            try:
                temp_obj = response['globalObjects']['tweets'][_id]
            except KeyError:
                logme.info('encountered a deleted tweet with id {}'.format(_id))
                config.deleted.append(_id)
                continue
            temp_obj['user_data'] = response['globalObjects']['users'][temp_obj['user_id_str']]
            if 'retweeted_status_id_str' in temp_obj:
                rt_id = temp_obj['retweeted_status_id_str']
                _dt = response['globalObjects']['tweets'][rt_id]['created_at']
                _dt = datetime.strptime(_dt, '%a %b %d %H:%M:%S %z %Y')
                _dt = utc_to_local(_dt)
                _dt = str(_dt.strftime(Tweet_formats['datetime']))
                temp_obj['retweet_data'] = {'user_rt_id': response['globalObjects']['tweets'][rt_id]['user_id_str'], 'user_rt': response['globalObjects']['tweets'][rt_id]['full_text'], 'retweet_id': rt_id, 'retweet_date': _dt}
            feed.append(temp_obj)
    next_cursor = _get_cursor(response)
    return (feed, next_cursor)