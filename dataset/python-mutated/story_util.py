import math
import time
from random import randint
import requests
from selenium.common.exceptions import NoSuchElementException, WebDriverException
from selenium.webdriver.common.by import By
from .settings import Settings
from .util import click_element, get_action_delay, update_activity, web_address_navigator
from .xpath import read_xpath

def get_story_data(browser, elem, action_type, logger, simulate=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    get the JSON data from the graphql URL\n    output the amount of segments we can watch\n    '
    query_hash = 'cda12de4f7fd3719c0569ce03589f4c4'
    elem_id = ''
    if action_type == 'user':
        try:
            reel_id = browser.execute_script('return window.__additionalData[Object.keys(window.__additionalData)[0]].data.graphql.user.id')
            elem_id = '"' + reel_id + '"'
            elem = ''
        except WebDriverException:
            try:
                reel_id = browser.execute_script('return window._sharedData.entry_data.ProfilePage[0].graphql.user.id')
                elem_id = '"' + reel_id + '"'
                elem = ''
            except WebDriverException:
                logger.error("--> Sorry, this page isn't available!\t~either " + 'link is broken or page is removed\n')
                return {'status': 'not_ok', 'reels_cnt': 0}
    else:
        reel_id = 'tag:{}'.format(elem)
    graphql_query_url = 'https://www.instagram.com/graphql/query/?query_hash={}&variables={{"reel_ids":[{}],"tag_names":["{}"],"location_ids":[],"highlight_reel_ids":[],"precomposed_overlay":false,"show_story_viewer_list":true,"story_viewer_fetch_count":50,"story_viewer_cursor":"","stories_video_dash_manifest":false}}'.format(query_hash, elem_id, elem)
    cookies = browser.get_cookies()
    session = requests.Session()
    csrftoken = ''
    for cookie in cookies:
        all_args = {'name': cookie['name'], 'value': cookie['value'], 'domain': cookie['domain'], 'secure': cookie['secure'], 'rest': {'HttpOnly': cookie['httpOnly']}, 'path': cookie['path']}
        if cookie['name'] == 'csrftoken':
            csrftoken = cookie['value']
        if not cookie['name'] == 'urlgen' and (not cookie['name'] == 'rur'):
            all_args['expires'] = cookie['expiry']
        session.cookies.set(**all_args)
    headers = {'User-Agent': Settings.user_agent, 'X-Requested-With': 'XMLHttpRequest', 'SameSite': 'Strict'}
    data = session.get(graphql_query_url, headers=headers)
    response = data.json()
    update_activity(browser, state=None)
    reels_cnt = 0
    if response['status'] == 'ok':
        media_cnt = len(response['data']['reels_media'])
        if media_cnt == 0:
            return {'status': 'ok', 'reels_cnt': 0}
        else:
            seen = 0
            if action_type != 'tag' and response['data']['reels_media'][0]['seen'] is not None:
                seen = response['data']['reels_media'][0]['seen']
            index = 1
            if simulate is True:
                for item in response['data']['reels_media'][0]['items']:
                    if item['taken_at_timestamp'] <= seen:
                        continue
                    else:
                        headers = {'User-Agent': Settings.user_agent, 'X-CSRFToken': csrftoken, 'X-Requested-With': 'XMLHttpRequest', 'Content-Type': 'application/x-www-form-urlencoded', 'SameSite': 'Strict'}
                        response = session.post('https://www.instagram.com/stories/reel/seen', data={'reelMediaId': item['id'], 'reelMediaOwnerId': item['owner']['id'], 'reelId': reel_id, 'reelMediaTakenAt': item['taken_at_timestamp'], 'viewSeenAt': math.floor(time.time())}, headers=headers)
                        logger.info('  --> simulated watch reel # {}'.format(index))
                        update_activity()
                        index += 1
                        time.sleep(randint(3, 6))
                        reels_cnt += 1
            else:
                story_elem = browser.find_element(By.XPATH, read_xpath(watch_story.__name__ + '_for_{}'.format(action_type), 'explore_stories'))
                click_element(browser, story_elem)
                logger.info('Watching stories...')
                for item in response['data']['reels_media'][0]['items']:
                    if item['taken_at_timestamp'] <= seen:
                        continue
                    else:
                        time.sleep(2)
                        if index == 1:
                            try:
                                next_elem = browser.find_element(By.XPATH, read_xpath(watch_story.__name__, 'next_first'))
                            except NoSuchElementException:
                                continue
                        else:
                            try:
                                next_elem = browser.find_element(By.XPATH, read_xpath(watch_story.__name__, 'next'))
                            except NoSuchElementException:
                                continue
                        click_element(browser, next_elem)
                        reels_cnt += 1
                        index += 1
            return {'status': 'ok', 'reels_cnt': reels_cnt}
    else:
        return {'status': 'not_ok', 'reels_cnt': 0}

def watch_story(browser, elem, logger, action_type, simulate=False):
    if False:
        while True:
            i = 10
    '\n    Load Stories, and watch it until there is no more stores\n    to watch for the related element\n    '
    elem = elem.lower()
    if action_type == 'tag':
        story_link = 'https://www.instagram.com/explore/tags/{}'.format(elem)
    else:
        story_link = 'https://www.instagram.com/{}'.format(elem)
    web_address_navigator(browser, story_link)
    time.sleep(randint(2, 6))
    story_data = get_story_data(browser, elem, action_type, logger, simulate)
    if story_data['status'] == 'not ok':
        raise NoSuchElementException
    if story_data['reels_cnt'] == 0:
        logger.info('No stories to watch (either there is none) or we have already watched everything')
        return 0
    logger.info('Watched {} reels from {}: {}'.format(story_data['reels_cnt'], action_type, elem.encode('utf-8')))
    naply = get_action_delay('story')
    time.sleep(naply)
    return story_data['reels_cnt']