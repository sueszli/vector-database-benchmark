""" Module that handles the like features """
import random
import re
from re import findall
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, WebDriverException
from selenium.webdriver.common.by import By
from .comment_util import open_comment_section
from .constants import MEDIA_ALL_TYPES, MEDIA_CAROUSEL, MEDIA_IGTV, MEDIA_PHOTO, MEDIA_VIDEO
from .event import Event
from .follow_util import get_following_status
from .quota_supervisor import quota_supervisor
from .time_util import sleep
from .util import add_user_to_blacklist, click_element, evaluate_mandatory_words, explicit_wait, extract_text_from_element, format_number, get_action_delay, get_additional_data, get_number_of_posts, is_page_available, is_private_profile, update_activity, web_address_navigator
from .xpath import read_xpath

def get_links_from_feed(browser, amount, num_of_search, logger):
    if False:
        print('Hello World!')
    'Fetches random number of links from feed and returns a list of links'
    feeds_link = 'https://www.instagram.com/'
    web_address_navigator(browser, feeds_link)
    for i in range(num_of_search + 1):
        browser.execute_script('window.scrollTo(0, document.body.scrollHeight);')
        update_activity(browser, state=None)
        sleep(2)
    link_elems = browser.find_elements(By.XPATH, read_xpath(get_links_from_feed.__name__, 'get_links'))
    total_links = len(link_elems)
    logger.info('Total of links feched for analysis: {}'.format(total_links))
    links = []
    try:
        if link_elems:
            links = [link_elem.get_attribute('href') for link_elem in link_elems]
            logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            for (i, link) in enumerate(links):
                print(i, link)
            logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    except BaseException as e:
        logger.error('link_elems error \n\t{}'.format(str(e).encode('utf-8')))
    return links

def get_main_element(browser, link_elems, skip_top_posts):
    if False:
        return 10
    main_elem = None
    if not link_elems:
        main_elem = browser.find_element(By.XPATH, read_xpath(get_links_for_location.__name__, 'top_elements'))
    elif skip_top_posts:
        main_elem = browser.find_element(By.XPATH, read_xpath(get_links_for_location.__name__, 'main_elem'))
    else:
        main_elem = browser.find_element(By.TAG_NAME, 'main')
    return main_elem

def get_links_for_location(browser, location, amount, logger, media=None, skip_top_posts=True):
    if False:
        return 10
    '\n    Fetches the number of links specified by amount and returns a list of links\n    '
    if media is None:
        media = MEDIA_ALL_TYPES
    elif media == MEDIA_PHOTO:
        media = [MEDIA_PHOTO, MEDIA_CAROUSEL]
    else:
        media = [media]
    location_link = 'https://www.instagram.com/explore/locations/{}'.format(location)
    web_address_navigator(browser, location_link)
    top_elements = browser.find_element(By.XPATH, read_xpath(get_links_for_location.__name__, 'top_elements'))
    top_posts = top_elements.find_elements(By.TAG_NAME, 'a')
    sleep(1)
    if skip_top_posts:
        main_elem = browser.find_element(By.XPATH, read_xpath(get_links_for_location.__name__, 'main_elem'))
    else:
        main_elem = browser.find_element(By.TAG_NAME, 'main')
    link_elems = main_elem.find_elements(By.TAG_NAME, 'a')
    sleep(1)
    if not link_elems:
        main_elem = browser.find_element(By.XPATH, get_links_for_location.__name__, 'top_elements')
        top_posts = []
    sleep(2)
    try:
        possible_posts = browser.execute_script('return window._sharedData.entry_data.LocationsPage[0].graphql.location.edge_location_to_media.count')
    except WebDriverException:
        logger.info("Failed to get the amount of possible posts in '{}' location".format(location))
        possible_posts = None
    logger.info('desired amount: {}  |  top posts [{}]: {}  |  possible posts: {}'.format(amount, 'enabled' if not skip_top_posts else 'disabled', len(top_posts), possible_posts))
    if possible_posts is not None:
        possible_posts = possible_posts if not skip_top_posts else possible_posts - len(top_posts)
        amount = possible_posts if amount > possible_posts else amount
    links = get_links(browser, location, logger, media, main_elem)
    filtered_links = len(links)
    try_again = 0
    sc_rolled = 0
    nap = 1.5
    put_sleep = 0
    try:
        while filtered_links in range(1, amount):
            if sc_rolled > 100:
                logger.info('Scrolled too much! ~ sleeping a bit :>')
                sleep(600)
                sc_rolled = 0
            for i in range(3):
                browser.execute_script('window.scrollTo(0, document.body.scrollHeight);')
                update_activity(browser, state=None)
                sc_rolled += 1
                sleep(nap)
            sleep(3)
            links.extend(get_links(browser, location, logger, media, main_elem))
            links_all = links
            s = set()
            links = []
            for i in links_all:
                if i not in s:
                    s.add(i)
                    links.append(i)
            if len(links) == filtered_links:
                try_again += 1
                nap = 3 if try_again == 1 else 5
                logger.info('Insufficient amount of links ~ trying again: {}'.format(try_again))
                sleep(3)
                if try_again > 2:
                    if put_sleep < 1 and filtered_links <= 21:
                        logger.info("Cor! Did you send too many requests?  ~let's rest some")
                        sleep(600)
                        put_sleep += 1
                        browser.execute_script('location.reload()')
                        update_activity(browser, state=None)
                        try_again = 0
                        sleep(10)
                        main_elem = get_main_element(browser, link_elems, skip_top_posts)
                    else:
                        logger.info("'{}' location POSSIBLY has less images than desired:{} found:{}...".format(location, amount, len(links)))
                        break
            else:
                filtered_links = len(links)
                try_again = 0
                nap = 1.5
    except Exception:
        raise
    sleep(4)
    return links[:amount]

def get_links_for_tag(browser, tag, amount, skip_top_posts, randomize, media, logger):
    if False:
        while True:
            i = 10
    '\n    Fetches the number of links specified by amount and returns a list of links\n    '
    if media is None:
        media = MEDIA_ALL_TYPES
    elif media == MEDIA_PHOTO:
        media = [MEDIA_PHOTO, MEDIA_CAROUSEL]
    else:
        media = [media]
    tag = tag[1:] if tag[:1] == '#' else tag
    tag_link = 'https://www.instagram.com/explore/tags/{}'.format(tag)
    web_address_navigator(browser, tag_link)
    top_elements = browser.find_element(By.XPATH, read_xpath(get_links_for_tag.__name__, 'top_elements'))
    top_posts = top_elements.find_elements(By.TAG_NAME, 'a')
    sleep(1)
    if skip_top_posts:
        main_elem = browser.find_element(By.XPATH, read_xpath(get_links_for_tag.__name__, 'main_elem'))
    else:
        main_elem = browser.find_element(By.TAG_NAME, 'main')
    link_elems = main_elem.find_elements(By.TAG_NAME, 'a')
    sleep(1)
    if not link_elems:
        main_elem = browser.find_element(By.XPATH, read_xpath(get_links_for_tag.__name__, 'top_elements'))
        top_posts = []
    sleep(2)
    try:
        possible_posts = browser.execute_script('return window._sharedData.entry_data.TagPage[0].graphql.hashtag.edge_hashtag_to_media.count')
    except WebDriverException:
        try:
            possible_posts = browser.find_element(By.XPATH, read_xpath(get_links_for_tag.__name__, 'possible_post')).text
            if possible_posts:
                possible_posts = format_number(possible_posts)
            else:
                logger.info("Failed to get the amount of possible posts in '{}' tag  ~empty string".format(tag))
                possible_posts = None
        except NoSuchElementException:
            logger.info('Failed to get the amount of possible posts in {} tag'.format(tag))
            possible_posts = None
    if skip_top_posts:
        amount = amount + 9
    logger.info('desired amount: {}  |  top posts [{}]: {}  |  possible posts: {}'.format(amount, 'enabled' if not skip_top_posts else 'disabled', len(top_posts), possible_posts))
    if possible_posts is not None:
        amount = possible_posts if amount > possible_posts else amount
    links = get_links(browser, tag, logger, media, main_elem)
    filtered_links = 1
    try_again = 0
    sc_rolled = 0
    nap = 1.5
    put_sleep = 0
    try:
        while filtered_links in range(1, amount):
            if sc_rolled > 100:
                logger.info('Scrolled too much! ~ sleeping a bit :>')
                sleep(600)
                sc_rolled = 0
            for i in range(3):
                browser.execute_script('window.scrollTo(0, document.body.scrollHeight);')
                update_activity(browser, state=None)
                sc_rolled += 1
                sleep(nap)
            sleep(3)
            links.extend(get_links(browser, tag, logger, media, main_elem))
            links_all = links
            s = set()
            links = []
            for i in links_all:
                if i not in s:
                    s.add(i)
                    links.append(i)
            if len(links) == filtered_links:
                try_again += 1
                nap = 3 if try_again == 1 else 5
                logger.info('Insufficient amount of links ~ trying again: {}'.format(try_again))
                sleep(3)
                if try_again > 2:
                    if put_sleep < 1 and filtered_links <= 21:
                        logger.info("Cor! Did you send too many requests?  ~let's rest some")
                        sleep(600)
                        put_sleep += 1
                        browser.execute_script('location.reload()')
                        update_activity(browser, state=None)
                        try_again = 0
                        sleep(10)
                        main_elem = get_main_element(browser, link_elems, skip_top_posts)
                    else:
                        logger.info("'{}' tag POSSIBLY has less images than desired:{} found:{}...".format(tag, amount, len(links)))
                        break
            else:
                filtered_links = len(links)
                try_again = 0
                nap = 1.5
    except Exception:
        raise
    sleep(4)
    if skip_top_posts:
        del links[0:9]
    if randomize is True:
        random.shuffle(links)
    return links[:amount]

def get_links_for_username(browser, username, person, amount, logger, logfolder, randomize=False, media=None, taggedImages=False):
    if False:
        while True:
            i = 10
    '\n    Fetches the number of links specified by amount and returns a list of links\n    '
    if media is None:
        media = MEDIA_ALL_TYPES
    elif media == MEDIA_PHOTO:
        media = [MEDIA_PHOTO, MEDIA_CAROUSEL]
    else:
        media = [media]
    logger.info('Getting {} image list...'.format(person))
    user_link = 'https://www.instagram.com/{}/'.format(person)
    if taggedImages:
        user_link = user_link + 'tagged/'
    (following_status, _) = get_following_status(browser, 'profile', username, person, None, logger, logfolder)
    web_address_navigator(browser, user_link)
    if not is_page_available(browser, logger):
        logger.error('Instagram error: The link you followed may be broken, or the page may have been removed...')
        return False
    is_private = is_private_profile(browser, logger, following_status == 'Following')
    if is_private is None or (is_private is True and following_status not in ['Following', True]) or following_status == 'Blocked':
        logger.info("This user is private and we are not following. '{}':'{}'".format(is_private, following_status))
        return False
    links = []
    main_elem = browser.find_element(By.TAG_NAME, 'article')
    posts_count = get_number_of_posts(browser)
    attempt = 0
    if posts_count is not None and amount > posts_count:
        logger.info("You have requested to get {} posts from {}'s profile page but there only {} posts available :D".format(amount, person, posts_count))
        amount = posts_count
    while len(links) < amount:
        initial_links = links
        browser.execute_script('window.scrollTo(0, document.body.scrollHeight);')
        update_activity(browser, state=None)
        sleep(0.66)
        main_elem = browser.find_element(By.TAG_NAME, 'article')
        links = links + get_links(browser, person, logger, media, main_elem)
        links = sorted(set(links), key=links.index)
        if len(links) == len(initial_links):
            if attempt >= 7:
                logger.info("There are possibly less posts than {} in {}'s profile page!".format(amount, person))
                break
            else:
                attempt += 1
        else:
            attempt = 0
    if randomize is True:
        random.shuffle(links)
    return links[:amount]

def get_media_edge_comment_string(media):
    if False:
        while True:
            i = 10
    'AB test (Issue 3712) alters the string for media edge, this resolves it'
    options = ['edge_media_to_comment', 'edge_media_preview_comment']
    for option in options:
        try:
            media[option]
        except KeyError:
            continue
        return option

def check_link(browser, post_link, dont_like, mandatory_words, mandatory_language, mandatory_character, is_mandatory_character, check_character_set, ignore_if_contains, logger):
    if False:
        while True:
            i = 10
    "\n    Check the given link if it is appropriate\n\n    :param browser: The selenium webdriver instance\n    :param post_link:\n    :param dont_like: hashtags of inappropriate phrases\n    :param mandatory_words: words of appropriate phrases\n    :param ignore_if_contains:\n    :param logger: the logger instance\n    :return: tuple of\n        boolean: True if inappropriate,\n        string: the username,\n        boolean: True if it is video media,\n        string: the message if inappropriate else 'None',\n        string: set the scope of the return value\n    "
    web_address_navigator(browser, post_link)
    post_page = get_additional_data(browser)
    if post_page is None:
        logger.warning('Unavailable Page: {}'.format(post_link.encode('utf-8')))
        return (True, None, None, 'Unavailable Page', 'Failure')
    graphql = 'graphql' in post_page
    location_name = None
    if graphql:
        media = post_page['graphql']['shortcode_media']
        is_video = media['is_video']
        user_name = media['owner']['username']
        image_text = media['edge_media_to_caption']['edges']
        image_text = image_text[0]['node']['text'] if image_text else None
        location = media['location']
        location_name = location['name'] if location else None
        media_edge_string = get_media_edge_comment_string(media)
        comments = media[media_edge_string]['edges'] if media[media_edge_string]['edges'] else None
        owner_comments = ''
        if comments is not None:
            for comment in comments:
                if comment['node']['owner']['username'] == user_name:
                    owner_comments = owner_comments + '\n' + comment['node']['text']
    else:
        media = post_page['items'][0]
        is_video = media['is_unified_video']
        user_name = media['user']['username']
        image_text = None
        if media['caption']:
            image_text = media['caption']['text']
        owner_comments = ''
    if owner_comments == '':
        owner_comments = None
    if image_text is None:
        image_text = owner_comments
    elif owner_comments:
        image_text = image_text + '\n' + owner_comments
    if image_text is None:
        image_text = 'No description'
    logger.info('Image from: {}'.format(user_name.encode('utf-8')))
    logger.info('Image link: {}'.format(post_link.encode('utf-8')))
    logger.info('Description: {}'.format(image_text.encode('utf-8')))
    if mandatory_language:
        if not check_character_set(image_text):
            return (True, user_name, is_video, 'Mandatory language not fulfilled', 'Not mandatory language')
    if location_name:
        logger.info('Location: {}'.format(location_name.encode('utf-8')))
        image_text = image_text + '\n' + location_name
    if mandatory_words:
        if not evaluate_mandatory_words(image_text, mandatory_words):
            return (True, user_name, is_video, 'Mandatory words not fulfilled', 'Not mandatory likes')
    image_text_lower = [x.lower() for x in image_text]
    ignore_if_contains_lower = [x.lower() for x in ignore_if_contains]
    if any((word in image_text_lower for word in ignore_if_contains_lower)):
        return (False, user_name, is_video, 'None', 'Pass')
    dont_like_regex = []
    for dont_likes in dont_like:
        if dont_likes.startswith('#'):
            dont_like_regex.append(dont_likes + '([^\\d\\w]|$)')
        elif dont_likes.startswith('['):
            dont_like_regex.append('#' + dont_likes[1:] + '[\\d\\w]+([^\\d\\w]|$)')
        elif dont_likes.startswith(']'):
            dont_like_regex.append('#[\\d\\w]+' + dont_likes[1:] + '([^\\d\\w]|$)')
        else:
            dont_like_regex.append('#[\\d\\w]*' + dont_likes + '[\\d\\w]*([^\\d\\w]|$)')
    for dont_likes_regex in dont_like_regex:
        quash = re.search(dont_likes_regex, image_text, re.IGNORECASE)
        if quash:
            quashed = quash.group(0).split('#')[1].split(' ')[0].split('\n')[0].encode('utf-8')
            iffy = re.split('\\W+', dont_likes_regex)[3] if dont_likes_regex.endswith('*([^\\d\\w]|$)') else re.split('\\W+', dont_likes_regex)[1] if dont_likes_regex.endswith('+([^\\d\\w]|$)') else re.split('\\W+', dont_likes_regex)[3] if dont_likes_regex.startswith('#[\\d\\w]+') else re.split('\\W+', dont_likes_regex)[1]
            inapp_unit = 'Inappropriate! ~ contains "{}"'.format(quashed if iffy == quashed else '" in "'.join([str(iffy), str(quashed)]))
            return (True, user_name, is_video, inapp_unit, 'Undesired word')
    return (False, user_name, is_video, 'None', 'Success')

def like_image(browser, username, blacklist, logger, logfolder, total_liked_img):
    if False:
        i = 10
        return i + 15
    'Likes the browser opened image'
    if quota_supervisor('likes') == 'jump':
        return (False, 'jumped')
    media = 'Image'
    like_xpath = read_xpath(like_image.__name__, 'like')
    unlike_xpath = read_xpath(like_image.__name__, 'unlike')
    play_xpath = read_xpath(like_image.__name__, 'play')
    play_elem = browser.find_elements(By.XPATH, play_xpath)
    if len(play_elem) == 1:
        media = 'Video'
        comment = read_xpath(open_comment_section.__name__, 'comment_elem')
        element = browser.find_element(By.XPATH, comment)
        logger.info("--> Found 'Play' button for a video, trying to like it")
        browser.execute_script('arguments[0].scrollIntoView(true);', element)
    like_elem = browser.find_elements(By.XPATH, like_xpath)
    if len(like_elem) == 1:
        sleep(2)
        logger.info('--> {}...'.format(media))
        like_elem = browser.find_elements(By.XPATH, like_xpath)
        if len(like_elem) > 0:
            click_element(browser, like_elem[0])
        liked_elem = browser.find_elements(By.XPATH, unlike_xpath)
        if len(liked_elem) == 1:
            logger.info('--> {} liked!'.format(media))
            Event().liked(username)
            update_activity(browser, action='likes', state=None, logfolder=logfolder, logger=logger)
            if blacklist['enabled'] is True:
                action = 'liked'
                add_user_to_blacklist(username, blacklist['campaign'], action, logger, logfolder)
            naply = get_action_delay('like')
            sleep(naply)
            if not verify_liked_image(browser, logger):
                return (False, 'block on likes')
            return (True, 'success')
        else:
            logger.info('--> {} was not able to get liked! maybe blocked?'.format(media))
            sleep(120)
    else:
        liked_elem = browser.find_elements(By.XPATH, unlike_xpath)
        if len(liked_elem) == 1:
            logger.info('--> {} already liked!'.format(media))
            return (False, 'already liked')
    logger.info('--> Invalid Like Element!')
    return (False, 'invalid element')

def verify_liked_image(browser, logger):
    if False:
        print('Hello World!')
    'Check for a ban on likes using the last liked image'
    browser.refresh()
    unlike_xpath = read_xpath(like_image.__name__, 'unlike')
    like_elem = browser.find_elements(By.XPATH, unlike_xpath)
    if len(like_elem) == 1:
        return True
    else:
        logger.warning('--> Image was NOT liked! You have a BLOCK on likes!')
        return False

def get_tags(browser, url):
    if False:
        i = 10
        return i + 15
    'Gets all the tags of the given description in the url'
    web_address_navigator(browser, url)
    additional_data = get_additional_data(browser)
    image_text = additional_data['graphql']['shortcode_media']['edge_media_to_caption']['edges'][0]['node']['text']
    if not image_text:
        image_text = ''
    tags = findall('#\\w*', image_text)
    return tags

def get_links(browser, page, logger, media, element):
    if False:
        i = 10
        return i + 15
    links = []
    post_href = None
    try:
        link_elems = element.find_elements(By.XPATH, '//a[starts-with(@href, "/p/")]')
        sleep(random.randint(2, 5))
        if link_elems:
            for link_elem in link_elems:
                try:
                    post_href = link_elem.get_attribute('href')
                    post_elem = element.find_elements(By.XPATH, "//a[@href='/p/" + post_href.split('/')[-2] + "/']/child::div")
                    if len(post_elem) == 1 and MEDIA_PHOTO in media:
                        logger.info('Found media type: {}'.format(MEDIA_PHOTO))
                        links.append(post_href)
                    if len(post_elem) == 2:
                        logger.info('Found media type: {} - {} - {}'.format(MEDIA_CAROUSEL, MEDIA_VIDEO, MEDIA_IGTV))
                        post_category = element.find_element(By.XPATH, "//a[@href='/p/" + post_href.split('/')[-2] + "/']/div[contains(@class,'_aatp')]/child::*/*[name()='svg']").get_attribute('aria-label')
                        logger.info('Post category: {}'.format(post_category))
                        if post_category in media:
                            links.append(post_href)
                except WebDriverException:
                    if post_href:
                        logger.info('Cannot detect post media type. Skip {}'.format(post_href))
        else:
            logger.info("'{}' page does not contain a picture".format(page))
    except BaseException as e:
        logger.error('link_elems error \n\t{}'.format(str(e).encode('utf-8')))
    for (i, link) in enumerate(links):
        logger.info('Links retrieved:: [{}/{}]'.format(i + 1, link))
    return links

def verify_liking(browser, maximum, minimum, logger):
    if False:
        while True:
            i = 10
    'Get the amount of existing existing likes and compare it against maximum\n    & minimum values defined by user'
    post_page = get_additional_data(browser)
    likes_count = post_page['items'][0]['like_count']
    if not likes_count:
        likes_count = 0
    if maximum is not None and likes_count > maximum:
        logger.info('Not liked this post! ~more likes exist off maximum limit at {}'.format(likes_count))
        return False
    elif minimum is not None and likes_count < minimum:
        logger.info('Not liked this post! ~less likes exist off minimum limit at {}'.format(likes_count))
        return False
    return True

def like_comment(browser, original_comment_text, logger):
    if False:
        while True:
            i = 10
    'Like the given comment'
    comments_block_XPath = read_xpath(like_comment.__name__, 'comments_block')
    try:
        comments_block = browser.find_elements(By.XPATH, comments_block_XPath)
        for comment_line in comments_block:
            comment_elem = comment_line.find_elements(By.TAG_NAME, 'span')[0]
            comment = extract_text_from_element(comment_elem)
            if comment and comment == original_comment_text:
                span_like_elements = comment_line.find_elements(By.XPATH, read_xpath(like_comment.__name__, 'span_like_elements'))
                if not span_like_elements:
                    return (True, 'success')
                span_like = span_like_elements[0]
                comment_like_button = span_like.find_element(By.XPATH, read_xpath(like_comment.__name__, 'comment_like_button'))
                click_element(browser, comment_like_button)
                button_change = explicit_wait(browser, 'SO', [comment_like_button], logger, 7, False)
                if button_change:
                    logger.info('--> Liked the comment!')
                    sleep(random.uniform(1, 2))
                    return (True, 'success')
                else:
                    logger.info('--> Unfortunately, comment was not liked.')
                    sleep(random.uniform(0, 1))
                    return (False, 'failure')
    except (NoSuchElementException, StaleElementReferenceException) as exc:
        logger.error('Error occurred while liking a comment.\n\t{}'.format(str(exc).encode('utf-8')))
        return (False, 'error')
    return (None, 'unknown')