import urllib.request, urllib.parse
import http.cookiejar
import json
__author__ = 'Dananjaya Ramanayake <dananjaya86@gmail.com>, Samuel Clay <samuel@newsblur.com>'
__version__ = '1.0'
API_URL = 'https://www.newsblur.com/'

class request:
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def __init__(self, endpoint=None, method='get'):
        if False:
            i = 10
            return i + 15
        self.endpoint = endpoint
        self.method = method

    def __call__(self, func):
        if False:
            while True:
                i = 10

        def wrapped(*args, **kwargs):
            if False:
                print('Hello World!')
            params = func(*args, **kwargs) or {}
            url = self.endpoint if self.endpoint else params.pop('url')
            params = urllib.parse.urlencode(params)
            url = '%s%s' % (API_URL, url)
            response = self.opener.open(url, params).read()
            return json.loads(response)
        return wrapped

class API:

    @request('api/login', method='post')
    def login(self, username, password):
        if False:
            print('Hello World!')
        '\n        Login as an existing user.\n        If a user has no password set, you cannot just send any old password. \n        Required parameters, username and password, must be of string type.\n        '
        return {'username': username, 'password': password}

    @request('api/logout')
    def logout(self):
        if False:
            print('Hello World!')
        '\n        Logout the currently logged in user.\n        '
        return

    @request('api/signup')
    def signup(self, username, password, email):
        if False:
            i = 10
            return i + 15
        '\n        Create a new user.\n        All three required parameters must be of type string.\n        '
        return {'signup_username': username, 'signup_password': password, 'signup_email': email}

    @request('rss_feeds/search_feed')
    def search_feed(self, address, offset=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieve information about a feed from its website or RSS address.\n        Parameter address must be of type string while parameter offset must be an integer.\n        Will return a feed.\n        '
        return {'address': address, 'offset': offset}

    @request('reader/feeds')
    def feeds(self, include_favicons=True, flat=False):
        if False:
            print('Hello World!')
        '\n        Retrieve a list of feeds to which a user is actively subscribed.\n        Includes the 3 unread counts (positive, neutral, negative), as well as optional favicons.\n        '
        return {'include_favicons': include_favicons, 'flat': flat}

    @request('reader/favicons')
    def favicons(self, feeds=None):
        if False:
            i = 10
            return i + 15
        '\n        Retrieve a list of favicons for a list of feeds. \n        Used when combined with /reader/feeds and include_favicons=false, so the feeds request contains far less data. \n        Useful for mobile devices, but requires a second request. \n        '
        data = []
        for feed in feeds:
            data.append(('feeds', feed))
        return data

    @request()
    def page(self, feed_id):
        if False:
            i = 10
            return i + 15
        '\n        Retrieve the original page from a single feed.\n        '
        return {'url': 'reader/page/%s' % feed_id}

    @request()
    def feed(self, feed_id, page=1):
        if False:
            print('Hello World!')
        '\n        Retrieve the stories from a single feed.\n        '
        return {'url': 'reader/feed/%s' % feed_id, 'page': page}

    @request('reader/refresh_feeds')
    def refresh_feeds(self):
        if False:
            i = 10
            return i + 15
        '\n        Up-to-the-second unread counts for each active feed.\n            Poll for these counts no more than once a minute.\n        '
        return

    @request('reader/feeds_trainer')
    def feeds_trainer(self, feed_id=None):
        if False:
            while True:
                i = 10
        "\n         Retrieves all popular and known intelligence classifiers.\n            Also includes user's own classifiers.\n        "
        return {'feed_id': feed_id}

    @request()
    def statistics(self, feed_id=None):
        if False:
            i = 10
            return i + 15
        "\n        If you only want a user's classifiers, use /classifiers/:id.\n            Omit the feed_id to get all classifiers for all subscriptions.\n        "
        return {'url': 'rss_feeds/statistics/%d' % feed_id}

    @request('rss_feeds/feed_autocomplete')
    def feed_autocomplete(self, term):
        if False:
            print('Hello World!')
        '\n        Get a list of feeds that contain a search phrase.\n        Searches by feed address, feed url, and feed title, in that order.\n        Will only show sites with 2+ subscribers.\n        '
        return {'term': term}

    @request('reader/starred_stories')
    def starred_stories(self, page=1):
        if False:
            i = 10
            return i + 15
        "\n        Retrieve a user's starred stories.\n        "
        return {'page': page}

    @request('reader/river_stories')
    def river_stories(self, feeds, page=1, read_stories_count=0):
        if False:
            print('Hello World!')
        '\n        Retrieve stories from a collection of feeds. This is known as the River of News.\n        Stories are ordered in reverse chronological order.\n        `read_stories_count` is the number of stories that have been read in this\n        continuation, so NewsBlur can efficiently skip those stories when retrieving\n        new stories. Takes an array of feed ids.\n        '
        data = [('page', page), ('read_stories_count', read_stories_count)]
        for feed in feeds:
            data.append(('feeds', feed))
        return data

    @request('reader/mark_story_hashes_as_read')
    def mark_story_hashes_as_read(self, story_hashes):
        if False:
            return 10
        '\n         Mark stories as read using their unique story_hash.\n        '
        data = []
        for hash in story_hashes:
            data.append(('story_hash', hash))
        return data

    @request('reader/mark_story_as_read')
    def mark_story_as_read(self, feed_id, story_ids):
        if False:
            print('Hello World!')
        '\n         Mark stories as read.\n            Multiple story ids can be sent at once.\n            Each story must be from the same feed.\n            Takes an array of story ids.\n        '
        data = [('feed_id', feed_id)]
        for story_id in story_ids:
            data.append(('story_id', story_id))
        return data

    @request('reader/mark_story_as_starred')
    def mark_story_as_starred(self, feed_id, story_id):
        if False:
            while True:
                i = 10
        '\n        Mark a story as starred (saved).\n        '
        return {'feed_id': feed_id, 'story_id': story_id}

    @request('reader/mark_all_as_read')
    def mark_all_as_read(self, days=0):
        if False:
            print('Hello World!')
        '\n        Mark all stories in a feed or list of feeds as read.\n        '
        return {'days': days}

    @request('reader/add_url')
    def add_url(self, url, folder=''):
        if False:
            print('Hello World!')
        '\n        Add a feed by its URL. \n        Can be either the RSS feed or the website itself.\n        '
        return {'url': url, 'folder': folder}

    @request('reader/add_folder')
    def add_folder(self, folder, parent_folder=''):
        if False:
            while True:
                i = 10
        '\n        Add a new folder.\n        '
        return {'folder': folder, 'parent_folder': parent_folder}

    @request('reader/rename_feed')
    def rename_feed(self, feed_id, feed_title):
        if False:
            print('Hello World!')
        '\n        Rename a feed title. Only the current user will see the new title.\n        '
        return {'feed_id': feed_id, 'feed_title': feed_title}

    @request('reader/delete_feed')
    def delete_feed(self, feed_id, in_folder):
        if False:
            i = 10
            return i + 15
        '\n        Unsubscribe from a feed. Removes it from the folder.\n        Set the in_folder parameter to remove a feed from the correct \n        folder, in case the user is subscribed to the feed in multiple folders.\n        '
        return {'feed_id': feed_id, 'in_folder': in_folder}

    @request('reader/rename_folder')
    def rename_folder(self, folder_to_rename, new_folder_name, in_folder):
        if False:
            print('Hello World!')
        '\n        Rename a folder.\n        '
        return {'folder_to_rename': folder_to_rename, 'new_folder_name': new_folder_name, 'in_folder': in_folder}

    @request('reader/delete_folder')
    def delete_folder(self, folder_to_delete, in_folder):
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete a folder and unsubscribe from all feeds inside.\n        '
        return {'folder_to_delete': folder_to_delete, 'in_folder': in_folder}

    @request('reader/mark_feed_as_read')
    def mark_feed_as_read(self, feed_ids):
        if False:
            print('Hello World!')
        '\n        Mark a list of feeds as read.\n        Takes an array of feeds.\n        '
        data = []
        for feed in feed_ids:
            data.append(('feed_id', feed))
        return data

    @request('reader/save_feed_order')
    def save_feed_order(self, folders):
        if False:
            for i in range(10):
                print('nop')
        '\n        Reorder feeds and move them around between folders.\n            The entire folder structure needs to be serialized.\n        '
        return {'folders': folders}

    @request()
    def classifier(self, feed_id):
        if False:
            for i in range(10):
                print('nop')
        "\n            Get the intelligence classifiers for a user's site.\n            Only includes the user's own classifiers. \n            Use /reader/feeds_trainer for popular classifiers.\n        "
        return {'url': '/classifier/%d' % feed_id}

    @request('classifier/save')
    def classifier_save(self, like_type, dislike_type, remove_like_type, remove_dislike_type):
        if False:
            while True:
                i = 10
        '\n        Save intelligence classifiers (tags, titles, authors, and the feed) for a feed.\n        \n        TODO: Make this usable.\n        '
        raise NotImplemented

    @request('import/opml_export')
    def opml_export(self):
        if False:
            while True:
                i = 10
        '\n        Download a backup of feeds and folders as an OPML file.\n        Contains folders and feeds in XML; useful for importing in another RSS reader.\n        '
        return

    @request('import/opml_upload')
    def opml_upload(self, opml_file):
        if False:
            for i in range(10):
                print('nop')
        '\n        Upload an OPML file.\n        '
        f = open(opml_file)
        return {'file': f}