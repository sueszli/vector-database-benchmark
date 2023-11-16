"""
Examples to demo the :py:mod:`twitterclient` code.

These demo functions should all run, with the following caveats:

* You must have obtained API keys from Twitter, and installed them according to
  the instructions in the `twitter HOWTO <https://www.nltk.org/howto/twitter.html>`_.

* If you are on a slow network, some of the calls to the Twitter API may
  timeout.

* If you are being rate limited while searching, you will receive a 420
  error response.

* Your terminal window / console must be able to display UTF-8 encoded characters.

For documentation about the Twitter APIs, see `The Streaming APIs Overview
<https://dev.twitter.com/streaming/overview>`_ and `The REST APIs Overview
<https://dev.twitter.com/rest/public>`_.

For error codes see Twitter's
`Error Codes and Responses <https://dev.twitter.com/overview/api/response-codes>`
"""
import datetime
import json
from functools import wraps
from io import StringIO
from nltk.twitter import Query, Streamer, TweetViewer, TweetWriter, Twitter, credsfromfile
SPACER = '###################################'

def verbose(func):
    if False:
        while True:
            i = 10
    'Decorator for demo functions'

    @wraps(func)
    def with_formatting(*args, **kwargs):
        if False:
            return 10
        print()
        print(SPACER)
        print('Using %s' % func.__name__)
        print(SPACER)
        return func(*args, **kwargs)
    return with_formatting

def yesterday():
    if False:
        print('Hello World!')
    "\n    Get yesterday's datetime as a 5-tuple.\n    "
    date = datetime.datetime.now()
    date -= datetime.timedelta(days=1)
    date_tuple = date.timetuple()[:6]
    return date_tuple

def setup():
    if False:
        i = 10
        return i + 15
    '\n    Initialize global variables for the demos.\n    '
    global USERIDS, FIELDS
    USERIDS = ['759251', '612473', '15108702', '6017542', '2673523800']
    FIELDS = ['id_str']

@verbose
def twitterclass_demo():
    if False:
        while True:
            i = 10
    '\n    Use the simplified :class:`Twitter` class to write some tweets to a file.\n    '
    tw = Twitter()
    print('Track from the public stream\n')
    tw.tweets(keywords='love, hate', limit=10)
    print(SPACER)
    print('Search past Tweets\n')
    tw = Twitter()
    tw.tweets(keywords='love, hate', stream=False, limit=10)
    print(SPACER)
    print('Follow two accounts in the public stream' + ' -- be prepared to wait a few minutes\n')
    tw = Twitter()
    tw.tweets(follow=['759251', '6017542'], stream=True, limit=5)

@verbose
def sampletoscreen_demo(limit=20):
    if False:
        print('Hello World!')
    '\n    Sample from the Streaming API and send output to terminal.\n    '
    oauth = credsfromfile()
    client = Streamer(**oauth)
    client.register(TweetViewer(limit=limit))
    client.sample()

@verbose
def tracktoscreen_demo(track='taylor swift', limit=10):
    if False:
        i = 10
        return i + 15
    '\n    Track keywords from the public Streaming API and send output to terminal.\n    '
    oauth = credsfromfile()
    client = Streamer(**oauth)
    client.register(TweetViewer(limit=limit))
    client.filter(track=track)

@verbose
def search_demo(keywords='nltk'):
    if False:
        while True:
            i = 10
    '\n    Use the REST API to search for past tweets containing a given keyword.\n    '
    oauth = credsfromfile()
    client = Query(**oauth)
    for tweet in client.search_tweets(keywords=keywords, limit=10):
        print(tweet['text'])

@verbose
def tweets_by_user_demo(user='NLTK_org', count=200):
    if False:
        i = 10
        return i + 15
    '\n    Use the REST API to search for past tweets by a given user.\n    '
    oauth = credsfromfile()
    client = Query(**oauth)
    client.register(TweetWriter())
    client.user_tweets(user, count)

@verbose
def lookup_by_userid_demo():
    if False:
        while True:
            i = 10
    '\n    Use the REST API to convert a userID to a screen name.\n    '
    oauth = credsfromfile()
    client = Query(**oauth)
    user_info = client.user_info_from_id(USERIDS)
    for info in user_info:
        name = info['screen_name']
        followers = info['followers_count']
        following = info['friends_count']
        print(f'{name}, followers: {followers}, following: {following}')

@verbose
def followtoscreen_demo(limit=10):
    if False:
        while True:
            i = 10
    '\n    Using the Streaming API, select just the tweets from a specified list of\n    userIDs.\n\n    This is will only give results in a reasonable time if the users in\n    question produce a high volume of tweets, and may even so show some delay.\n    '
    oauth = credsfromfile()
    client = Streamer(**oauth)
    client.register(TweetViewer(limit=limit))
    client.statuses.filter(follow=USERIDS)

@verbose
def streamtofile_demo(limit=20):
    if False:
        while True:
            i = 10
    '\n    Write 20 tweets sampled from the public Streaming API to a file.\n    '
    oauth = credsfromfile()
    client = Streamer(**oauth)
    client.register(TweetWriter(limit=limit, repeat=False))
    client.statuses.sample()

@verbose
def limit_by_time_demo(keywords='nltk'):
    if False:
        while True:
            i = 10
    '\n    Query the REST API for Tweets about NLTK since yesterday and send\n    the output to terminal.\n\n    This example makes the assumption that there are sufficient Tweets since\n    yesterday for the date to be an effective cut-off.\n    '
    date = yesterday()
    dt_date = datetime.datetime(*date)
    oauth = credsfromfile()
    client = Query(**oauth)
    client.register(TweetViewer(limit=100, lower_date_limit=date))
    print(f'Cutoff date: {dt_date}\n')
    for tweet in client.search_tweets(keywords=keywords):
        print('{} '.format(tweet['created_at']), end='')
        client.handler.handle(tweet)

@verbose
def corpusreader_demo():
    if False:
        return 10
    '\n    Use `TwitterCorpusReader` tp read a file of tweets, and print out\n\n    * some full tweets in JSON format;\n    * some raw strings from the tweets (i.e., the value of the `text` field); and\n    * the result of tokenising the raw strings.\n\n    '
    from nltk.corpus import twitter_samples as tweets
    print()
    print('Complete tweet documents')
    print(SPACER)
    for tweet in tweets.docs('tweets.20150430-223406.json')[:1]:
        print(json.dumps(tweet, indent=1, sort_keys=True))
    print()
    print('Raw tweet strings:')
    print(SPACER)
    for text in tweets.strings('tweets.20150430-223406.json')[:15]:
        print(text)
    print()
    print('Tokenized tweet strings:')
    print(SPACER)
    for toks in tweets.tokenized('tweets.20150430-223406.json')[:15]:
        print(toks)

@verbose
def expand_tweetids_demo():
    if False:
        print('Hello World!')
    '\n    Given a file object containing a list of Tweet IDs, fetch the\n    corresponding full Tweets, if available.\n\n    '
    ids_f = StringIO('        588665495492124672\n        588665495487909888\n        588665495508766721\n        588665495513006080\n        588665495517200384\n        588665495487811584\n        588665495525588992\n        588665495487844352\n        588665495492014081\n        588665495512948737')
    oauth = credsfromfile()
    client = Query(**oauth)
    hydrated = client.expand_tweetids(ids_f)
    for tweet in hydrated:
        id_str = tweet['id_str']
        print(f'id: {id_str}')
        text = tweet['text']
        if text.startswith('@null'):
            text = '[Tweet not available]'
        print(text + '\n')
ALL = [twitterclass_demo, sampletoscreen_demo, tracktoscreen_demo, search_demo, tweets_by_user_demo, lookup_by_userid_demo, followtoscreen_demo, streamtofile_demo, limit_by_time_demo, corpusreader_demo, expand_tweetids_demo]
'\nSelect demo functions to run. E.g. replace the following line with "DEMOS =\nALL[8:]" to execute only the final three demos.\n'
DEMOS = ALL[:]
if __name__ == '__main__':
    setup()
    for demo in DEMOS:
        demo()
    print('\n' + SPACER)
    print('All demos completed')
    print(SPACER)