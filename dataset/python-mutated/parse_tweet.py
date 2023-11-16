import re

class Emoticons:
    POSITIVE = ['*O', '*-*', '*O*', '*o*', '* *', ':P', ':D', ':d', ':p', ';P', ';D', ';d', ';p', ':-)', ';-)', ':=)', ';=)', ':<)', ':>)', ';>)', ';=)', '=}', ':)', '(:;)', '(;', ':}', '{:', ';}', '{;:]', '[;', ":')", ";')", ':-3', '{;', ':]', ';-3', ':-x', ';-x', ':-X', ';-X', ':-}', ';-=}', ':-]', ';-]', ':-.)', '^_^', '^-^']
    NEGATIVE = [':(', ';(', ":'(", '=(', '={', '):', ');', ")':", ")';", ')=', '}=', ';-{{', ';-{', ':-{{', ':-{', ':-(', ';-(', ':,)', ":'{", '[:', ';]']

class ParseTweet(object):
    regexp = {'RT': '^RT', 'MT': '^MT', 'ALNUM': '(@[a-zA-Z0-9_]+)', 'HASHTAG': '(#[\\w\\d]+)', 'URL': '([https://|http://]?[a-zA-Z\\d\\/]+[\\.]+[a-zA-Z\\d\\/\\.]+)', 'SPACES': '\\s+'}
    regexp = dict(((key, re.compile(value)) for (key, value) in regexp.items()))

    def __init__(self, timeline_owner, tweet):
        if False:
            print('Hello World!')
        ' timeline_owner : twitter handle of user account. tweet - 140 chars from feed; object does all computation on construction\n            properties:\n            RT, MT - boolean\n            URLs - list of URL\n            Hashtags - list of tags\n        '
        self.Owner = timeline_owner
        self.tweet = tweet
        self.UserHandles = ParseTweet.getUserHandles(tweet)
        self.Hashtags = ParseTweet.getHashtags(tweet)
        self.URLs = ParseTweet.getURLs(tweet)
        self.RT = ParseTweet.getAttributeRT(tweet)
        self.MT = ParseTweet.getAttributeMT(tweet)
        self.Emoticon = ParseTweet.getAttributeEmoticon(tweet)
        if self.RT and len(self.UserHandles) > 0:
            self.Owner = self.UserHandles[0]
        return

    def __str__(self):
        if False:
            i = 10
            return i + 15
        ' for display method '
        return 'owner %s, urls: %d, hashtags %d, user_handles %d, len_tweet %d, RT = %s, MT = %s' % (self.Owner, len(self.URLs), len(self.Hashtags), len(self.UserHandles), len(self.tweet), self.RT, self.MT)

    @staticmethod
    def getAttributeEmoticon(tweet):
        if False:
            i = 10
            return i + 15
        ' see if tweet is contains any emoticons, +ve, -ve or neutral '
        emoji = list()
        for tok in re.split(ParseTweet.regexp['SPACES'], tweet.strip()):
            if tok in Emoticons.POSITIVE:
                emoji.append(tok)
                continue
            if tok in Emoticons.NEGATIVE:
                emoji.append(tok)
        return emoji

    @staticmethod
    def getAttributeRT(tweet):
        if False:
            print('Hello World!')
        ' see if tweet is a RT '
        return re.search(ParseTweet.regexp['RT'], tweet.strip()) is not None

    @staticmethod
    def getAttributeMT(tweet):
        if False:
            for i in range(10):
                print('nop')
        ' see if tweet is a MT '
        return re.search(ParseTweet.regexp['MT'], tweet.strip()) is not None

    @staticmethod
    def getUserHandles(tweet):
        if False:
            i = 10
            return i + 15
        ' given a tweet we try and extract all user handles in order of occurrence'
        return re.findall(ParseTweet.regexp['ALNUM'], tweet)

    @staticmethod
    def getHashtags(tweet):
        if False:
            i = 10
            return i + 15
        ' return all hashtags'
        return re.findall(ParseTweet.regexp['HASHTAG'], tweet)

    @staticmethod
    def getURLs(tweet):
        if False:
            for i in range(10):
                print('nop')
        ' URL : [http://]?[\\w\\.?/]+'
        return re.findall(ParseTweet.regexp['URL'], tweet)