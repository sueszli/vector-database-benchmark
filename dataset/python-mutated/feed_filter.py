from django.contrib.auth.models import User
from apps.rss_feeds.models import Feed
from apps.reader.models import UserSubscription
from apps.analyzer.models import Category, FeatureCategory
import datetime
import re
import math

def entry_features(self, entry):
    if False:
        return 10
    splitter = re.compile('\\W*')
    f = {}
    titlewords = [s.lower() for s in splitter.split(entry['title']) if len(s) > 2 and len(s) < 20]
    for w in titlewords:
        f['Title:' + w] = 1
    summarywords = [s.lower() for s in splitter.split(entry['summary']) if len(s) > 2 and len(s) < 20]
    uc = 0
    for i in range(len(summarywords)):
        w = summarywords[i]
        f[w] = 1
        if w.isupper():
            uc += 1
        if i < len(summarywords) - 1:
            twowords = ' '.join(summarywords[i:i + 1])
            f[twowords] = 1
    f['Publisher:' + entry['publisher']] = 1
    if float(uc) / len(summarywords) > 0.3:
        f['UPPERCASE'] = 1
    return f