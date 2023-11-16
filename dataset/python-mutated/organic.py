from r2.models import *
from r2.lib.normalized_hot import normalized_hot
from r2.lib import count
from r2.lib.utils import UniqueIterator, timeago
import random
from time import time
organic_max_length = 50

def cached_organic_links(*sr_ids):
    if False:
        i = 10
        return i + 15
    sr_count = count.get_link_counts()
    link_names = filter(lambda n: sr_count[n][1] in sr_ids, sr_count.keys())
    link_names.sort(key=lambda n: sr_count[n][0])
    if not link_names and g.debug:
        q = All.get_links('new', 'all')
        q._limit = 100
        link_names = [x._fullname for x in q if x.promoted is None]
        g.log.debug('Used inorganic links')
    if random.choice((True, False)) and sr_ids:
        sr_id = random.choice(sr_ids)
        fnames = normalized_hot([sr_id])
        if fnames:
            if len(fnames) == 1:
                new_item = fnames[0]
            else:
                new_item = random.choice(fnames[1:4])
            link_names.insert(0, new_item)
    return link_names

def organic_links(user):
    if False:
        return 10
    sr_ids = Subreddit.user_subreddits(user)
    sr_ids.sort()
    user_id = None if isinstance(user, FakeAccount) else user
    sr_ids = Subreddit.user_subreddits(user, True)
    sr_ids.sort()
    return cached_organic_links(*sr_ids)[:organic_max_length]