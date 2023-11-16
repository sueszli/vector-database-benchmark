from r2.lib import count
from r2.models import Subreddit

def set_downs():
    if False:
        i = 10
        return i + 15
    sr_counts = count.get_sr_counts()
    names = [k for (k, v) in sr_counts.iteritems() if v != 0]
    srs = Subreddit._by_fullname(names)
    for name in names:
        (sr, c) = (srs[name], sr_counts[name])
        if c != sr._downs and c > 0:
            sr._downs = max(c, 0)
            sr._commit()

def run():
    if False:
        return 10
    set_downs()