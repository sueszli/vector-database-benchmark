from itertools import chain
import math
import random
from collections import defaultdict
from datetime import timedelta
from operator import itemgetter
from pycassa.types import LongType
from r2.lib import rising
from r2.lib.db import operators, tdb_cassandra
from r2.lib.pages import ExploreItem
from r2.lib.normalized_hot import normalized_hot
from r2.lib.utils import roundrobin, tup, to36
from r2.models import Link, Subreddit
from r2.models.builder import CommentBuilder
from r2.models.listing import NestedListing
from r2.models.recommend import AccountSRPrefs, AccountSRFeedback
from pylons import app_globals as g
from pylons.i18n import _
SRC_MULTIREDDITS = 'mr'
SRC_EXPLORE = 'e'
TYPE_RISING = _('rising')
TYPE_DISCOVERY = _('discovery')
TYPE_HOT = _('hot')
TYPE_COMMENT = _('comment')

def get_recommendations(srs, count=10, source=SRC_MULTIREDDITS, to_omit=None, match_set=True, over18=False):
    if False:
        for i in range(10):
            print('nop')
    'Return subreddits recommended if you like the given subreddits.\n\n    Args:\n    - srs is one Subreddit object or a list of Subreddits\n    - count is total number of results to return\n    - source is a prefix telling which set of recommendations to use\n    - to_omit is a single or list of subreddit id36s that should not be\n        be included. (Useful for omitting recs that were already rejected.)\n    - match_set=True will return recs that are similar to each other, useful\n        for matching the "theme" of the original set\n    - over18 content is filtered unless over18=True or one of the original srs\n        is over18\n\n    '
    srs = tup(srs)
    to_omit = tup(to_omit) if to_omit else []
    rec_id36s = SRRecommendation.for_srs([sr._id36 for sr in srs], to_omit, count * 2, source, match_set=match_set)
    rec_srs = Subreddit._byID36(rec_id36s, return_dict=False)
    filtered = [sr for sr in rec_srs if is_visible(sr)]
    if not over18 and (not any((sr.over_18 for sr in srs))):
        filtered = [sr for sr in filtered if not sr.over_18]
    return filtered[:count]

def get_recommended_content_for_user(account, settings, record_views=False, src=SRC_EXPLORE):
    if False:
        i = 10
        return i + 15
    "Wrapper around get_recommended_content() that fills in user info.\n\n    If record_views == True, the srs will be noted in the user's preferences\n    to keep from showing them again too soon.\n\n    settings is an ExploreSettings object that controls what types of content\n    will be included.\n\n    Returns a list of ExploreItems.\n\n    "
    prefs = AccountSRPrefs.for_user(account)
    recs = get_recommended_content(prefs, src, settings)
    if record_views:
        sr_data = {r.sr: r.src for r in recs}
        AccountSRFeedback.record_views(account, sr_data)
    return recs

def get_recommended_content(prefs, src, settings):
    if False:
        for i in range(10):
            print('nop')
    'Get a mix of content from subreddits recommended for someone with\n    the given preferences (likes and dislikes.)\n\n    Returns a list of ExploreItems.\n\n    '
    num_liked = 10
    num_recs = 20
    num_discovery = 2
    num_rising = 4
    num_items = 20
    rising_items = discovery_items = comment_items = hot_items = []
    default_srid36s = [to36(srid) for srid in Subreddit.default_subreddits()]
    omit_srid36s = list(prefs.likes.union(prefs.dislikes, prefs.recent_views, default_srid36s))
    liked_srid36s = random_sample(prefs.likes, num_liked) if settings.personalized else []
    candidates = set(get_discovery_srid36s()).difference(prefs.dislikes)
    discovery_srid36s = random_sample(candidates, num_discovery)
    to_fetch = liked_srid36s + discovery_srid36s
    srs = Subreddit._byID36(to_fetch)
    liked_srs = [srs[sr_id36] for sr_id36 in liked_srid36s]
    discovery_srs = [srs[sr_id36] for sr_id36 in discovery_srid36s]
    if settings.personalized:
        recommended_srs = get_recommendations(liked_srs, count=num_recs, to_omit=omit_srid36s, source=src, match_set=False, over18=settings.nsfw)
        random.shuffle(recommended_srs)
        midpoint = len(recommended_srs) / 2
        srs_slice1 = recommended_srs[:midpoint]
        srs_slice2 = recommended_srs[midpoint:]
        comment_items = get_comment_items(srs_slice1, src)
        hot_items = get_hot_items(srs_slice2, TYPE_HOT, src)
    if settings.discovery:
        discovery_items = get_hot_items(discovery_srs, TYPE_DISCOVERY, 'disc')
    if settings.rising:
        omit_sr_ids = set((int(id36, 36) for id36 in omit_srid36s))
        rising_items = get_rising_items(omit_sr_ids, count=num_rising)
    all_recs = list(chain(rising_items, comment_items, discovery_items, hot_items))
    random.shuffle(all_recs)
    seen_srs = set()
    recs = []
    for r in all_recs:
        if not settings.nsfw and r.is_over18():
            continue
        if not is_visible(r.sr):
            continue
        if r.sr._id not in seen_srs:
            recs.append(r)
            seen_srs.add(r.sr._id)
        if len(recs) >= num_items:
            break
    return recs

def get_hot_items(srs, item_type, src):
    if False:
        while True:
            i = 10
    'Get hot links from specified srs.'
    hot_srs = {sr._id: sr for sr in srs}
    hot_link_fullnames = normalized_hot([sr._id for sr in srs])
    hot_links = Link._by_fullname(hot_link_fullnames, return_dict=False)
    hot_items = []
    for l in hot_links:
        hot_items.append(ExploreItem(item_type, src, hot_srs[l.sr_id], l))
    return hot_items

def get_rising_items(omit_sr_ids, count=4):
    if False:
        return 10
    'Get links that are rising right now.'
    all_rising = rising.get_all_rising()
    candidate_sr_ids = {sr_id for (link, score, sr_id) in all_rising}.difference(omit_sr_ids)
    link_fullnames = [link for (link, score, sr_id) in all_rising if sr_id in candidate_sr_ids]
    link_fullnames_to_show = random_sample(link_fullnames, count)
    rising_links = Link._by_fullname(link_fullnames_to_show, return_dict=False, data=True)
    rising_items = [ExploreItem(TYPE_RISING, 'ris', Subreddit._byID(l.sr_id), l) for l in rising_links]
    return rising_items

def get_comment_items(srs, src, count=4):
    if False:
        print('Hello World!')
    'Get hot links from srs, plus top comment from each link.'
    link_fullnames = normalized_hot([sr._id for sr in srs])
    hot_links = Link._by_fullname(link_fullnames[:count], return_dict=False)
    top_comments = []
    for link in hot_links:
        builder = CommentBuilder(link, operators.desc('_confidence'), comment=None, context=None, num=1, load_more=False)
        listing = NestedListing(builder, parent_name=link._fullname).listing()
        top_comments.extend(listing.things)
    srs = Subreddit._byID([com.sr_id for com in top_comments])
    links = Link._byID([com.link_id for com in top_comments])
    comment_items = [ExploreItem(TYPE_COMMENT, src, srs[com.sr_id], links[com.link_id], com) for com in top_comments]
    return comment_items

def get_discovery_srid36s():
    if False:
        while True:
            i = 10
    'Get list of srs that help people discover other srs.'
    srs = Subreddit._by_name(g.live_config['discovery_srs'])
    return [sr._id36 for sr in srs.itervalues()]

def random_sample(items, count):
    if False:
        print('Hello World!')
    "Safe random sample that won't choke if len(items) < count."
    sample_size = min(count, len(items))
    return random.sample(items, sample_size)

def is_visible(sr):
    if False:
        i = 10
        return i + 15
    'True if sr is visible to regular users, false if private or banned.'
    return sr.type not in Subreddit.private_types and (not sr._spam) and sr.discoverable

class SRRecommendation(tdb_cassandra.View):
    _use_db = True
    _compare_with = LongType()
    _ttl = timedelta(days=7, hours=12)
    _warn_on_partial_ttl = False

    @classmethod
    def for_srs(cls, srid36, to_omit, count, source, match_set=True):
        if False:
            return 10
        srid36s = tup(srid36)
        to_omit = set(to_omit)
        to_omit.update(srid36s)
        rowkeys = ['%s.%s' % (source, srid36) for srid36 in srid36s]
        rows = cls._byID(rowkeys, return_dict=False)
        if match_set:
            sorted_recs = cls._merge_and_sort_by_count(rows)
            min_count = math.floor(0.1 * len(srid36s))
            sorted_recs = (rec[0] for rec in sorted_recs if rec[1] > min_count)
        else:
            sorted_recs = cls._merge_roundrobin(rows)
        filtered = []
        for r in sorted_recs:
            if r not in to_omit:
                filtered.append(r)
                to_omit.add(r)
        return filtered[:count]

    @classmethod
    def _merge_roundrobin(cls, rows):
        if False:
            for i in range(10):
                print('nop')
        'Combine multiple sets of recs, preserving order.\n\n        Picks items equally from each input sr, which can be useful for\n        getting a diverse set of recommendations instead of one that matches\n        a theme. Preserves ordering, so all rank 1 recs will be listed first,\n        then all rank 2, etc.\n\n        Returns a list of id36s.\n\n        '
        return roundrobin(*[row._values().itervalues() for row in rows])

    @classmethod
    def _merge_and_sort_by_count(cls, rows):
        if False:
            while True:
                i = 10
        'Combine and sort multiple sets of recs.\n\n        Combines multiple sets of recs and sorts by number of times each rec\n        appears, the reasoning being that an item recommended for several of\n        the original srs is more likely to match the "theme" of the set.\n\n        '
        rank_id36_pairs = chain.from_iterable((row._values().iteritems() for row in rows))
        ranks = defaultdict(list)
        for (rank, id36) in rank_id36_pairs:
            ranks[id36].append(rank)
        recs = [(id36, len(ranks), max(ranks)) for (id36, ranks) in ranks.iteritems()]
        recs = sorted(recs, key=itemgetter(2))
        return sorted(recs, key=itemgetter(1), reverse=True)