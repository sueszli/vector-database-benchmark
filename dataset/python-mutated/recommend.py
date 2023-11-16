import pycassa
import time
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import chain
from pylons import app_globals as g
from r2.lib.db import tdb_cassandra
from r2.lib.db.tdb_cassandra import max_column_count
from r2.lib.utils import utils, tup
from r2.models import Account, LabeledMulti, Subreddit
from r2.lib.pages import ExploreItem
VIEW = 'imp'
CLICK = 'clk'
DISMISS = 'dis'
FEEDBACK_ACTIONS = [VIEW, CLICK, DISMISS]
FEEDBACK_TTL = {VIEW: timedelta(hours=6).total_seconds(), CLICK: timedelta(minutes=30).total_seconds(), DISMISS: timedelta(days=60).total_seconds()}

class AccountSRPrefs(object):
    """Class for managing user recommendation preferences.

    Builds a user profile on-the-fly based on the user's subscriptions,
    multireddits, and recent interactions with the recommender UI.

    Likes are used to generate recommendations, dislikes to filter out
    unwanted results, and recent views to make sure the same subreddits aren't
    recommended too often.

    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.likes = set()
        self.dislikes = set()
        self.recent_views = set()

    @classmethod
    def for_user(cls, account):
        if False:
            return 10
        "Return a new AccountSRPrefs obj populated with user's data."
        prefs = cls()
        multis = LabeledMulti.by_owner(account)
        multi_srs = set(chain.from_iterable((multi.srs for multi in multis)))
        feedback = AccountSRFeedback.for_user(account)
        subscriptions = Subreddit.user_subreddits(account, limit=None)
        prefs.likes.update((utils.to36(sr_id) for sr_id in subscriptions))
        prefs.likes.update((sr._id36 for sr in multi_srs))
        prefs.likes.update(feedback[CLICK])
        prefs.dislikes.update(feedback[DISMISS])
        prefs.likes = prefs.likes.difference(prefs.dislikes)
        prefs.recent_views.update(feedback[VIEW])
        return prefs

class AccountSRFeedback(tdb_cassandra.DenormalizedRelation):
    """Column family for storing users' recommendation feedback."""
    _use_db = True
    _views = []
    _write_last_modified = False
    _read_consistency_level = tdb_cassandra.CL.QUORUM
    _write_consistency_level = tdb_cassandra.CL.QUORUM

    @classmethod
    def for_user(cls, account):
        if False:
            for i in range(10):
                print('nop')
        'Return dict mapping each feedback type to a set of sr id36s.'
        feedback = defaultdict(set)
        try:
            row = AccountSRFeedback._cf.get(account._id36, column_count=max_column_count)
        except pycassa.NotFoundException:
            return feedback
        for (colkey, colval) in row.iteritems():
            (action, sr_id36) = colkey.split('.')
            feedback[action].add(sr_id36)
        return feedback

    @classmethod
    def record_feedback(cls, account, srs, action):
        if False:
            print('Hello World!')
        if action not in FEEDBACK_ACTIONS:
            g.log.error('Unrecognized feedback: %s' % action)
            return
        srs = tup(srs)
        fb_rowkey = account._id36
        fb_colkeys = ['%s.%s' % (action, sr._id36) for sr in srs]
        col_data = {col: '' for col in fb_colkeys}
        ttl = FEEDBACK_TTL.get(action, 0)
        if ttl > 0:
            AccountSRFeedback._cf.insert(fb_rowkey, col_data, ttl=ttl)
        else:
            AccountSRFeedback._cf.insert(fb_rowkey, col_data)

    @classmethod
    def record_views(cls, account, srs):
        if False:
            print('Hello World!')
        cls.record_feedback(account, srs, VIEW)

class ExploreSettings(tdb_cassandra.Thing):
    """Column family for storing users' view prefs for the /explore page."""
    _use_db = True
    _bool_props = ('personalized', 'discovery', 'rising', 'nsfw')

    @classmethod
    def for_user(cls, account):
        if False:
            i = 10
            return i + 15
        "Return user's prefs or default prefs if user has none."
        try:
            return cls._byID(account._id36)
        except tdb_cassandra.NotFound:
            return DefaultExploreSettings()

    @classmethod
    def record_settings(cls, user, personalized=False, discovery=False, rising=False, nsfw=False):
        if False:
            for i in range(10):
                print('nop')
        'Update or create settings for user.'
        try:
            settings = cls._byID(user._id36)
        except tdb_cassandra.NotFound:
            settings = ExploreSettings(_id=user._id36, personalized=personalized, discovery=discovery, rising=rising, nsfw=nsfw)
        else:
            settings.personalized = personalized
            settings.discovery = discovery
            settings.rising = rising
            settings.nsfw = nsfw
        settings._commit()

class DefaultExploreSettings(object):
    """Default values to use when no settings have been saved for the user."""

    def __init__(self):
        if False:
            return 10
        self.personalized = True
        self.discovery = True
        self.rising = True
        self.nsfw = False