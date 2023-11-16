from pylons import request
from pylons import tmpl_context as c
from r2.lib.strings import Score
from r2.lib import hooks

class Printable(object):
    show_spam = False
    show_reports = False
    is_special = False
    can_ban = False
    deleted = False
    rowstyle_cls = ''
    collapsed = False
    author = None
    margin = 0
    is_focal = False
    childlisting = None
    cache_ignore = set(['c', 'author', 'score_fmt', 'child', 'voting_score', 'display_score', 'render_score', 'score', '_score', 'upvotes', '_ups', 'downvotes', '_downs', 'subreddit_slow', '_deleted', '_spam', 'cachable', 'make_permalink', 'permalink', 'timesince', 'num', 'rowstyle_cls', 'upvote_ratio', 'should_incr_counts', 'keep_item'])

    @classmethod
    def update_nofollow(cls, user, wrapped):
        if False:
            print('Hello World!')
        pass

    @classmethod
    def add_props(cls, user, wrapped):
        if False:
            while True:
                i = 10
        from r2.lib.wrapped import CachedVariable
        for item in wrapped:
            item.display = CachedVariable('display')
            item.timesince = CachedVariable('timesince')
            item.childlisting = CachedVariable('childlisting')
            score_fmt = getattr(item, 'score_fmt', Score.number_only)
            item.display_score = map(score_fmt, item.voting_score)
            if item.cachable:
                item.render_score = item.display_score
                item.display_score = map(CachedVariable, ['scoredislikes', 'scoreunvoted', 'scorelikes'])
        hooks.get_hook('add_props').call(items=wrapped)

    @property
    def permalink(self, *a, **kw):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def keep_item(self, wrapped):
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def wrapped_cache_key(wrapped, style):
        if False:
            for i in range(10):
                print('nop')
        s = [wrapped._fullname, wrapped._spam]
        if c.site.flair_enabled and c.user.pref_show_flair:
            s.append('user_flair_enabled')
        if style == 'htmllite':
            s.extend([c.bgcolor, c.bordercolor, request.GET.has_key('style'), request.GET.get('expanded'), getattr(wrapped, 'embed_voting_style', None)])
        return s