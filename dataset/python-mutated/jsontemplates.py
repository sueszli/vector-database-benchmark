import calendar
from collections import defaultdict
from utils import to36, tup, iters
from wrapped import Wrapped, StringTemplate, CacheStub, Templated
from mako.template import Template
from r2.config import feature
from r2.config.extensions import get_api_subtype
from r2.lib import hooks
from r2.lib.filters import spaceCompress, safemarkdown, _force_unicode
from r2.models import Account, Comment, Link, Report, Subreddit, SubredditUserRelations, Trophy
from r2.models.token import OAuth2Scope, extra_oauth2_scope
import time, pytz
from pylons import response
from pylons import tmpl_context as c
from pylons import app_globals as g
from pylons.i18n import _
from r2.models.wiki import ImagesByWikiPage

def make_typename(typ):
    if False:
        print('Hello World!')
    return 't%s' % to36(typ._type_id)

def make_fullname(typ, _id):
    if False:
        print('Hello World!')
    return '%s_%s' % (make_typename(typ), to36(_id))

class ObjectTemplate(StringTemplate):

    def __init__(self, d):
        if False:
            return 10
        self.d = d

    def update(self, kw):
        if False:
            while True:
                i = 10

        def _update(obj):
            if False:
                return 10
            if isinstance(obj, (str, unicode)):
                return _force_unicode(obj)
            elif isinstance(obj, dict):
                return dict(((k, _update(v)) for (k, v) in obj.iteritems()))
            elif isinstance(obj, (list, tuple)):
                return map(_update, obj)
            elif isinstance(obj, CacheStub) and kw.has_key(obj.name):
                return kw[obj.name]
            else:
                return obj
        res = _update(self.d)
        return ObjectTemplate(res)

    def finalize(self, kw={}):
        if False:
            i = 10
            return i + 15
        return self.update(kw).d

class JsonTemplate(Template):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def render(self, thing=None, *a, **kw):
        if False:
            for i in range(10):
                print('nop')
        return ObjectTemplate({})

class TakedownJsonTemplate(JsonTemplate):

    def render(self, thing=None, *a, **kw):
        if False:
            while True:
                i = 10
        return thing.explanation

class ThingTemplate(object):

    @classmethod
    def render(cls, thing):
        if False:
            return 10
        '\n        Return a JSON representation of a Wrapped Thing object.\n\n        The Thing object should be Wrapped and been run through add_props just\n        like is required for regular HTML rendering. The return value is an\n        ObjectTemplate wrapped dictionary.\n\n        '
        api_subtype = get_api_subtype()
        item = thing
        if api_subtype:
            data = cls.get_rendered(item, render_style=api_subtype)
        else:
            data = cls.get_json(item)
        d = {'kind': cls.get_kind(item), 'data': data}
        return ObjectTemplate(d)

    @classmethod
    def get_kind(cls, item):
        if False:
            return 10
        thing = item.lookups[0]
        return make_typename(thing.__class__)

    @classmethod
    def get_json(cls, item):
        if False:
            while True:
                i = 10
        data = {'created': time.mktime(item._date.timetuple()), 'created_utc': time.mktime(item._date.astimezone(pytz.UTC).timetuple()) - time.timezone, 'id': item._id36, 'name': item._fullname}
        return data

    @classmethod
    def get_rendered(cls, item, render_style):
        if False:
            while True:
                i = 10
        data = {'id': item._fullname, 'content': item.render(style=render_style)}
        return data

class ThingJsonTemplate(JsonTemplate):
    _data_attrs_ = dict(created='created', created_utc='created_utc', id='_id36', name='_fullname')

    @classmethod
    def data_attrs(cls, **kw):
        if False:
            i = 10
            return i + 15
        d = cls._data_attrs_.copy()
        d.update(kw)
        return d

    def kind(self, wrapped):
        if False:
            while True:
                i = 10
        "\n        Returns a string literal which identifies the type of this\n        thing.  For subclasses of Thing, it will be 't's + kind_id.\n        "
        _thing = wrapped.lookups[0] if isinstance(wrapped, Wrapped) else wrapped
        return make_typename(_thing.__class__)

    def rendered_data(self, thing):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called only when get_api_type is non-None (i.e., a JSON\n        request has been made with partial rendering of the object to\n        be returned)\n\n        Canonical Thing data representation for JS, which is currently\n        a dictionary of three elements (translated into a JS Object\n        when sent out).  The elements are:\n\n         * id : Thing _fullname of thing.\n         * content : rendered  representation of the thing by\n           calling render on it using the style of get_api_subtype().\n        '
        res = dict(id=thing._fullname, content=thing.render(style=get_api_subtype()))
        return res

    def raw_data(self, thing):
        if False:
            while True:
                i = 10
        '\n        Complement to rendered_data.  Called when a dictionary of\n        thing data attributes is to be sent across the wire.\n        '
        attrs = dict(self._data_attrs_)
        if hasattr(self, '_optional_data_attrs'):
            for (attr, attrv) in self._optional_data_attrs.iteritems():
                if hasattr(thing, attr):
                    attrs[attr] = attrv
        return dict(((k, self.thing_attr(thing, v)) for (k, v) in attrs.iteritems()))

    def thing_attr(self, thing, attr):
        if False:
            while True:
                i = 10
        "\n        For the benefit of subclasses, to lookup attributes which may\n        require more work than a simple getattr (for example, 'author'\n        which has to be gotten from the author_id attribute on most\n        things).\n        "
        if attr == 'author':
            if thing.author._deleted:
                return '[deleted]'
            return thing.author.name
        if attr == 'created':
            return time.mktime(thing._date.timetuple())
        elif attr == 'created_utc':
            return time.mktime(thing._date.astimezone(pytz.UTC).timetuple()) - time.timezone
        elif attr == 'child':
            child = getattr(thing, 'child', None)
            if child:
                return child.render()
            else:
                return ''
        if attr == 'distinguished':
            distinguished = getattr(thing, attr, 'no')
            if distinguished == 'no':
                return None
            return distinguished
        return getattr(thing, attr, None)

    def data(self, thing):
        if False:
            while True:
                i = 10
        if get_api_subtype():
            return self.rendered_data(thing)
        else:
            return self.raw_data(thing)

    def render(self, thing=None, action=None, *a, **kw):
        if False:
            return 10
        return ObjectTemplate(dict(kind=self.kind(thing), data=self.data(thing)))

class SubredditJsonTemplate(ThingJsonTemplate):
    _data_attrs_ = ThingJsonTemplate.data_attrs(accounts_active='accounts_active_count', banner_img='banner_img', banner_size='banner_size', collapse_deleted_comments='collapse_deleted_comments', comment_score_hide_mins='comment_score_hide_mins', community_rules='community_rules', description='description', description_html='description_html', display_name='name', header_img='header', header_size='header_size', header_title='header_title', icon_img='icon_img', icon_size='icon_size', lang='lang', over18='over_18', public_description='public_description', public_description_html='public_description_html', public_traffic='public_traffic', hide_ads='hide_ads', quarantine='quarantine', show_media='show_media', show_media_preview='show_media_preview', submission_type='link_type', submit_link_label='submit_link_label', submit_text_label='submit_text_label', submit_text='submit_text', submit_text_html='submit_text_html', subreddit_type='type', subscribers='_ups', suggested_comment_sort='suggested_comment_sort', title='title', url='path', user_is_banned='is_banned', user_is_muted='is_muted', user_is_contributor='is_contributor', user_is_moderator='is_moderator', user_is_subscriber='is_subscriber', user_sr_theme_enabled='user_sr_style_enabled', wiki_enabled='wiki_enabled')
    _public_attrs = {'_id36', '_fullname', 'created', 'created_utc', 'name', 'path', 'public_description', 'public_description_html', 'title', 'type'}

    def raw_data(self, thing):
        if False:
            print('Hello World!')
        data = ThingJsonTemplate.raw_data(self, thing)
        if feature.is_enabled('mobile_settings'):
            data['key_color'] = self.thing_attr(thing, 'key_color')
        if feature.is_enabled('related_subreddits'):
            data['related_subreddits'] = self.thing_attr(thing, 'related_subreddits')
        permissions = getattr(thing, 'mod_permissions', None)
        if permissions:
            permissions = [perm for (perm, has) in permissions.iteritems() if has]
            data['mod_permissions'] = permissions
        return data

    def thing_attr(self, thing, attr):
        if False:
            while True:
                i = 10
        if attr not in self._public_attrs and (not thing.can_view(c.user)):
            return None
        if attr == '_ups' and (thing.hide_subscribers or thing.hide_num_users_info):
            return 0
        elif attr == 'description_html':
            return safemarkdown(thing.description)
        elif attr == 'public_description_html':
            return safemarkdown(thing.public_description)
        elif attr == 'is_moderator':
            if c.user_is_loggedin:
                return thing.moderator
            return None
        elif attr == 'is_contributor':
            if c.user_is_loggedin:
                return thing.contributor
            return None
        elif attr == 'is_subscriber':
            if c.user_is_loggedin:
                return thing.subscriber
            return None
        elif attr == 'is_banned':
            if c.user_is_loggedin:
                return thing.banned
            return None
        elif attr == 'is_muted':
            if c.user_is_loggedin:
                return thing.muted
            return None
        elif attr == 'submit_text_html':
            return safemarkdown(thing.submit_text)
        elif attr == 'user_sr_style_enabled':
            if c.user_is_loggedin:
                return c.user.use_subreddit_style(thing)
            else:
                return True
        elif attr == 'wiki_enabled':
            is_admin_or_mod = c.user_is_loggedin and (c.user_is_admin or thing.is_moderator_with_perms(c.user, 'wiki'))
            return thing.wikimode == 'anyone' or (thing.wikimode == 'modonly' and is_admin_or_mod)
        else:
            return ThingJsonTemplate.thing_attr(self, thing, attr)

class LabeledMultiDescriptionJsonTemplate(ThingJsonTemplate):
    _data_attrs_ = dict(body_html='description_html', body_md='description_md')

    def kind(self, wrapped):
        if False:
            i = 10
            return i + 15
        return 'LabeledMultiDescription'

    def thing_attr(self, thing, attr):
        if False:
            while True:
                i = 10
        if attr == 'description_html':
            description_html = safemarkdown(thing.description_md) or ''
            return description_html
        else:
            return ThingJsonTemplate.thing_attr(self, thing, attr)

class LabeledMultiJsonTemplate(LabeledMultiDescriptionJsonTemplate):
    _data_attrs_ = ThingJsonTemplate.data_attrs(can_edit='can_edit', copied_from='copied_from', description_html='description_html', description_md='description_md', display_name='display_name', key_color='key_color', icon_name='icon_id', icon_url='icon_url', name='name', path='path', subreddits='srs', visibility='visibility', weighting_scheme='weighting_scheme')
    del _data_attrs_['id']

    def __init__(self, expand_srs=False):
        if False:
            i = 10
            return i + 15
        super(LabeledMultiJsonTemplate, self).__init__()
        self.expand_srs = expand_srs

    def kind(self, wrapped):
        if False:
            for i in range(10):
                print('nop')
        return 'LabeledMulti'

    @classmethod
    def sr_props(cls, thing, srs, expand=False):
        if False:
            print('Hello World!')
        sr_props = dict(thing.sr_props)
        if expand:
            sr_dicts = get_trimmed_sr_dicts(srs, c.user)
            for sr in srs:
                sr_props[sr._id]['data'] = sr_dicts[sr._id]
        return [dict(sr_props[sr._id], name=sr.name) for sr in srs]

    def thing_attr(self, thing, attr):
        if False:
            for i in range(10):
                print('nop')
        if attr == 'srs':
            return self.sr_props(thing, thing.srs, expand=self.expand_srs)
        elif attr == 'can_edit':
            return c.user_is_loggedin and thing.can_edit(c.user)
        elif attr == 'copied_from':
            if thing.can_edit(c.user):
                return thing.copied_from
            else:
                return None
        elif attr == 'display_name':
            return thing.display_name or thing.name
        else:
            super_ = super(LabeledMultiJsonTemplate, self)
            return super_.thing_attr(thing, attr)

def get_trimmed_sr_dicts(srs, user):
    if False:
        i = 10
        return i + 15
    if c.user_is_loggedin:
        sr_user_relations = Subreddit.get_sr_user_relations(user, srs)
    else:
        NO_SR_USER_RELATIONS = SubredditUserRelations(subscriber=None, moderator=None, contributor=None, banned=None, muted=None)
        sr_user_relations = defaultdict(lambda : NO_SR_USER_RELATIONS)
    ret = {}
    for sr in srs:
        relations = sr_user_relations[sr._id]
        can_view = sr.can_view(user)
        subscribers = sr._ups if not sr.hide_subscribers else 0
        data = dict(name=sr._fullname, display_name=sr.name, url=sr.path, banner_img=sr.banner_img if can_view else None, banner_size=sr.banner_size if can_view else None, header_img=sr.header if can_view else None, header_size=sr.header_size if can_view else None, icon_img=sr.icon_img if can_view else None, icon_size=sr.icon_size if can_view else None, key_color=sr.key_color if can_view else None, subscribers=subscribers if can_view else None, user_is_banned=relations.banned if can_view else None, user_is_muted=relations.muted if can_view else None, user_is_contributor=relations.contributor if can_view else None, user_is_moderator=relations.moderator if can_view else None, user_is_subscriber=relations.subscriber if can_view else None)
        if feature.is_enabled('mobile_settings'):
            data['key_color'] = sr.key_color if can_view else None
        ret[sr._id] = data
    return ret

class IdentityJsonTemplate(ThingJsonTemplate):
    _data_attrs_ = ThingJsonTemplate.data_attrs(comment_karma='comment_karma', has_verified_email='email_verified', is_gold='gold', is_mod='is_mod', link_karma='link_karma', name='name', hide_from_robots='pref_hide_from_robots')
    _private_data_attrs = dict(inbox_count='inbox_count', over_18='pref_over_18', gold_creddits='gold_creddits', gold_expiration='gold_expiration', is_suspended='in_timeout', suspension_expiration_utc='timeout_expiration_utc', features='features')
    _public_attrs = {'name', 'is_suspended'}

    def raw_data(self, thing):
        if False:
            i = 10
            return i + 15
        viewable = True
        attrs = self._data_attrs_.copy()
        if c.user_is_loggedin and thing._id == c.user._id:
            attrs.update(self._private_data_attrs)
        elif thing.in_timeout and thing.timeout_expiration is None:
            attrs.update({'is_suspended': 'in_timeout'})
            viewable = False
        if thing.pref_hide_from_robots:
            response.headers['X-Robots-Tag'] = 'noindex, nofollow'
        data = {k: self.thing_attr(thing, v) for (k, v) in attrs.iteritems() if viewable or k in self._public_attrs}
        try:
            self.add_message_data(data, thing)
        except OAuth2Scope.InsufficientScopeError:
            pass
        if c.user_is_loggedin and thing._id == c.user._id:
            data['is_employee'] = thing.employee
            data['in_beta'] = thing.pref_beta
        return data

    @extra_oauth2_scope('privatemessages')
    def add_message_data(self, data, thing):
        if False:
            print('Hello World!')
        if c.user_is_loggedin and thing._id == c.user._id:
            data['has_mail'] = self.thing_attr(thing, 'has_mail')
            data['has_mod_mail'] = self.thing_attr(thing, 'has_mod_mail')

    def thing_attr(self, thing, attr):
        if False:
            for i in range(10):
                print('nop')
        from r2.lib.template_helpers import display_comment_karma, display_link_karma
        if attr == 'is_mod':
            t = thing.lookups[0] if isinstance(thing, Wrapped) else thing
            return t.is_moderator_somewhere
        elif attr == 'has_mail':
            return bool(c.have_messages)
        elif attr == 'has_mod_mail':
            return bool(c.have_mod_messages)
        elif attr == 'comment_karma':
            return display_comment_karma(thing.comment_karma)
        elif attr == 'link_karma':
            return display_link_karma(thing.link_karma)
        elif attr == 'gold_expiration':
            if not thing.gold:
                return None
            return calendar.timegm(thing.gold_expiration.utctimetuple())
        elif attr == 'timeout_expiration_utc':
            expiration_date = thing.timeout_expiration
            if not expiration_date:
                return None
            return calendar.timegm(expiration_date.utctimetuple())
        elif attr == 'features':
            return feature.all_enabled(c.user)
        return ThingJsonTemplate.thing_attr(self, thing, attr)

class AccountJsonTemplate(IdentityJsonTemplate):
    _data_attrs_ = IdentityJsonTemplate.data_attrs(is_friend='is_friend')
    _private_data_attrs = dict(modhash='modhash', **IdentityJsonTemplate._private_data_attrs)

    def thing_attr(self, thing, attr):
        if False:
            return 10
        if attr == 'is_friend':
            return c.user_is_loggedin and thing._id in c.user.friends
        elif attr == 'modhash':
            return c.modhash
        return IdentityJsonTemplate.thing_attr(self, thing, attr)

class PrefsJsonTemplate(ThingJsonTemplate):
    _data_attrs_ = dict(((k[len('pref_'):], k) for k in Account._preference_attrs))

    def __init__(self, fields=None):
        if False:
            print('Hello World!')
        if fields is not None:
            _data_attrs_ = {}
            for field in fields:
                if field not in self._data_attrs_:
                    raise KeyError(field)
                _data_attrs_[field] = self._data_attrs_[field]
            self._data_attrs_ = _data_attrs_

    def thing_attr(self, thing, attr):
        if False:
            for i in range(10):
                print('nop')
        if attr == 'pref_clickgadget':
            return bool(thing.pref_clickgadget)
        return ThingJsonTemplate.thing_attr(self, thing, attr)

def get_mod_attributes(item):
    if False:
        return 10
    data = {}
    if c.user_is_loggedin and item.can_ban:
        data['num_reports'] = item.reported
        data['report_reasons'] = Report.get_reasons(item)
        ban_info = getattr(item, 'ban_info', {})
        if item._spam:
            data['approved_by'] = None
            if ban_info.get('moderator_banned'):
                data['banned_by'] = ban_info.get('banner')
            else:
                data['banned_by'] = True
        else:
            data['approved_by'] = ban_info.get('unbanner')
            data['banned_by'] = None
    else:
        data['num_reports'] = None
        data['report_reasons'] = None
        data['approved_by'] = None
        data['banned_by'] = None
    return data

def get_author_attributes(item):
    if False:
        i = 10
        return i + 15
    data = {}
    if not item.author._deleted:
        author = item.author
        sr_id = item.subreddit._id
        data['author'] = author.name
        if author.flair_enabled_in_sr(sr_id):
            flair_text = getattr(author, 'flair_%s_text' % sr_id, None)
            flair_css = getattr(author, 'flair_%s_css_class' % sr_id, None)
        else:
            flair_text = None
            flair_css = None
        data['author_flair_text'] = flair_text
        data['author_flair_css_class'] = flair_css
    else:
        data['author'] = '[deleted]'
        data['author_flair_text'] = None
        data['author_flair_css_class'] = None
    return data

def get_distinguished_attributes(item):
    if False:
        for i in range(10):
            print('nop')
    data = {}
    distinguished = getattr(item, 'distinguished', 'no')
    data['distinguished'] = distinguished if distinguished != 'no' else None
    return data

def get_edited_attributes(item):
    if False:
        i = 10
        return i + 15
    data = {}
    if isinstance(item.editted, bool):
        data['edited'] = item.editted
    else:
        editted_timetuple = item.editted.astimezone(pytz.UTC).timetuple()
        data['edited'] = time.mktime(editted_timetuple) - time.timezone
    return data

def get_report_reason_attributes(item):
    if False:
        while True:
            i = 10
    if c.user_is_loggedin and c.user.in_timeout:
        data = {'user_reports': [], 'mod_reports': []}
    else:
        data = {'user_reports': item.user_reports, 'mod_reports': item.mod_reports}
    return data

def get_removal_reason_attributes(item):
    if False:
        for i in range(10):
            print('nop')
    data = {}
    if getattr(item, 'admin_takedown', None):
        data['removal_reason'] = 'legal'
    else:
        data['removal_reason'] = None
    return data

def get_media_embed_attributes(item):
    if False:
        print('Hello World!')
    from r2.lib.media import get_media_embed
    data = {'media_embed': {}, 'secure_media_embed': {}}
    media_object = item.media_object
    if media_object and (not isinstance(media_object, basestring)):
        media_embed = get_media_embed(media_object)
        if media_embed:
            data['media_embed'] = {'scrolling': media_embed.scrolling, 'width': media_embed.width, 'height': media_embed.height, 'content': media_embed.content}
    secure_media_object = item.secure_media_object
    if secure_media_object and (not isinstance(secure_media_object, basestring)):
        secure_media_embed = get_media_embed(secure_media_object)
        if secure_media_embed:
            data['secure_media_embed'] = {'scrolling': secure_media_embed.scrolling, 'width': secure_media_embed.width, 'height': secure_media_embed.height, 'content': secure_media_embed.content}
    return data

def get_selftext_attributes(item):
    if False:
        for i in range(10):
            print('nop')
    data = {}
    if not item.expunged:
        data['selftext'] = item.selftext
        data['selftext_html'] = safemarkdown(item.selftext)
    else:
        data['selftext'] = '[removed]'
        data['selftext_html'] = safemarkdown(_('[removed]'))
    return data

def generate_image_links(preview_object, file_type=None, censor_nsfw=False):
    if False:
        print('Hello World!')
    PREVIEW_RESOLUTIONS = (108, 216, 320, 640, 960, 1080)
    PREVIEW_MAX_RATIO = 2
    source_width = preview_object['width']
    source_height = preview_object['height']
    source_ratio = float(source_height) / source_width
    max_ratio = float(PREVIEW_MAX_RATIO)
    preview_ratio = min(source_ratio, max_ratio)
    preview_resolutions = []
    for w in PREVIEW_RESOLUTIONS:
        if w > source_width:
            continue
        url = g.image_resizing_provider.resize_image(preview_object, w, file_type=file_type, censor_nsfw=censor_nsfw, max_ratio=PREVIEW_MAX_RATIO)
        h = int(w * preview_ratio)
        preview_resolutions.append({'url': url, 'width': w, 'height': h})
    url = g.image_resizing_provider.resize_image(preview_object, file_type=file_type, censor_nsfw=censor_nsfw)
    return {'source': {'url': url, 'width': source_width, 'height': source_height}, 'resolutions': preview_resolutions}

class LinkJsonTemplate(ThingTemplate):

    @classmethod
    def get_json(cls, item):
        if False:
            print('Hello World!')
        data = ThingTemplate.get_json(item)
        data.update(get_mod_attributes(item))
        data.update(get_author_attributes(item))
        data.update(get_distinguished_attributes(item))
        data.update(get_edited_attributes(item))
        data.update(get_media_embed_attributes(item))
        data.update(get_report_reason_attributes(item))
        data.update(get_removal_reason_attributes(item))
        data.update(get_selftext_attributes(item))
        data.update({'archived': not item.votable, 'visited': item.visited, 'clicked': False, 'contest_mode': item.contest_mode, 'domain': item.domain, 'downs': 0, 'gilded': item.gildings, 'hidden': item.hidden, 'hide_score': item.hide_score, 'is_self': item.is_self, 'likes': item.likes, 'link_flair_css_class': item.flair_css_class, 'link_flair_text': item.flair_text, 'locked': item.locked, 'media': item.media_object, 'secure_media': item.secure_media_object, 'num_comments': item.num_comments, 'over_18': item.over_18, 'quarantine': item.quarantine, 'permalink': item.permalink, 'saved': item.saved, 'score': item.score, 'stickied': item.stickied, 'subreddit': item.subreddit.name, 'subreddit_id': item.subreddit._fullname, 'suggested_sort': item.sort_if_suggested(sr=item.subreddit), 'thumbnail': item.thumbnail, 'title': item.title, 'ups': item.score, 'url': item.url})
        if hasattr(item, 'action_type'):
            data['action_type'] = item.action_type
        if hasattr(item, 'sr_detail'):
            data['sr_detail'] = item.sr_detail
        if hasattr(item, 'show_media'):
            data['show_media'] = item.show_media
        if c.permalink_page:
            data['upvote_ratio'] = item.upvote_ratio
        preview_object = item.preview_image
        if preview_object:
            preview_is_gif = preview_object.get('url', '').endswith('.gif')
            data['preview'] = {}
            data['post_hint'] = item.post_hint
            if preview_is_gif:
                images = generate_image_links(preview_object, file_type='jpg')
            else:
                images = generate_image_links(preview_object)
            images['id'] = preview_object['uid']
            images['variants'] = {}
            if item.nsfw:
                images['variants']['nsfw'] = generate_image_links(preview_object, censor_nsfw=True, file_type='png')
            if preview_is_gif:
                images['variants']['gif'] = generate_image_links(preview_object)
                images['variants']['mp4'] = generate_image_links(preview_object, file_type='mp4')
            data['preview']['images'] = [images]
        return data

    @classmethod
    def get_rendered(cls, item, render_style):
        if False:
            while True:
                i = 10
        data = ThingTemplate.get_rendered(item, render_style)
        data.update({'sr': item.subreddit._fullname})
        return data

class PromotedLinkJsonTemplate(LinkJsonTemplate):

    @classmethod
    def get_json(cls, item):
        if False:
            print('Hello World!')
        data = LinkJsonTemplate.get_json(item)
        data.update({'promoted': item.promoted, 'imp_pixel': getattr(item, 'imp_pixel', None), 'href_url': item.href_url, 'adserver_imp_pixel': getattr(item, 'adserver_imp_pixel', None), 'adserver_click_url': getattr(item, 'adserver_click_url', None), 'mobile_ad_url': item.mobile_ad_url, 'disable_comments': item.disable_comments, 'third_party_tracking': item.third_party_tracking, 'third_party_tracking_2': item.third_party_tracking_2})
        del data['subreddit']
        del data['subreddit_id']
        return data

class CommentJsonTemplate(ThingTemplate):

    @classmethod
    def get_parent_id(cls, item):
        if False:
            for i in range(10):
                print('nop')
        from r2.models import Comment, Link
        if getattr(item, 'parent_id', None):
            return make_fullname(Comment, item.parent_id)
        else:
            return make_fullname(Link, item.link_id)

    @classmethod
    def get_link_name(cls, item):
        if False:
            print('Hello World!')
        from r2.models import Link
        return make_fullname(Link, item.link_id)

    @classmethod
    def render_child(cls, item):
        if False:
            i = 10
            return i + 15
        child = getattr(item, 'child', None)
        if child:
            return child.render()
        else:
            return ''

    @classmethod
    def get_json(cls, item):
        if False:
            print('Hello World!')
        from r2.models import Link
        data = ThingTemplate.get_json(item)
        data.update(get_mod_attributes(item))
        data.update(get_author_attributes(item))
        data.update(get_distinguished_attributes(item))
        data.update(get_edited_attributes(item))
        data.update(get_report_reason_attributes(item))
        data.update(get_removal_reason_attributes(item))
        data.update({'archived': not item.votable, 'body': item.body, 'body_html': spaceCompress(safemarkdown(item.body)), 'controversiality': 1 if item.is_controversial else 0, 'downs': 0, 'gilded': item.gildings, 'likes': item.likes, 'link_id': cls.get_link_name(item), 'saved': item.saved, 'score': item.score, 'score_hidden': item.score_hidden, 'subreddit': item.subreddit.name, 'subreddit_id': item.subreddit._fullname, 'ups': item.score, 'replies': cls.render_child(item), 'parent_id': cls.get_parent_id(item)})
        if feature.is_enabled('sticky_comments'):
            data['stickied'] = item.link.sticky_comment_id == item._id
        if hasattr(item, 'action_type'):
            data['action_type'] = item.action_type
        if c.profilepage:
            data['quarantine'] = item.subreddit.quarantine
            data['over_18'] = item.link.is_nsfw
            data['link_title'] = item.link.title
            data['link_author'] = item.link_author.name
            if item.link.is_self:
                link_url = item.link.make_permalink(item.subreddit, force_domain=True)
            else:
                link_url = item.link.url
            data['link_url'] = link_url
        return data

    @classmethod
    def get_rendered(cls, item, render_style):
        if False:
            i = 10
            return i + 15
        data = ThingTemplate.get_rendered(item, render_style)
        data.update({'replies': cls.render_child(item), 'contentText': item.body, 'contentHTML': spaceCompress(safemarkdown(item.body)), 'link': cls.get_link_name(item), 'parent': cls.get_parent_id(item)})
        return data

class MoreCommentJsonTemplate(ThingTemplate):

    @classmethod
    def get_kind(cls, item):
        if False:
            for i in range(10):
                print('nop')
        return 'more'

    @classmethod
    def get_json(cls, item):
        if False:
            print('Hello World!')
        data = {'children': [to36(comment_id) for comment_id in item.children], 'count': item.count, 'id': item._id36, 'name': item._fullname, 'parent_id': CommentJsonTemplate.get_parent_id(item)}
        return data

    @classmethod
    def get_rendered(cls, item, render_style):
        if False:
            for i in range(10):
                print('nop')
        data = ThingTemplate.get_rendered(item, render_style)
        data.update({'replies': '', 'contentText': '', 'contentHTML': '', 'link': CommentJsonTemplate.get_link_name(item), 'parent': CommentJsonTemplate.get_parent_id(item)})
        return data

class MessageJsonTemplate(ThingJsonTemplate):
    _data_attrs_ = ThingJsonTemplate.data_attrs(author='author', body='body', body_html='body_html', context='context', created='created', dest='dest', distinguished='distinguished', first_message='first_message', first_message_name='first_message_name', new='new', parent_id='parent_id', replies='child', subject='subject', subreddit='subreddit', was_comment='was_comment')

    def thing_attr(self, thing, attr):
        if False:
            return 10
        from r2.models import Comment, Link, Message
        if attr == 'was_comment':
            return thing.was_comment
        elif attr == 'context':
            return '' if not thing.was_comment else thing.permalink + '?context=3'
        elif attr == 'dest':
            if thing.to_id:
                return thing.to.name
            else:
                return '#' + thing.subreddit.name
        elif attr == 'subreddit':
            if thing.sr_id:
                return thing.subreddit.name
            return None
        elif attr == 'body_html':
            return safemarkdown(thing.body)
        elif attr == 'author' and getattr(thing, 'hide_author', False):
            return None
        elif attr == 'parent_id':
            if thing.was_comment:
                if getattr(thing, 'parent_id', None):
                    return make_fullname(Comment, thing.parent_id)
                else:
                    return make_fullname(Link, thing.link_id)
            elif getattr(thing, 'parent_id', None):
                return make_fullname(Message, thing.parent_id)
        elif attr == 'first_message_name':
            if getattr(thing, 'first_message', None):
                return make_fullname(Message, thing.first_message)
        return ThingJsonTemplate.thing_attr(self, thing, attr)

    def raw_data(self, thing):
        if False:
            while True:
                i = 10
        d = ThingJsonTemplate.raw_data(self, thing)
        if thing.was_comment:
            d['link_title'] = thing.link_title
            d['likes'] = thing.likes
        return d

    def rendered_data(self, wrapped):
        if False:
            for i in range(10):
                print('nop')
        from r2.models import Message
        parent_id = wrapped.parent_id
        if parent_id:
            parent_id = make_fullname(Message, parent_id)
        d = ThingJsonTemplate.rendered_data(self, wrapped)
        d['parent'] = parent_id
        d['contentText'] = self.thing_attr(wrapped, 'body')
        d['contentHTML'] = self.thing_attr(wrapped, 'body_html')
        return d

class RedditJsonTemplate(JsonTemplate):

    def render(self, thing=None, *a, **kw):
        if False:
            print('Hello World!')
        return ObjectTemplate(thing.content().render() if thing else {})

class PanestackJsonTemplate(JsonTemplate):

    def render(self, thing=None, *a, **kw):
        if False:
            i = 10
            return i + 15
        res = [t.render() for t in thing.stack if t] if thing else []
        res = [x for x in res if x]
        if not res:
            return {}
        return ObjectTemplate(res if len(res) > 1 else res[0])

class NullJsonTemplate(JsonTemplate):

    def render(self, thing=None, *a, **kw):
        if False:
            for i in range(10):
                print('nop')
        return ''

    def get_def(self, name):
        if False:
            i = 10
            return i + 15
        return self

class ListingJsonTemplate(ThingJsonTemplate):
    _data_attrs_ = dict(after='after', before='before', children='things', modhash='modhash')

    def thing_attr(self, thing, attr):
        if False:
            i = 10
            return i + 15
        if attr == 'modhash':
            return c.modhash
        elif attr == 'things':
            res = []
            for a in thing.things:
                a.childlisting = False
                r = a.render()
                res.append(r)
            return res
        return ThingJsonTemplate.thing_attr(self, thing, attr)

    def rendered_data(self, thing):
        if False:
            return 10
        return self.thing_attr(thing, 'things')

    def kind(self, wrapped):
        if False:
            print('Hello World!')
        return 'Listing'

class SearchListingJsonTemplate(ListingJsonTemplate):

    def raw_data(self, thing):
        if False:
            i = 10
            return i + 15
        data = ThingJsonTemplate.raw_data(self, thing)

        def format_sr(sr, count):
            if False:
                while True:
                    i = 10
            return {'name': sr.name, 'url': sr.path, 'count': count}
        facets = {}
        if thing.subreddit_facets:
            facets['subreddits'] = [format_sr(sr, count) for (sr, count) in thing.subreddit_facets]
        data['facets'] = facets
        return data

class UserListingJsonTemplate(ListingJsonTemplate):

    def raw_data(self, thing):
        if False:
            i = 10
            return i + 15
        if not thing.nextprev:
            return {'children': self.rendered_data(thing)}
        return ListingJsonTemplate.raw_data(self, thing)

    def kind(self, wrapped):
        if False:
            while True:
                i = 10
        return 'Listing' if wrapped.nextprev else 'UserList'

class UserListJsonTemplate(ThingJsonTemplate):
    _data_attrs_ = dict(children='users')

    def thing_attr(self, thing, attr):
        if False:
            for i in range(10):
                print('nop')
        if attr == 'users':
            res = []
            for a in thing.user_rows:
                r = a.render()
                res.append(r)
            return res
        return ThingJsonTemplate.thing_attr(self, thing, attr)

    def rendered_data(self, thing):
        if False:
            print('Hello World!')
        return self.thing_attr(thing, 'users')

    def kind(self, wrapped):
        if False:
            for i in range(10):
                print('nop')
        return 'UserList'

class UserTableItemJsonTemplate(ThingJsonTemplate):
    _data_attrs_ = dict(id='_fullname', name='name')

    def thing_attr(self, thing, attr):
        if False:
            for i in range(10):
                print('nop')
        return ThingJsonTemplate.thing_attr(self, thing.user, attr)

    def render(self, thing, *a, **kw):
        if False:
            while True:
                i = 10
        return ObjectTemplate(self.data(thing))

class RelTableItemJsonTemplate(UserTableItemJsonTemplate):
    _data_attrs_ = UserTableItemJsonTemplate.data_attrs(date='date')

    def thing_attr(self, thing, attr):
        if False:
            for i in range(10):
                print('nop')
        (rel_attr, splitter, attr) = attr.partition('.')
        if attr == 'note':
            return ThingJsonTemplate.thing_attr(self, thing.rel, attr) or ''
        elif attr:
            return ThingJsonTemplate.thing_attr(self, thing.rel, attr)
        elif rel_attr == 'date':
            date = self.thing_attr(thing, 'rel._date')
            date = time.mktime(date.astimezone(pytz.UTC).timetuple())
            return date - time.timezone
        else:
            return UserTableItemJsonTemplate.thing_attr(self, thing, rel_attr)

class FriendTableItemJsonTemplate(RelTableItemJsonTemplate):

    def inject_data(self, thing, d):
        if False:
            print('Hello World!')
        if c.user.gold and thing.type == 'friend':
            d['note'] = self.thing_attr(thing, 'rel.note')
        return d

    def rendered_data(self, thing):
        if False:
            print('Hello World!')
        d = RelTableItemJsonTemplate.rendered_data(self, thing)
        return self.inject_data(thing, d)

    def raw_data(self, thing):
        if False:
            i = 10
            return i + 15
        d = RelTableItemJsonTemplate.raw_data(self, thing)
        return self.inject_data(thing, d)

class BannedTableItemJsonTemplate(RelTableItemJsonTemplate):
    _data_attrs_ = RelTableItemJsonTemplate.data_attrs(note='rel.note')

class MutedTableItemJsonTemplate(RelTableItemJsonTemplate):
    pass

class InvitedModTableItemJsonTemplate(RelTableItemJsonTemplate):
    _data_attrs_ = RelTableItemJsonTemplate.data_attrs(mod_permissions='permissions')

    def thing_attr(self, thing, attr):
        if False:
            i = 10
            return i + 15
        if attr == 'permissions':
            permissions = thing.permissions.items()
            return [perm for (perm, has) in permissions if has]
        else:
            return RelTableItemJsonTemplate.thing_attr(self, thing, attr)

class OrganicListingJsonTemplate(ListingJsonTemplate):

    def kind(self, wrapped):
        if False:
            return 10
        return 'OrganicListing'

class TrafficJsonTemplate(JsonTemplate):

    def render(self, thing, *a, **kw):
        if False:
            print('Hello World!')
        res = {}
        for interval in ('hour', 'day', 'month'):
            interval_data = thing.get_data_for_interval(interval, [])
            res[interval] = [(calendar.timegm(date.timetuple()),) + data for (date, data) in interval_data]
        return ObjectTemplate(res)

class WikiJsonTemplate(JsonTemplate):

    def render(self, thing, *a, **kw):
        if False:
            print('Hello World!')
        try:
            content = thing.content()
        except AttributeError:
            content = thing.listing
        return ObjectTemplate(content.render() if thing else {})

class WikiPageListingJsonTemplate(ThingJsonTemplate):

    def kind(self, thing):
        if False:
            i = 10
            return i + 15
        return 'wikipagelisting'

    def data(self, thing):
        if False:
            for i in range(10):
                print('nop')
        pages = [p.name for p in thing.linear_pages]
        return pages

class WikiViewJsonTemplate(ThingJsonTemplate):

    def kind(self, thing):
        if False:
            i = 10
            return i + 15
        return 'wikipage'

    def data(self, thing):
        if False:
            for i in range(10):
                print('nop')
        edit_date = time.mktime(thing.edit_date.timetuple()) if thing.edit_date else None
        edit_by = None
        if thing.edit_by and (not thing.edit_by._deleted):
            edit_by = Wrapped(thing.edit_by).render()
        return dict(content_md=thing.page_content_md, content_html=thing.page_content, revision_by=edit_by, revision_date=edit_date, may_revise=thing.may_revise)

class WikiSettingsJsonTemplate(ThingJsonTemplate):

    def kind(self, thing):
        if False:
            return 10
        return 'wikipagesettings'

    def data(self, thing):
        if False:
            while True:
                i = 10
        editors = [Wrapped(e).render() for e in thing.mayedit]
        return dict(permlevel=thing.permlevel, listed=thing.listed, editors=editors)

class WikiRevisionJsonTemplate(ThingJsonTemplate):

    def render(self, thing, *a, **kw):
        if False:
            while True:
                i = 10
        timestamp = time.mktime(thing.date.timetuple()) if thing.date else None
        author = thing.get_author()
        if author and (not author._deleted):
            author = Wrapped(author).render()
        else:
            author = None
        return ObjectTemplate(dict(author=author, id=str(thing._id), timestamp=timestamp, reason=thing._get('reason'), page=thing.page))

class FlairListJsonTemplate(JsonTemplate):

    def render(self, thing, *a, **kw):
        if False:
            print('Hello World!')

        def row_to_json(row):
            if False:
                return 10
            if hasattr(row, 'user'):
                return dict(user=row.user.name, flair_text=row.flair_text, flair_css_class=row.flair_css_class)
            else:
                return dict(after=row.after, reverse=row.previous)
        json_rows = [row_to_json(row) for row in thing.flair]
        result = dict(users=[row for row in json_rows if 'user' in row])
        for row in json_rows:
            if 'after' in row:
                if row['reverse']:
                    result['prev'] = row['after']
                else:
                    result['next'] = row['after']
        return ObjectTemplate(result)

class FlairCsvJsonTemplate(JsonTemplate):

    def render(self, thing, *a, **kw):
        if False:
            print('Hello World!')
        return ObjectTemplate([l.__dict__ for l in thing.results_by_line])

class FlairSelectorJsonTemplate(JsonTemplate):

    def _template_dict(self, flair):
        if False:
            while True:
                i = 10
        return {'flair_template_id': flair.flair_template_id, 'flair_position': flair.flair_position, 'flair_text': flair.flair_text, 'flair_css_class': flair.flair_css_class, 'flair_text_editable': flair.flair_text_editable}

    def render(self, thing, *a, **kw):
        if False:
            while True:
                i = 10
        'Render a list of flair choices into JSON\n\n        Sample output:\n        {\n            "choices": [\n                {\n                    "flair_css_class": "flair-444",\n                    "flair_position": "right",\n                    "flair_template_id": "5668d204-9388-11e3-8109-080027a38559",\n                    "flair_text": "444",\n                    "flair_text_editable": true\n                },\n                {\n                    "flair_css_class": "flair-nouser",\n                    "flair_position": "right",\n                    "flair_template_id": "58e34d7a-9388-11e3-ab01-080027a38559",\n                    "flair_text": "nouser",\n                    "flair_text_editable": true\n                },\n                {\n                    "flair_css_class": "flair-bar",\n                    "flair_position": "right",\n                    "flair_template_id": "fb01cc04-9391-11e3-b1d6-080027a38559",\n                    "flair_text": "foooooo",\n                    "flair_text_editable": true\n                }\n            ],\n            "current": {\n                "flair_css_class": "444",\n                "flair_position": "right",\n                "flair_template_id": "5668d204-9388-11e3-8109-080027a38559",\n                "flair_text": "444"\n            }\n        }\n\n        '
        choices = [self._template_dict(choice) for choice in thing.choices]
        current_flair = {'flair_text': thing.text, 'flair_css_class': thing.css_class, 'flair_position': thing.position, 'flair_template_id': thing.matching_template}
        return ObjectTemplate({'current': current_flair, 'choices': choices})

class StylesheetTemplate(ThingJsonTemplate):
    _data_attrs_ = dict(images='_images', stylesheet='stylesheet_contents', subreddit_id='_fullname')

    def kind(self, wrapped):
        if False:
            for i in range(10):
                print('nop')
        return 'stylesheet'

    def images(self):
        if False:
            print('Hello World!')
        sr_images = ImagesByWikiPage.get_images(c.site, 'config/stylesheet')
        images = []
        for (name, url) in sr_images.iteritems():
            images.append({'name': name, 'link': 'url(%%%%%s%%%%)' % name, 'url': url})
        return images

    def thing_attr(self, thing, attr):
        if False:
            for i in range(10):
                print('nop')
        if attr == '_images':
            return self.images()
        elif attr == '_fullname':
            return c.site._fullname
        return ThingJsonTemplate.thing_attr(self, thing, attr)

class SubredditSettingsTemplate(ThingJsonTemplate):
    _data_attrs_ = dict(allow_images='site.allow_images', collapse_deleted_comments='site.collapse_deleted_comments', comment_score_hide_mins='site.comment_score_hide_mins', content_options='site.link_type', default_set='site.allow_top', description='site.description', domain='site.domain', exclude_banned_modqueue='site.exclude_banned_modqueue', header_hover_text='site.header_title', language='site.lang', over_18='site.over_18', public_description='site.public_description', public_traffic='site.public_traffic', hide_ads='site.hide_ads', show_media='site.show_media', show_media_preview='site.show_media_preview', submit_link_label='site.submit_link_label', submit_text_label='site.submit_text_label', submit_text='site.submit_text', subreddit_id='site._fullname', subreddit_type='site.type', suggested_comment_sort='site.suggested_comment_sort', title='site.title', wiki_edit_age='site.wiki_edit_age', wiki_edit_karma='site.wiki_edit_karma', wikimode='site.wikimode', spam_links='site.spam_links', spam_selfposts='site.spam_selfposts', spam_comments='site.spam_comments')

    def kind(self, wrapped):
        if False:
            return 10
        return 'subreddit_settings'

    def thing_attr(self, thing, attr):
        if False:
            for i in range(10):
                print('nop')
        if attr.startswith('site.') and thing.site:
            return getattr(thing.site, attr[5:])
        if attr == 'related_subreddits' and thing.site:
            return '\n'.join(thing.site.related_subreddits)
        return ThingJsonTemplate.thing_attr(self, thing, attr)

    def raw_data(self, thing):
        if False:
            return 10
        data = ThingJsonTemplate.raw_data(self, thing)
        if feature.is_enabled('mobile_settings'):
            data['key_color'] = self.thing_attr(thing, 'key_color')
        if feature.is_enabled('related_subreddits'):
            data['related_subreddits'] = self.thing_attr(thing, 'related_subreddits')
        return data

class UploadedImageJsonTemplate(JsonTemplate):

    def render(self, thing, *a, **kw):
        if False:
            while True:
                i = 10
        return ObjectTemplate({'errors': list((k for (k, v) in thing.errors if v)), 'img_src': thing.img_src})

class ModActionTemplate(ThingJsonTemplate):
    _data_attrs_ = dict(action='action', created_utc='date', description='description', details='details', id='_fullname', mod='moderator', mod_id36='mod_id36', sr_id36='sr_id36', subreddit='subreddit', target_author='target_author', target_fullname='target_fullname', target_permalink='target_permalink', target_title='target_title', target_body='target_body')

    def thing_attr(self, thing, attr):
        if False:
            return 10
        if attr == 'date':
            return time.mktime(thing.date.astimezone(pytz.UTC).timetuple()) - time.timezone
        elif attr == 'target_author':
            if thing.target_author and thing.target_author._deleted:
                return '[deleted]'
            elif thing.target_author:
                return thing.target_author.name
            return ''
        elif attr == 'target_permalink':
            try:
                return thing.target.make_permalink_slow()
            except AttributeError:
                return None
        elif attr == 'moderator':
            return thing.moderator.name
        elif attr == 'subreddit':
            return thing.subreddit.name
        elif attr == 'target_title' and isinstance(thing.target, Link):
            return thing.target.title
        elif attr == 'target_body' and isinstance(thing.target, Comment):
            return thing.target.body
        elif attr == 'target_body' and isinstance(thing.target, Link) and getattr(thing.target, 'selftext', None):
            return thing.target.selftext
        return ThingJsonTemplate.thing_attr(self, thing, attr)

    def kind(self, wrapped):
        if False:
            while True:
                i = 10
        return 'modaction'

class PolicyViewJsonTemplate(ThingJsonTemplate):
    _data_attrs_ = dict(body_html='body_html', display_rev='display_rev', revs='revs', toc_html='toc_html')

    def kind(self, wrapped):
        if False:
            return 10
        return 'Policy'

class KarmaListJsonTemplate(ThingJsonTemplate):

    def data(self, karmas):
        if False:
            return 10
        from r2.lib.template_helpers import display_comment_karma, display_link_karma
        karmas = [{'sr': sr, 'link_karma': display_link_karma(link_karma), 'comment_karma': display_comment_karma(comment_karma)} for (sr, (link_karma, comment_karma)) in karmas.iteritems()]
        return karmas

    def kind(self, wrapped):
        if False:
            print('Hello World!')
        return 'KarmaList'

def get_usertrophies(user):
    if False:
        for i in range(10):
            print('nop')
    trophies = Trophy.by_account(user)

    def visible_trophy(trophy):
        if False:
            print('Hello World!')
        return trophy._thing2.awardtype != 'invisible'
    trophies = filter(visible_trophy, trophies)
    resp = TrophyListJsonTemplate().render(trophies)
    return resp.finalize()

class TrophyJsonTemplate(ThingJsonTemplate):
    _data_attrs_ = dict(award_id='award._id36', description='description', name='award.title', id='_id36', icon_40='icon_40', icon_70='icon_70', url='trophy_url')

    def thing_attr(self, thing, attr):
        if False:
            print('Hello World!')
        if attr == 'icon_40':
            return 'https:' + thing._thing2.imgurl % 40
        elif attr == 'icon_70':
            return 'https:' + thing._thing2.imgurl % 70
        (rel_attr, splitter, attr) = attr.partition('.')
        if attr:
            return ThingJsonTemplate.thing_attr(self, thing._thing2, attr)
        else:
            return ThingJsonTemplate.thing_attr(self, thing, rel_attr)

    def kind(self, thing):
        if False:
            for i in range(10):
                print('nop')
        return ThingJsonTemplate.kind(self, thing._thing2)

class TrophyListJsonTemplate(ThingJsonTemplate):

    def data(self, trophies):
        if False:
            return 10
        trophies = [Wrapped(t).render() for t in trophies]
        return dict(trophies=trophies)

    def kind(self, wrapped):
        if False:
            for i in range(10):
                print('nop')
        return 'TrophyList'

class RulesJsonTemplate(JsonTemplate):

    def render(self, thing, *a, **kw):
        if False:
            while True:
                i = 10
        rules = {}
        rules['site_rules'] = thing.site_rules
        rules['rules'] = thing.rules
        for rule in rules['rules']:
            if rule.get('description'):
                rule['description_html'] = safemarkdown(rule['description'])
            if not rule.get('kind'):
                rule['kind'] = 'all'
        return ObjectTemplate(rules)