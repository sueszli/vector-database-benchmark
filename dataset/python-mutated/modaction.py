from datetime import timedelta
import itertools
from uuid import UUID
from pycassa.system_manager import TIME_UUID_TYPE
from pylons import request
from pylons import tmpl_context as c
from pylons import app_globals as g
from pylons.i18n import _
from r2.lib.db import tdb_cassandra
from r2.lib.utils import tup

class ModAction(tdb_cassandra.UuidThing):
    """
    Columns:
    sr_id - Subreddit id36
    mod_id - Account id36 of moderator
    action - specific name of action, must be in ModAction.actions
    target_fullname - optional fullname of the target of the action
    details - subcategory available for some actions, must show up in 
    description - optional user
    """
    _read_consistency_level = tdb_cassandra.CL.ONE
    _use_db = True
    _connection_pool = 'main'
    _ttl = timedelta(days=120)
    _str_props = ('sr_id36', 'mod_id36', 'target_fullname', 'action', 'details', 'description')
    _defaults = {}
    actions = ('banuser', 'unbanuser', 'removelink', 'approvelink', 'removecomment', 'approvecomment', 'addmoderator', 'invitemoderator', 'uninvitemoderator', 'acceptmoderatorinvite', 'removemoderator', 'addcontributor', 'removecontributor', 'editsettings', 'editflair', 'distinguish', 'marknsfw', 'wikibanned', 'wikicontributor', 'wikiunbanned', 'wikipagelisted', 'removewikicontributor', 'wikirevise', 'wikipermlevel', 'ignorereports', 'unignorereports', 'setpermissions', 'setsuggestedsort', 'sticky', 'unsticky', 'setcontestmode', 'unsetcontestmode', 'lock', 'unlock', 'muteuser', 'unmuteuser', 'createrule', 'editrule', 'deleterule')
    _menu = {'banuser': _('ban user'), 'unbanuser': _('unban user'), 'removelink': _('remove post'), 'approvelink': _('approve post'), 'removecomment': _('remove comment'), 'approvecomment': _('approve comment'), 'addmoderator': _('add moderator'), 'removemoderator': _('remove moderator'), 'invitemoderator': _('invite moderator'), 'uninvitemoderator': _('uninvite moderator'), 'acceptmoderatorinvite': _('accept moderator invite'), 'addcontributor': _('add contributor'), 'removecontributor': _('remove contributor'), 'editsettings': _('edit settings'), 'editflair': _('edit flair'), 'distinguish': _('distinguish'), 'marknsfw': _('mark nsfw'), 'wikibanned': _('ban from wiki'), 'wikiunbanned': _('unban from wiki'), 'wikicontributor': _('add wiki contributor'), 'wikipagelisted': _('delist/relist wiki pages'), 'removewikicontributor': _('remove wiki contributor'), 'wikirevise': _('wiki revise page'), 'wikipermlevel': _('wiki page permissions'), 'ignorereports': _('ignore reports'), 'unignorereports': _('unignore reports'), 'setpermissions': _('permissions'), 'setsuggestedsort': _('set suggested sort'), 'sticky': _('sticky post'), 'unsticky': _('unsticky post'), 'setcontestmode': _('set contest mode'), 'unsetcontestmode': _('unset contest mode'), 'lock': _('lock post'), 'unlock': _('unlock post'), 'muteuser': _('mute user'), 'unmuteuser': _('unmute user'), 'createrule': _('create rule'), 'editrule': _('edit rule'), 'deleterule': _('delete rule')}
    _text = {'banuser': _('banned'), 'wikibanned': _('wiki banned'), 'wikiunbanned': _('unbanned from wiki'), 'wikicontributor': _('added wiki contributor'), 'removewikicontributor': _('removed wiki contributor'), 'unbanuser': _('unbanned'), 'removelink': _('removed'), 'approvelink': _('approved'), 'removecomment': _('removed'), 'approvecomment': _('approved'), 'addmoderator': _('added moderator'), 'removemoderator': _('removed moderator'), 'invitemoderator': _('invited moderator'), 'uninvitemoderator': _('uninvited moderator'), 'acceptmoderatorinvite': _('accepted moderator invitation'), 'addcontributor': _('added approved contributor'), 'removecontributor': _('removed approved contributor'), 'editsettings': _('edited settings'), 'editflair': _('edited flair'), 'wikirevise': _('edited wiki page'), 'wikipermlevel': _('changed wiki page permission level'), 'wikipagelisted': _('changed wiki page listing preference'), 'distinguish': _('distinguished'), 'marknsfw': _('marked nsfw'), 'ignorereports': _('ignored reports'), 'unignorereports': _('unignored reports'), 'setpermissions': _('changed permissions on'), 'setsuggestedsort': _('set suggested sort'), 'sticky': _('stickied'), 'unsticky': _('unstickied'), 'setcontestmode': _('set contest mode on'), 'unsetcontestmode': _('unset contest mode on'), 'lock': _('locked'), 'unlock': _('unlocked'), 'muteuser': _('muted'), 'unmuteuser': _('unmuted'), 'createrule': _('created rule'), 'editrule': _('edited rule'), 'deleterule': _('deleted rule')}
    _details_text = {'unspam': _('unspam'), 'confirm_ham': _('approved'), 'confirm_spam': _('confirmed spam'), 'remove': _('removed not spam'), 'spam': _('removed spam'), 'remove_self': _('removed self'), 'title': _('title'), 'public_description': _('description'), 'description': _('sidebar'), 'lang': _('language'), 'type': _('type'), 'link_type': _('link type'), 'submit_link_label': _('submit link button label'), 'submit_text_label': _('submit text post button label'), 'comment_score_hide_mins': _('comment score hide period'), 'over_18': _('toggle viewers must be over 18'), 'allow_top': _('toggle allow in default/trending lists'), 'show_media': _('toggle show thumbnail images of content'), 'public_traffic': _('toggle public traffic stats page'), 'collapse_deleted_comments': _('toggle collapse deleted/removed comments'), 'exclude_banned_modqueue': _("toggle exclude banned users' posts from modqueue"), 'domain': _('domain'), 'show_cname_sidebar': _('toggle show sidebar from cname'), 'css_on_cname': _('toggle custom CSS from cname'), 'header_title': _('header title'), 'stylesheet': _('stylesheet'), 'del_header': _('delete header image'), 'del_image': _('delete image'), 'del_icon': _('delete icon image'), 'del_banner': _('delete banner image'), 'upload_image_header': _('upload header image'), 'upload_image_icon': _('upload icon image'), 'upload_image_banner': _('upload banner image'), 'upload_image': _('upload image'), 'flair_edit': _('add/edit flair'), 'flair_delete': _('delete flair'), 'flair_csv': _('edit by csv'), 'flair_enabled': _('toggle flair enabled'), 'flair_position': _('toggle user flair position'), 'link_flair_position': _('toggle link flair position'), 'flair_self_enabled': _('toggle user assigned flair enabled'), 'link_flair_self_enabled': _('toggle submitter assigned link flair enabled'), 'flair_template': _('add/edit flair templates'), 'flair_delete_template': _('delete flair template'), 'flair_clear_template': _('clear flair templates'), 'remove': _('remove'), 'ignore_reports': _('ignore reports'), 'permission_moderator': _('set permissions on moderator'), 'permission_moderator_invite': _('set permissions on moderator invitation')}

    @classmethod
    def create(cls, sr, mod, action, details=None, target=None, description=None):
        if False:
            return 10
        from r2.models import DefaultSR
        if not action in cls.actions:
            raise ValueError('Invalid ModAction: %s' % action)
        sr = sr._base if isinstance(sr, DefaultSR) else sr
        kw = dict(sr_id36=sr._id36, mod_id36=mod._id36, action=action)
        if target:
            kw['target_fullname'] = target._fullname
        if details:
            kw['details'] = details
        if description:
            kw['description'] = description
        ma = cls(**kw)
        ma._commit()
        g.events.mod_event(modaction=ma, subreddit=sr, mod=mod, target=target, request=request if c.user_is_loggedin else None, context=c if c.user_is_loggedin else None)
        return ma

    def _on_create(self):
        if False:
            return 10
        '\n        Update all Views.\n        '
        views = (ModActionBySR, ModActionBySRMod, ModActionBySRAction, ModActionBySRActionMod)
        for v in views:
            v.add_object(self)

    @classmethod
    def get_actions(cls, srs, mod=None, action=None, after=None, reverse=False, count=1000):
        if False:
            print('Hello World!')
        '\n        Get a ColumnQuery that yields ModAction objects according to\n        specified criteria.\n        '
        if after and isinstance(after, basestring):
            after = cls._byID(UUID(after))
        elif after and isinstance(after, UUID):
            after = cls._byID(after)
        if not isinstance(after, cls):
            after = None
        srs = tup(srs)
        if not mod and (not action):
            rowkeys = [sr._id36 for sr in srs]
            q = ModActionBySR.query(rowkeys, after=after, reverse=reverse, count=count)
        elif mod:
            mods = tup(mod)
            key = '%s_%s' if not action else '%%s_%%s_%s' % action
            rowkeys = itertools.product([sr._id36 for sr in srs], [mod._id36 for mod in mods])
            rowkeys = [key % (sr, mod) for (sr, mod) in rowkeys]
            view = ModActionBySRActionMod if action else ModActionBySRMod
            q = view.query(rowkeys, after=after, reverse=reverse, count=count)
        else:
            rowkeys = ['%s_%s' % (sr._id36, action) for sr in srs]
            q = ModActionBySRAction.query(rowkeys, after=after, reverse=reverse, count=count)
        return q

    @property
    def details_text(self):
        if False:
            for i in range(10):
                print('nop')
        text = ''
        if getattr(self, 'details', None):
            text += self._details_text.get(self.details, self.details)
        if getattr(self, 'description', None):
            if text:
                text += ': '
            text += self.description
        return text

    @classmethod
    def add_props(cls, user, wrapped):
        if False:
            print('Hello World!')
        from r2.lib.db.thing import Thing
        from r2.lib.menus import QueryButton
        from r2.lib.pages import WrappedUser
        from r2.models import Account, Link, ModSR, MultiReddit, Subreddit
        target_names = {item.target_fullname for item in wrapped if hasattr(item, 'target_fullname')}
        targets = Thing._by_fullname(target_names, data=True)
        moderators = Account._byID36({item.mod_id36 for item in wrapped}, data=True)
        target_author_names = {target.author_id for target in targets.values() if hasattr(target, 'author_id')}
        target_authors = Account._byID(target_author_names, data=True)
        parent_link_names = {target.link_id for target in targets.values() if hasattr(target, 'link_id')}
        parent_links = Link._byID(parent_link_names, data=True)
        srs = Subreddit._byID36({item.sr_id36 for item in wrapped}, data=True)
        for item in wrapped:
            item.moderator = moderators[item.mod_id36]
            item.subreddit = srs[item.sr_id36]
            item.text = cls._text.get(item.action, '')
            item.target = None
            item.target_author = None
            if hasattr(item, 'target_fullname') and item.target_fullname:
                item.target = targets[item.target_fullname]
                if hasattr(item.target, 'author_id'):
                    author_name = item.target.author_id
                    item.target_author = target_authors[author_name]
                if hasattr(item.target, 'link_id'):
                    parent_link_name = item.target.link_id
                    item.parent_link = parent_links[parent_link_name]
                if isinstance(item.target, Account):
                    item.target_author = item.target
        if c.render_style == 'html':
            request_path = request.path
            user_targets = filter(lambda target: isinstance(target, Account), targets.values())
            wrapped_user_targets = {user._fullname: WrappedUser(user) for user in user_targets}
            for item in wrapped:
                if isinstance(item.target, Account):
                    user_name = item.target._fullname
                    item.wrapped_user_target = wrapped_user_targets[user_name]
                css_class = 'modactions %s' % item.action
                action_button = QueryButton('', item.action, query_param='type', css_class=css_class)
                action_button.build(base_path=request_path)
                item.action_button = action_button
                mod_button = QueryButton(item.moderator.name, item.moderator.name, query_param='mod')
                mod_button.build(base_path=request_path)
                item.mod_button = mod_button
                if isinstance(c.site, ModSR) or isinstance(c.site, MultiReddit):
                    rgb = item.subreddit.get_rgb()
                    item.bgcolor = 'rgb(%s,%s,%s)' % rgb
                    item.is_multi = True
                else:
                    item.bgcolor = 'rgb(255,255,255)'
                    item.is_multi = False

class ModActionBySR(tdb_cassandra.View):
    _use_db = True
    _connection_pool = 'main'
    _compare_with = TIME_UUID_TYPE
    _view_of = ModAction
    _ttl = timedelta(days=90)
    _read_consistency_level = tdb_cassandra.CL.ONE

    @classmethod
    def _rowkey(cls, ma):
        if False:
            i = 10
            return i + 15
        return ma.sr_id36

class ModActionBySRMod(tdb_cassandra.View):
    _use_db = True
    _connection_pool = 'main'
    _compare_with = TIME_UUID_TYPE
    _view_of = ModAction
    _ttl = timedelta(days=90)
    _read_consistency_level = tdb_cassandra.CL.ONE

    @classmethod
    def _rowkey(cls, ma):
        if False:
            while True:
                i = 10
        return '%s_%s' % (ma.sr_id36, ma.mod_id36)

class ModActionBySRActionMod(tdb_cassandra.View):
    _use_db = True
    _connection_pool = 'main'
    _compare_with = TIME_UUID_TYPE
    _view_of = ModAction
    _ttl = timedelta(days=90)
    _read_consistency_level = tdb_cassandra.CL.ONE

    @classmethod
    def _rowkey(cls, ma):
        if False:
            print('Hello World!')
        return '%s_%s_%s' % (ma.sr_id36, ma.mod_id36, ma.action)

class ModActionBySRAction(tdb_cassandra.View):
    _use_db = True
    _connection_pool = 'main'
    _compare_with = TIME_UUID_TYPE
    _view_of = ModAction
    _ttl = timedelta(days=90)
    _read_consistency_level = tdb_cassandra.CL.ONE

    @classmethod
    def _rowkey(cls, ma):
        if False:
            while True:
                i = 10
        return '%s_%s' % (ma.sr_id36, ma.action)