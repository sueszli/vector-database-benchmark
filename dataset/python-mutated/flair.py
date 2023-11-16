import uuid
from pylons import app_globals as g
from r2.lib.db import tdb_cassandra
from r2.lib.db.thing import Relation
from r2.lib.db.userrel import UserRel
from r2.lib.utils import to36
from r2.models import Account, Subreddit
USER_FLAIR = 'USER_FLAIR'
LINK_FLAIR = 'LINK_FLAIR'

class Flair(Relation(Subreddit, Account)):
    _cache = g.thingcache

    @classmethod
    def _cache_prefix(cls):
        if False:
            i = 10
            return i + 15
        return 'flair:'
Subreddit.__bases__ += (UserRel(name='flair', relation=Flair, disable_ids_fn=True, disable_reverse_ids_fn=True),)

class FlairTemplate(tdb_cassandra.Thing):
    """A template for some flair."""
    _defaults = dict(text='', css_class='', text_editable=False)
    _bool_props = ('text_editable',)
    _use_db = True
    _connection_pool = 'main'

    @classmethod
    def _new(cls, text='', css_class='', text_editable=False):
        if False:
            i = 10
            return i + 15
        if text is None:
            text = ''
        if css_class is None:
            css_class = ''
        ft = cls(text=text, css_class=css_class, text_editable=text_editable)
        ft._commit()
        return ft

    def _commit(self, *a, **kw):
        if False:
            i = 10
            return i + 15
        if not self._id:
            self._id = str(uuid.uuid1())
        return tdb_cassandra.Thing._commit(self, *a, **kw)

    def covers(self, other_template):
        if False:
            for i in range(10):
                print('nop')
        "Returns true if other_template is a subset of this one.\n\n        The value for other_template may be another FlairTemplate, or a tuple\n        of (text, css_class). The latter case is treated like a FlairTemplate\n        that doesn't permit editable text.\n\n        For example, if self permits editable text, then this method will return\n        True as long as just the css_classes match. On the other hand, if self\n        doesn't permit editable text but other_template does, this method will\n        return False.\n        "
        if isinstance(other_template, FlairTemplate):
            text_editable = other_template.text_editable
            (text, css_class) = (other_template.text, other_template.css_class)
        else:
            text_editable = False
            (text, css_class) = other_template
        if self.css_class != css_class:
            return False
        return self.text_editable or (not text_editable and self.text == text)

class FlairTemplateBySubredditIndex(tdb_cassandra.Thing):
    """Lists of FlairTemplate IDs for a subreddit.

    The FlairTemplate references are stored as an arbitrary number of attrs.
    The lexicographical ordering of these attr names gives the ordering for
    flair templates within the subreddit.
    """
    MAX_FLAIR_TEMPLATES = 350
    _int_props = ('sr_id',)
    _use_db = True
    _connection_pool = 'main'
    _key_prefixes = {USER_FLAIR: 'ft_', LINK_FLAIR: 'link_ft_'}

    @classmethod
    def _new(cls, sr_id, flair_type=USER_FLAIR):
        if False:
            return 10
        idx = cls(_id=to36(sr_id), sr_id=sr_id)
        idx._commit()
        return idx

    @classmethod
    def by_sr(cls, sr_id, create=False):
        if False:
            i = 10
            return i + 15
        try:
            return cls._byID(to36(sr_id))
        except tdb_cassandra.NotFound:
            if create:
                return cls._new(sr_id)
            raise

    @classmethod
    def create_template(cls, sr_id, text='', css_class='', text_editable=False, flair_type=USER_FLAIR):
        if False:
            i = 10
            return i + 15
        idx = cls.by_sr(sr_id, create=True)
        if len(idx._index_keys(flair_type)) >= cls.MAX_FLAIR_TEMPLATES:
            raise OverflowError
        ft = FlairTemplate._new(text=text, css_class=css_class, text_editable=text_editable)
        idx.insert(ft._id, flair_type=flair_type)
        return ft

    @classmethod
    def get_template_ids(cls, sr_id, flair_type=USER_FLAIR):
        if False:
            i = 10
            return i + 15
        try:
            return list(cls.by_sr(sr_id).iter_template_ids(flair_type))
        except tdb_cassandra.NotFound:
            return []

    @classmethod
    def get_template(cls, sr_id, ft_id, flair_type=None):
        if False:
            return 10
        if flair_type:
            flair_types = [flair_type]
        else:
            flair_types = [USER_FLAIR, LINK_FLAIR]
        for flair_type in flair_types:
            if ft_id in cls.get_template_ids(sr_id, flair_type=flair_type):
                return FlairTemplate._byID(ft_id)
        return None

    @classmethod
    def clear(cls, sr_id, flair_type=USER_FLAIR):
        if False:
            i = 10
            return i + 15
        try:
            idx = cls.by_sr(sr_id)
        except tdb_cassandra.NotFound:
            return
        for k in idx._index_keys(flair_type):
            del idx[k]
        idx._commit()

    def _index_keys(self, flair_type):
        if False:
            print('Hello World!')
        keys = set(self._dirties.iterkeys())
        keys |= frozenset(self._orig.iterkeys())
        keys -= self._deletes
        key_prefix = self._key_prefixes[flair_type]
        return [k for k in keys if k.startswith(key_prefix)]

    @classmethod
    def _make_index_key(cls, position, flair_type):
        if False:
            for i in range(10):
                print('nop')
        return '%s%08d' % (cls._key_prefixes[flair_type], position)

    def iter_template_ids(self, flair_type):
        if False:
            for i in range(10):
                print('nop')
        return (getattr(self, key) for key in sorted(self._index_keys(flair_type)))

    def insert(self, ft_id, position=None, flair_type=USER_FLAIR):
        if False:
            print('Hello World!')
        'Insert template reference into index at position.\n\n        A position value of None means to simply append.\n        '
        ft_ids = list(self.iter_template_ids(flair_type))
        if position is None:
            position = len(ft_ids)
        if position < 0 or position > len(ft_ids):
            raise IndexError(position)
        ft_ids.insert(position, ft_id)
        for k in self._index_keys(flair_type):
            del self[k]
        for (i, ft_id) in enumerate(ft_ids):
            setattr(self, self._make_index_key(i, flair_type), ft_id)
        self._commit()

    def delete_by_id(self, ft_id, flair_type=None):
        if False:
            return 10
        if flair_type:
            flair_types = [flair_type]
        else:
            flair_types = [USER_FLAIR, LINK_FLAIR]
        for flair_type in flair_types:
            if self._delete_by_id(ft_id, flair_type):
                return True
        g.log.debug("couldn't find %s to delete", ft_id)
        return False

    def _delete_by_id(self, ft_id, flair_type):
        if False:
            return 10
        for key in self._index_keys(flair_type):
            ft = getattr(self, key)
            if ft == ft_id:
                g.log.debug('deleting ft %s (%s)', ft, key)
                del self[key]
                self._commit()
                return True
        return False