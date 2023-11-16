from datetime import datetime
from struct import unpack
from pony import orm
from pony.orm import db_session
from tribler.core import notifications
from tribler.core.components.metadata_store.category_filter.category import Category, default_category_filter
from tribler.core.components.metadata_store.category_filter.family_filter import default_xxx_filter
from tribler.core.components.metadata_store.db.orm_bindings.channel_node import COMMITTED
from tribler.core.components.metadata_store.db.serialization import EPOCH, REGULAR_TORRENT, TorrentMetadataPayload
from tribler.core.utilities.notifier import Notifier
from tribler.core.utilities.tracker_utils import get_uniformed_tracker_url
from tribler.core.utilities.unicode import ensure_unicode, hexlify
NULL_KEY_SUBST = b'\x00'

def infohash_to_id(infohash):
    if False:
        i = 10
        return i + 15
    return abs(unpack('>q', infohash[:8])[0])

def tdef_to_metadata_dict(tdef, category_filter: Category=None):
    if False:
        while True:
            i = 10
    '\n    Helper function to create a TorrentMetadata-compatible dict from TorrentDef\n    '
    category_filter = category_filter or default_category_filter
    try:
        tags = category_filter.calculateCategory(tdef.metainfo, tdef.get_name_as_unicode())
    except UnicodeDecodeError:
        tags = 'Unknown'
    try:
        torrent_date = datetime.fromtimestamp(tdef.get_creation_date())
    except (ValueError, TypeError):
        torrent_date = EPOCH
    tracker = tdef.get_tracker()
    if not isinstance(tracker, bytes):
        tracker = b''
    tracker_url = ensure_unicode(tracker, 'utf-8')
    tracker_info = get_uniformed_tracker_url(tracker_url) or ''
    return {'infohash': tdef.get_infohash(), 'title': tdef.get_name_as_unicode()[:300], 'tags': tags[:200], 'size': tdef.get_length(), 'torrent_date': torrent_date if torrent_date >= EPOCH else EPOCH, 'tracker_info': tracker_info}

def define_binding(db, notifier: Notifier, tag_processor_version: int):
    if False:
        i = 10
        return i + 15

    class TorrentMetadata(db.MetadataNode):
        """
        This ORM binding class is intended to store Torrent objects, i.e. infohashes along with some related metadata.
        """
        _discriminator_ = REGULAR_TORRENT
        infohash = orm.Required(bytes, index=True)
        size = orm.Optional(int, size=64, default=0)
        torrent_date = orm.Optional(datetime, default=datetime.utcnow, index=True)
        tracker_info = orm.Optional(str, default='')
        xxx = orm.Optional(float, default=0)
        health = orm.Optional('TorrentState', reverse='metadata')
        tag_processor_version = orm.Required(int, default=0)
        _payload_class = TorrentMetadataPayload
        payload_arguments = _payload_class.__init__.__code__.co_varnames[:_payload_class.__init__.__code__.co_argcount][1:]
        nonpersonal_attributes = db.MetadataNode.nonpersonal_attributes + ('infohash', 'size', 'torrent_date', 'tracker_info')

        def __init__(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            if 'health' not in kwargs and 'infohash' in kwargs:
                infohash = kwargs['infohash']
                health = db.TorrentState.get_for_update(infohash=infohash) or db.TorrentState(infohash=infohash)
                kwargs['health'] = health
            if 'xxx' not in kwargs:
                kwargs['xxx'] = default_xxx_filter.isXXXTorrentMetadataDict(kwargs)
            super().__init__(*args, **kwargs)
            if 'tracker_info' in kwargs:
                self.add_tracker(kwargs['tracker_info'])
            if notifier:
                notifier[notifications.new_torrent_metadata_created](infohash=kwargs.get('infohash'), title=self.title)
                self.tag_processor_version = tag_processor_version

        def add_tracker(self, tracker_url):
            if False:
                i = 10
                return i + 15
            sanitized_url = get_uniformed_tracker_url(tracker_url)
            if sanitized_url:
                tracker = db.TrackerState.get_for_update(url=sanitized_url) or db.TrackerState(url=sanitized_url)
                self.health.trackers.add(tracker)

        def before_update(self):
            if False:
                print('Hello World!')
            self.add_tracker(self.tracker_info)

        def get_magnet(self):
            if False:
                i = 10
                return i + 15
            return f'magnet:?xt=urn:btih:{hexlify(self.infohash)}&dn={self.title}' + (f'&tr={self.tracker_info}' if self.tracker_info else '')

        @classmethod
        @db_session
        def add_ffa_from_dict(cls, metadata: dict):
            if False:
                print('Hello World!')
            id_ = infohash_to_id(metadata['infohash'])
            ih_blob = metadata['infohash']
            pk_blob = b''
            if cls.exists(lambda g: g.infohash == ih_blob or (g.id_ == id_ and g.public_key == pk_blob)):
                return None
            return cls.from_dict(dict(metadata, public_key=b'', status=COMMITTED, id_=id_))

        @db_session
        def to_simple_dict(self):
            if False:
                return 10
            '\n            Return a basic dictionary with information about the channel.\n            '
            simple_dict = super().to_simple_dict()
            epoch = datetime.utcfromtimestamp(0)
            simple_dict.update({'infohash': hexlify(self.infohash), 'size': self.size, 'num_seeders': self.health.seeders, 'num_leechers': self.health.leechers, 'last_tracker_check': self.health.last_check, 'created': int((self.torrent_date - epoch).total_seconds()), 'tag_processor_version': self.tag_processor_version})
            return simple_dict

        def metadata_conflicting(self, b):
            if False:
                return 10
            a = self.to_dict()
            for comp in ['title', 'size', 'tags', 'torrent_date', 'tracker_info']:
                if comp not in b or str(a[comp]) == str(b[comp]):
                    continue
                return True
            return False

        @classmethod
        @db_session
        def get_with_infohash(cls, infohash):
            if False:
                while True:
                    i = 10
            return cls.select(lambda g: g.infohash == infohash).first()

        @classmethod
        @db_session
        def get_torrent_title(cls, infohash):
            if False:
                i = 10
                return i + 15
            md = cls.get_with_infohash(infohash)
            return md.title if md else None

        def serialized_health(self) -> bytes:
            if False:
                for i in range(10):
                    print('nop')
            health = self.health
            if not health or (not health.seeders and (not health.leechers) and (not health.last_check)):
                return b';'
            return b'%d,%d,%d;' % (health.seeders or 0, health.leechers or 0, health.last_check or 0)
    return TorrentMetadata