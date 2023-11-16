import datetime
from typing import Optional
from pony import orm
from tribler.core.utilities.pony_utils import TrackedDatabase, get_or_create

class TagDatabase:

    def __init__(self, filename: Optional[str]=None, *, create_tables: bool=True, **generate_mapping_kwargs):
        if False:
            while True:
                i = 10
        self.instance = TrackedDatabase()
        self.define_binding(self.instance)
        self.instance.bind('sqlite', filename or ':memory:', create_db=True)
        generate_mapping_kwargs['create_tables'] = create_tables
        self.instance.generate_mapping(**generate_mapping_kwargs)

    @staticmethod
    def define_binding(db):
        if False:
            return 10

        class Peer(db.Entity):
            id = orm.PrimaryKey(int, auto=True)
            public_key = orm.Required(bytes, unique=True)
            added_at = orm.Optional(datetime.datetime, default=datetime.datetime.utcnow)
            operations = orm.Set(lambda : TorrentTagOp)

        class Torrent(db.Entity):
            id = orm.PrimaryKey(int, auto=True)
            infohash = orm.Required(bytes, unique=True)
            tags = orm.Set(lambda : TorrentTag)

        class TorrentTag(db.Entity):
            id = orm.PrimaryKey(int, auto=True)
            torrent = orm.Required(lambda : Torrent)
            tag = orm.Required(lambda : Tag)
            operations = orm.Set(lambda : TorrentTagOp)
            added_count = orm.Required(int, default=0)
            removed_count = orm.Required(int, default=0)
            local_operation = orm.Optional(int)
            orm.composite_key(torrent, tag)

            @property
            def score(self):
                if False:
                    while True:
                        i = 10
                return self.added_count - self.removed_count

            def update_counter(self, operation: int, increment: int=1, is_local_peer: bool=False):
                if False:
                    i = 10
                    return i + 15
                " Update TorrentTag's counter\n                Args:\n                    operation: Tag operation\n                    increment:\n                    is_local_peer: The flag indicates whether do we performs operations from a local user or from\n                        a remote user. In case of the local user, his operations will be considered as\n                        authoritative for his (only) local Tribler instance.\n\n                Returns:\n                "
                if is_local_peer:
                    self.local_operation = operation
                if operation == 1:
                    self.added_count += increment
                if operation == 2:
                    self.removed_count += increment

        class Tag(db.Entity):
            id = orm.PrimaryKey(int, auto=True)
            name = orm.Required(str, unique=True)
            torrents = orm.Set(lambda : TorrentTag)

        class TorrentTagOp(db.Entity):
            id = orm.PrimaryKey(int, auto=True)
            torrent_tag = orm.Required(lambda : TorrentTag)
            peer = orm.Required(lambda : Peer)
            operation = orm.Required(int)
            clock = orm.Required(int)
            signature = orm.Required(bytes)
            updated_at = orm.Required(datetime.datetime, default=datetime.datetime.utcnow)
            auto_generated = orm.Required(bool, default=False)
            orm.composite_key(torrent_tag, peer)

    def add_tag_operation(self, infohash: bytes, tag: str, signature: bytes, operation: int, clock: int, creator_public_key: bytes, is_local_peer: bool=False, is_auto_generated: bool=False, counter_increment: int=1) -> bool:
        if False:
            print('Hello World!')
        ' Add the operation that will be applied to the tag.\n        Args:\n            operation: the class describes the adding operation\n            signature: the signature of the operation\n            is_local_peer: local operations processes differently than remote operations. They affects\n                `TorrentTag.local_operation` field which is used in `self.get_tags()` function.\n\n        Returns: True if the operation has been added/updated, False otherwise.\n        '
        peer = get_or_create(self.instance.Peer, public_key=creator_public_key)
        tag = get_or_create(self.instance.Tag, name=tag)
        torrent = get_or_create(self.instance.Torrent, infohash=infohash)
        torrent_tag = get_or_create(self.instance.TorrentTag, tag=tag, torrent=torrent)
        op = self.instance.TorrentTagOp.get_for_update(torrent_tag=torrent_tag, peer=peer)
        if not op:
            self.instance.TorrentTagOp(torrent_tag=torrent_tag, peer=peer, operation=operation, clock=clock, signature=signature, auto_generated=is_auto_generated)
            torrent_tag.update_counter(operation, increment=counter_increment, is_local_peer=is_local_peer)
            return True
        if clock <= op.clock:
            return False
        torrent_tag.update_counter(op.operation, increment=-counter_increment, is_local_peer=is_local_peer)
        torrent_tag.update_counter(operation.operation, increment=counter_increment, is_local_peer=is_local_peer)
        op.set(operation=operation.operation, clock=operation.clock, signature=signature, updated_at=datetime.datetime.utcnow(), auto_generated=is_auto_generated)
        return True

    def shutdown(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.instance.disconnect()