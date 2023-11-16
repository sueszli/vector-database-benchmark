import os
from binascii import unhexlify
from datetime import datetime
from lz4.frame import LZ4FrameCompressor
from pony import orm
from pony.orm import db_session, raw_sql, select
from tribler.core.components.libtorrent.utils.libtorrent_helper import libtorrent as lt
from tribler.core.components.metadata_store.db.orm_bindings.channel_node import CHANNEL_DESCRIPTION_FLAG, CHANNEL_THUMBNAIL_FLAG, COMMITTED, LEGACY_ENTRY, NEW, PUBLIC_KEY_LEN, TODELETE, UPDATED
from tribler.core.components.metadata_store.db.orm_bindings.discrete_clock import clock
from tribler.core.components.metadata_store.db.serialization import CHANNEL_TORRENT, ChannelMetadataPayload, HealthItemsPayload
from tribler.core.utilities.path_util import Path
from tribler.core.utilities.simpledefs import CHANNEL_STATE
from tribler.core.utilities.unicode import hexlify
from tribler.core.utilities.utilities import random_infohash
CHANNEL_DIR_NAME_PK_LENGTH = 32
CHANNEL_DIR_NAME_ID_LENGTH = 16
CHANNEL_DIR_NAME_LENGTH = CHANNEL_DIR_NAME_PK_LENGTH + CHANNEL_DIR_NAME_ID_LENGTH
BLOB_EXTENSION = '.mdblob'
LZ4_END_MARK_SIZE = 4
HEALTH_ITEM_HEADER_SIZE = 4
LZ4_EMPTY_ARCHIVE = unhexlify('04224d184040c000000000')

def chunks(l, n):
    if False:
        print('Hello World!')
    'Yield successive n-sized chunks from l.'
    for i in range(0, len(l), n):
        yield l[i:i + n]

def create_torrent_from_dir(directory, torrent_filename):
    if False:
        print('Hello World!')
    fs = lt.file_storage()
    lt.add_files(fs, str(directory))
    t = lt.create_torrent(fs)
    t.set_priv(False)
    lt.set_piece_hashes(t, str(directory.parent))
    torrent = t.generate()
    with open(torrent_filename, 'wb') as f:
        f.write(lt.bencode(torrent))
    infohash = lt.torrent_info(torrent).info_hash().to_bytes()
    return (torrent, infohash)

def get_mdblob_sequence_number(filename):
    if False:
        for i in range(10):
            print('nop')
    filepath = Path(filename)
    if filepath.suffixes == [BLOB_EXTENSION]:
        return int(filename.stem)
    if filepath.suffixes == [BLOB_EXTENSION, '.lz4']:
        return int(Path(filepath.stem).stem)
    return None

def entries_to_chunk(metadata_list, chunk_size, start_index=0, include_health=False):
    if False:
        while True:
            i = 10
    '\n    :param metadata_list: the list of metadata to process.\n    :param chunk_size: the desired chunk size limit, in bytes.\n    :param start_index: the index of the element of metadata_list from which the processing should start.\n    :param include_health: if True, put metadata health information into the chunk.\n    :return: (chunk, last_entry_index) tuple, where chunk is the resulting chunk in string form and\n        last_entry_index is the index of the element of the input list that was put into the chunk the last.\n    '
    if start_index >= len(metadata_list):
        raise Exception('Could not serialize chunk: incorrect start_index', metadata_list, chunk_size, start_index)
    compressor = MetadataCompressor(chunk_size, include_health)
    index = start_index
    while index < len(metadata_list):
        metadata = metadata_list[index]
        was_able_to_add = compressor.put(metadata)
        if not was_able_to_add:
            break
        index += 1
    return (compressor.close(), index)

class MetadataCompressor:
    """
    This class provides methods to put serialized data of one or more metadata entries into a single binary chunk.

    The data is added incrementally until it stops fitting into the designated chunk size. The first entry is added
    regardless of violating the chunk size limit.

    The chunk format is:

        <LZ4-compressed sequence of serialized metadata entries>
        [<optional HealthItemsPayload>]

    The optional health information is serialized separately, as it was not originally included in the serialized
    metadata format. If present, it contains the same number of items as the serialized list of metadata
    entries. The N-th health info item in the health block corresponds to the N-th metadata entry.

    For the details of the health info format see the documentation: doc/metadata_store/serialization_format.rst

    While it is possible to put the health info items into the second LZ4-compressed frame, it is more efficient to
    serialize them without any compression. The reason for this is that a typical health info item has a 1-byte
    length (about 17 bytes if a torrent has actual health information), and the number of items is few for a single
    chunk (usually less then 10 items). If we use LZ4 compressor, we want to use it incrementally in order to detect
    when items stop fitting into a chunk. LZ4 algorithm cannot compress such small items efficiently in an incremental
    fashion, and the resulting "compressed" size can be significantly bigger than the original data size.
    """

    def __init__(self, chunk_size: int, include_health: bool=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param chunk_size: the desired chunk size limit, in bytes.\n        :param include_health: if True, put metadata health information into the chunk.\n        '
        self.chunk_size = chunk_size
        self.include_health = include_health
        self.compressor = LZ4FrameCompressor(auto_flush=True)
        assert self.compressor.__enter__() is self.compressor
        metadata_header: bytes = self.compressor.begin()
        self.count = 0
        self.size = len(metadata_header) + LZ4_END_MARK_SIZE
        self.metadata_buffer = [metadata_header]
        if include_health:
            self.health_buffer = []
            self.size += HEALTH_ITEM_HEADER_SIZE
        else:
            self.health_buffer = None
        self.closed = False

    def put(self, metadata) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Tries to add a metadata entry to chunk. The first entry is always added successfully. Then next entries are\n        added only if it possible to fit data into the chunk.\n\n        :param metadata: a metadata entry to process.\n        :return: False if it was not possible to fit data into the chunk\n        '
        if self.closed:
            raise TypeError('Compressor is already closed')
        metadata_bytes = metadata.serialized_delete() if metadata.status == TODELETE else metadata.serialized()
        compressed_metadata_bytes = self.compressor.compress(metadata_bytes)
        new_size = self.size + len(compressed_metadata_bytes)
        health_bytes = b''
        if self.include_health:
            health_bytes = metadata.serialized_health()
            new_size += len(health_bytes)
        if new_size > self.chunk_size and self.count > 0:
            return False
        self.count += 1
        self.size = new_size
        self.metadata_buffer.append(compressed_metadata_bytes)
        if self.include_health:
            self.health_buffer.append(health_bytes)
        return True

    def close(self) -> bytes:
        if False:
            print('Hello World!')
        '\n        Closes compressor object and returns packed data.\n\n        :return: serialized binary data\n        '
        if self.closed:
            raise TypeError('Compressor is already closed')
        self.closed = True
        end_mark = self.compressor.flush()
        self.metadata_buffer.append(end_mark)
        result = b''.join(self.metadata_buffer)
        self.compressor.__exit__(None, None, None)
        if self.include_health:
            result += HealthItemsPayload(b''.join(self.health_buffer)).serialize()
        return result

def define_binding(db):
    if False:
        for i in range(10):
            print('nop')

    class ChannelMetadata(db.TorrentMetadata, db.CollectionNode):
        """
        This ORM binding represents Channel entries in the GigaChannel system. Each channel is a Collection that
        additionally has Torrent properties, such as infohash, etc. The torrent properties are used to associate
        a torrent that holds the contents of the channel dumped on the disk in the serialized form.
        Methods for committing channels into the torrent form are implemented in this class.
        """
        _discriminator_ = CHANNEL_TORRENT
        start_timestamp = orm.Optional(int, size=64, default=0)
        subscribed = orm.Optional(bool, default=False)
        share = orm.Optional(bool, default=False)
        votes = orm.Optional(float, default=0.0)
        individual_votes = orm.Set('ChannelVote', reverse='channel')
        local_version = orm.Optional(int, size=64, default=0)
        votes_scaling = 1.0
        _payload_class = ChannelMetadataPayload
        _channels_dir = None
        _category_filter = None
        _CHUNK_SIZE_LIMIT = 1 * 1024 * 1024
        payload_arguments = _payload_class.__init__.__code__.co_varnames[:_payload_class.__init__.__code__.co_argcount][1:]
        nonpersonal_attributes = set(db.CollectionNode.nonpersonal_attributes)
        infohash_to_channel_name_cache = {}

        @classmethod
        @db_session
        def get_my_channels(cls):
            if False:
                return 10
            return ChannelMetadata.select(lambda g: g.origin_id == 0 and g.public_key == cls._my_key.pub().key_to_bin()[10:])

        @classmethod
        @db_session
        def create_channel(cls, title, description='', origin_id=0):
            if False:
                print('Hello World!')
            '\n            Create a channel and sign it with a given key.\n            :param title: The title of the channel\n            :param description: The description of the channel\n            :param origin_id: id_ of the parent channel\n            :return: The channel metadata\n            '
            my_channel = cls(origin_id=origin_id, public_key=cls._my_key.pub().key_to_bin()[10:], title=title, tags=description, subscribed=True, share=True, status=NEW, infohash=random_infohash())
            my_channel.sign()
            return my_channel

        @db_session
        def consolidate_channel_torrent(self):
            if False:
                while True:
                    i = 10
            '\n            Delete the channel dir contents and create it anew.\n            Use it to consolidate fragmented channel torrent directories.\n            :param key: The public/private key, used to sign the data\n            '
            db.CollectionNode.collapse_deleted_subtrees()
            commit_queue = self.get_contents_to_commit()
            for entry in commit_queue:
                if entry.status == TODELETE:
                    entry.delete()
            folder = Path(self._channels_dir) / self.dirname
            if not folder.is_dir():
                os.makedirs(folder)
            for filename in os.listdir(folder):
                file_path = folder / filename
                if filename.endswith(BLOB_EXTENSION) or filename.endswith(BLOB_EXTENSION + '.lz4'):
                    os.unlink(Path.fix_win_long_file(file_path))
            start_timestamp = clock.tick()

            def update_timestamps_recursive(node):
                if False:
                    print('Hello World!')
                if issubclass(type(node), db.CollectionNode):
                    for child in node.contents:
                        update_timestamps_recursive(child)
                if node.status in [COMMITTED, UPDATED, NEW]:
                    node.status = UPDATED
                    node.timestamp = clock.tick()
                    node.sign()
            update_timestamps_recursive(self)
            return self.commit_channel_torrent(new_start_timestamp=start_timestamp)

        def update_channel_torrent(self, metadata_list):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Channel torrents are append-only to support seeding the old versions\n            from the same dir and avoid updating already downloaded blobs.\n            :param metadata_list: The list of metadata entries to add to the torrent dir.\n            ACHTUNG: TODELETE entries _MUST_ be sorted to the end of the list to prevent channel corruption!\n            :return The newly create channel torrent infohash, final timestamp for the channel and torrent date\n            '
            channel_dir = Path(self._channels_dir / self.dirname).absolute()
            if not channel_dir.is_dir():
                os.makedirs(Path.fix_win_long_file(channel_dir))
            existing_contents = sorted(channel_dir.iterdir())
            last_existing_blob_number = get_mdblob_sequence_number(existing_contents[-1]) if existing_contents else None
            index = 0
            while index < len(metadata_list):
                (data, index) = entries_to_chunk(metadata_list, self._CHUNK_SIZE_LIMIT, start_index=index)
                if metadata_list[index - 1].status == TODELETE:
                    blob_timestamp = clock.tick()
                else:
                    blob_timestamp = metadata_list[index - 1].timestamp
                if index >= len(metadata_list):
                    blob_timestamp = clock.tick()
                assert last_existing_blob_number is None or blob_timestamp > last_existing_blob_number
                blob_filename = Path(channel_dir, str(blob_timestamp).zfill(12) + BLOB_EXTENSION + '.lz4')
                assert not blob_filename.exists()
                blob_filename.write_bytes(data)
                last_existing_blob_number = blob_timestamp
            with db_session:
                thumb_exists = db.ChannelThumbnail.exists(lambda g: g.public_key == self.public_key and g.origin_id == self.id_ and (g.status != TODELETE))
                descr_exists = db.ChannelDescription.exists(lambda g: g.public_key == self.public_key and g.origin_id == self.id_ and (g.status != TODELETE))
                flags = CHANNEL_THUMBNAIL_FLAG * int(thumb_exists) + CHANNEL_DESCRIPTION_FLAG * int(descr_exists)
            (torrent, infohash) = create_torrent_from_dir(channel_dir, self._channels_dir / (self.dirname + '.torrent'))
            torrent_date = datetime.utcfromtimestamp(torrent[b'creation date'])
            return ({'infohash': infohash, 'timestamp': last_existing_blob_number, 'torrent_date': torrent_date, 'reserved_flags': flags}, torrent)

        def commit_channel_torrent(self, new_start_timestamp=None, commit_list=None):
            if False:
                return 10
            '\n            Collect new/uncommitted and marked for deletion metadata entries, commit them to a channel torrent and\n            remove the obsolete entries if the commit succeeds.\n            :param new_start_timestamp: change the start_timestamp of the committed channel entry to this value\n            :param commit_list: the list of ORM objects to commit into this channel torrent\n            :return The new infohash, should be used to update the downloads\n            '
            md_list = commit_list or self.get_contents_to_commit()
            if not md_list:
                return None
            try:
                (update_dict, torrent) = self.update_channel_torrent(md_list)
            except OSError:
                self._logger.error('Error during channel torrent commit, not going to garbage collect the channel. Channel %s', hexlify(self.public_key))
                return None
            if new_start_timestamp:
                update_dict['start_timestamp'] = new_start_timestamp
            for (attr, val) in update_dict.items():
                setattr(self, attr, val)
            self.local_version = self.timestamp
            self.sign()
            for g in md_list:
                if g.status in [NEW, UPDATED]:
                    g.status = COMMITTED
                elif g.status == TODELETE:
                    g.delete()
            self.status = COMMITTED
            self.to_file(self._channels_dir / (self.dirname + BLOB_EXTENSION))
            self._logger.info('Channel %s committed with %i new entries. New version is %i', hexlify(self.public_key), len(md_list), update_dict['timestamp'])
            return torrent

        @property
        def dirname(self):
            if False:
                for i in range(10):
                    print('nop')
            return hexlify(self.public_key)[:CHANNEL_DIR_NAME_PK_LENGTH] + f'{self.id_:0>16x}'

        @classmethod
        @db_session
        def get_channels_by_title(cls, title):
            if False:
                return 10
            return cls.select(lambda g: g.title == title)

        @classmethod
        @db_session
        def get_channel_with_infohash(cls, infohash):
            if False:
                for i in range(10):
                    print('nop')
            return cls.get(infohash=infohash)

        @classmethod
        @db_session
        def get_channel_with_dirname(cls, dirname):
            if False:
                for i in range(10):
                    print('nop')
            pk_part = dirname[:-CHANNEL_DIR_NAME_ID_LENGTH]

            def extend_to_bitmask(txt):
                if False:
                    return 10
                return txt + '0' * (PUBLIC_KEY_LEN * 2 - CHANNEL_DIR_NAME_LENGTH)
            pk_binmask_start = "x'" + extend_to_bitmask(pk_part) + "'"
            pk_plus_one = f'{int(pk_part, 16) + 1:X}'.zfill(len(pk_part))
            pk_binmask_end = "x'" + extend_to_bitmask(pk_plus_one) + "'"
            sql = 'g.public_key >= ' + pk_binmask_start + ' AND g.public_key < ' + pk_binmask_end
            id_part = dirname[-CHANNEL_DIR_NAME_ID_LENGTH:]
            id_ = int(id_part, 16)
            return orm.select((g for g in cls if g.id_ == id_ and raw_sql(sql))).first()

        @classmethod
        @db_session
        def get_updated_channels(cls):
            if False:
                i = 10
                return i + 15
            return select((g for g in cls if g.subscribed == 1 and g.status != LEGACY_ENTRY and (g.local_version < g.timestamp) and (g.public_key != cls._my_key.pub().key_to_bin()[10:])))

        @property
        @db_session
        def state(self):
            if False:
                print('Hello World!')
            '\n            This property describes the current state of the channel.\n            :return: Text-based status\n            '
            if self.is_personal:
                return CHANNEL_STATE.PERSONAL.value
            if self.status == LEGACY_ENTRY:
                return CHANNEL_STATE.LEGACY.value
            if self.local_version == self.timestamp:
                return CHANNEL_STATE.COMPLETE.value
            if self.local_version > 0:
                return CHANNEL_STATE.UPDATING.value
            if self.subscribed:
                return CHANNEL_STATE.METAINFO_LOOKUP.value
            return CHANNEL_STATE.PREVIEW.value

        def to_simple_dict(self, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Return a basic dictionary with information about the channel.\n            '
            result = super().to_simple_dict(**kwargs)
            result.update({'state': self.state, 'subscribed': self.subscribed, 'votes': self.votes / db.ChannelMetadata.votes_scaling, 'dirty': self.dirty if self.is_personal else False})
            return result

        @classmethod
        def get_channel_name_cached(cls, dl_name, infohash):
            if False:
                return 10
            chan_name = cls.infohash_to_channel_name_cache.get(infohash)
            if chan_name is None:
                chan_name = cls.get_channel_name(dl_name, infohash)
                cls.infohash_to_channel_name_cache[infohash] = chan_name
            return chan_name

        @classmethod
        @db_session
        def get_channel_name(cls, dl_name, infohash):
            if False:
                for i in range(10):
                    print('nop')
            "\n            Try to translate a Tribler download name into matching channel name. By searching for a channel with the\n            given dirname and/or infohash. Try do determine if infohash belongs to an older version of\n            some channel we already have.\n            :param dl_name - name of the download. Should match the directory name of the channel.\n            :param infohash - infohash of the download.\n            :return: Channel title as a string, prefixed with 'OLD:' for older versions\n            "
            channel = cls.get_channel_with_infohash(infohash)
            if not channel:
                try:
                    channel = cls.get_channel_with_dirname(dl_name)
                except UnicodeEncodeError:
                    channel = None
            if not channel:
                return dl_name
            if channel.infohash == infohash:
                return channel.title
            return 'OLD:' + channel.title

        @db_session
        def update_properties(self, update_dict):
            if False:
                return 10
            updated_self = super().update_properties(update_dict)
            if updated_self.origin_id != 0:
                self_dict = updated_self.to_dict()
                updated_self.delete(recursive=False)
                self_dict.pop('rowid')
                self_dict.pop('metadata_type')
                self_dict['sign_with'] = self._my_key
                updated_self = db.CollectionNode.from_dict(self_dict)
            return updated_self

        def make_copy(self, tgt_parent_id, **kwargs):
            if False:
                i = 10
                return i + 15
            return db.CollectionNode.make_copy(self, tgt_parent_id, attributes_override={'infohash': random_infohash()}, **kwargs)
    return ChannelMetadata