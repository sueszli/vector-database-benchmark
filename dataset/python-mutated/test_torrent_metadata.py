from datetime import datetime
from time import time
from unittest.mock import MagicMock, Mock
import pytest
from ipv8.keyvault.crypto import default_eccrypto
from pony import orm
from pony.orm import db_session
from tribler.core.components.conftest import TEST_PERSONAL_KEY
from tribler.core.components.libtorrent.torrentdef import TorrentDef
from tribler.core.components.metadata_store.db.orm_bindings.channel_node import TODELETE
from tribler.core.components.metadata_store.db.orm_bindings.discrete_clock import clock
from tribler.core.components.metadata_store.db.orm_bindings.torrent_metadata import tdef_to_metadata_dict
from tribler.core.components.metadata_store.db.serialization import CHANNEL_TORRENT, REGULAR_TORRENT
from tribler.core.tests.tools.common import TORRENT_UBUNTU_FILE
from tribler.core.utilities.utilities import random_infohash
EMPTY_BLOB = b''

def rnd_torrent():
    if False:
        while True:
            i = 10
    return {'title': '', 'infohash': random_infohash(), 'torrent_date': datetime(1970, 1, 1), 'tags': 'video'}

@db_session
def test_serialization(metadata_store):
    if False:
        return 10
    '\n    Test converting torrent metadata to serialized data\n    '
    torrent_metadata = metadata_store.TorrentMetadata.from_dict({'infohash': random_infohash()})
    assert torrent_metadata.serialized()

async def test_create_ffa_from_dict(metadata_store):
    """
    Test creating a free-for-all torrent entry
    """
    tdef = await TorrentDef.load(TORRENT_UBUNTU_FILE)
    with db_session:
        signed_entry = metadata_store.TorrentMetadata.from_dict(tdef_to_metadata_dict(tdef))
        metadata_store.TorrentMetadata.add_ffa_from_dict(tdef_to_metadata_dict(tdef))
        assert metadata_store.TorrentMetadata.select(lambda g: g.public_key == EMPTY_BLOB).count() == 0
        signed_entry.delete()
        metadata_store.TorrentMetadata.add_ffa_from_dict(tdef_to_metadata_dict(tdef))
        assert metadata_store.TorrentMetadata.select(lambda g: g.public_key == EMPTY_BLOB).count() == 1

async def test_sanitize_tdef(metadata_store):
    tdef = await TorrentDef.load(TORRENT_UBUNTU_FILE)
    tdef.metainfo['creation date'] = -100000
    with db_session:
        assert metadata_store.TorrentMetadata.from_dict(tdef_to_metadata_dict(tdef))

@db_session
def test_get_magnet(metadata_store):
    if False:
        print('Hello World!')
    '\n    Test converting torrent metadata to a magnet link\n    '
    torrent_metadata = metadata_store.TorrentMetadata.from_dict({'infohash': random_infohash()})
    assert torrent_metadata.get_magnet()
    torrent_metadata2 = metadata_store.TorrentMetadata.from_dict({'title': '💩', 'infohash': random_infohash()})
    assert torrent_metadata2.get_magnet()

@db_session
def test_search_keyword(metadata_store):
    if False:
        i = 10
        return i + 15
    '\n    Test searching in a database with some torrent metadata inserted\n    '
    torrent1 = metadata_store.TorrentMetadata.from_dict(dict(rnd_torrent(), title='foo bar 123'))
    torrent2 = metadata_store.TorrentMetadata.from_dict(dict(rnd_torrent(), title='eee 123'))
    metadata_store.TorrentMetadata.from_dict(dict(rnd_torrent(), title='xoxoxo bar'))
    metadata_store.TorrentMetadata.from_dict(dict(rnd_torrent(), title='xoxoxo bar'))
    metadata_store.TorrentMetadata.from_dict(dict(rnd_torrent(), title='"'))
    metadata_store.TorrentMetadata.from_dict(dict(rnd_torrent(), title="'"))
    orm.flush()
    results = metadata_store.search_keyword('foo')[:]
    assert len(results) == 1
    assert results[0].rowid == torrent1.rowid
    results = metadata_store.search_keyword('eee')[:]
    assert len(results) == 1
    assert results[0].rowid == torrent2.rowid
    results = metadata_store.search_keyword('123')[:]
    assert len(results) == 2

@db_session
def test_search_deduplicated(metadata_store):
    if False:
        print('Hello World!')
    '\n    Test SQL-query base deduplication of search results with the same infohash\n    '
    key2 = default_eccrypto.generate_key('curve25519')
    torrent = rnd_torrent()
    metadata_store.TorrentMetadata.from_dict(dict(torrent, title='foo bar 123'))
    metadata_store.TorrentMetadata.from_dict(dict(torrent, title='eee 123', sign_with=key2))
    results = metadata_store.search_keyword('foo')[:]
    assert len(results) == 1

def test_search_empty_query(metadata_store):
    if False:
        while True:
            i = 10
    '\n    Test whether an empty query returns nothing\n    '
    assert not metadata_store.search_keyword(None)[:]

@db_session
def test_unicode_search(metadata_store):
    if False:
        return 10
    '\n    Test searching in the database with unicode characters\n    '
    metadata_store.TorrentMetadata.from_dict(dict(rnd_torrent(), title='я маленький апельсин'))
    results = metadata_store.search_keyword('маленький')[:]
    assert len(results) == 1

@db_session
def test_wildcard_search(metadata_store):
    if False:
        i = 10
        return i + 15
    '\n    Test searching in the database with a wildcard\n    '
    metadata_store.TorrentMetadata.from_dict(dict(rnd_torrent(), title='foobar 123'))
    metadata_store.TorrentMetadata.from_dict(dict(rnd_torrent(), title='foobla 123'))
    assert not metadata_store.search_keyword('*')[:]
    assert len(metadata_store.search_keyword('foobl*')[:]) == 1
    assert len(metadata_store.search_keyword('foo*')[:]) == 2
    assert len(metadata_store.search_keyword('("12"* AND "foobl"*)')[:]) == 1

@db_session
def test_stemming_search(metadata_store):
    if False:
        i = 10
        return i + 15
    '\n    Test searching in the database with stemmed words\n    '
    torrent = metadata_store.TorrentMetadata.from_dict(dict(rnd_torrent(), title='mountains sheep', tags='video'))
    results = metadata_store.search_keyword('mountain')[:]
    assert torrent.rowid == results[0].rowid
    results = metadata_store.search_keyword('sheeps')[:]
    assert torrent.rowid == results[0].rowid

@db_session
def test_get_autocomplete_terms(metadata_store):
    if False:
        while True:
            i = 10
    '\n    Test fetching autocompletion terms from the database\n    '
    metadata_store.TorrentMetadata.from_dict(dict(rnd_torrent(), title='foo: bar baz', tags='video'))
    metadata_store.TorrentMetadata.from_dict(dict(rnd_torrent(), title='foo - bar, xyz', tags='video'))
    metadata_store.TorrentMetadata.from_dict(dict(rnd_torrent(), title='barbarian xyz!', tags='video'))
    metadata_store.TorrentMetadata.from_dict(dict(rnd_torrent(), title='n.a.m.e: foobar', tags='video'))
    metadata_store.TorrentMetadata.from_dict(dict(rnd_torrent(), title='xyz n.a.m.e', tags='video'))
    autocomplete_terms = metadata_store.get_auto_complete_terms('', 10)
    assert autocomplete_terms == []
    autocomplete_terms = metadata_store.get_auto_complete_terms('foo', 10)
    assert set(autocomplete_terms) == {'foo: bar', 'foo - bar', 'foobar'}
    autocomplete_terms = metadata_store.get_auto_complete_terms('foo: bar', 10)
    assert set(autocomplete_terms) == {'foo: bar baz', 'foo: bar, xyz'}
    autocomplete_terms = metadata_store.get_auto_complete_terms('foo ', 10)
    assert set(autocomplete_terms) == {'foo bar'}
    autocomplete_terms = metadata_store.get_auto_complete_terms('bar', 10)
    assert set(autocomplete_terms) == {'bar baz', 'bar, xyz', 'barbarian'}
    autocomplete_terms = metadata_store.get_auto_complete_terms('barb', 10)
    assert set(autocomplete_terms) == {'barbarian'}
    autocomplete_terms = metadata_store.get_auto_complete_terms('barbarian', 10)
    assert set(autocomplete_terms) == {'barbarian xyz'}
    autocomplete_terms = metadata_store.get_auto_complete_terms('barbarian ', 10)
    assert set(autocomplete_terms) == {'barbarian xyz'}
    autocomplete_terms = metadata_store.get_auto_complete_terms('barbarian x', 10)
    assert set(autocomplete_terms) == {'barbarian xyz'}
    autocomplete_terms = metadata_store.get_auto_complete_terms('n.a.m', 10)
    assert set(autocomplete_terms) == {'n.a.m.e'}
    autocomplete_terms = metadata_store.get_auto_complete_terms('n.a.m.', 10)
    assert set(autocomplete_terms) == {'n.a.m.e'}
    autocomplete_terms = metadata_store.get_auto_complete_terms('n.a.m.e', 10)
    assert set(autocomplete_terms) == {'n.a.m.e', 'n.a.m.e: foobar'}
    autocomplete_terms = metadata_store.get_auto_complete_terms('n.a.m.e ', 10)
    assert set(autocomplete_terms) == {'n.a.m.e ', 'n.a.m.e foobar'}
    autocomplete_terms = metadata_store.get_auto_complete_terms('n.a.m.e f', 10)
    assert set(autocomplete_terms) == {'n.a.m.e foobar'}

@db_session
def test_get_autocomplete_terms_max(metadata_store):
    if False:
        i = 10
        return i + 15
    '\n    Test fetching autocompletion terms from the database with a maximum number of terms\n    '
    metadata_store.TorrentMetadata.from_dict(dict(rnd_torrent(), title='mountains sheeps wolf', tags='video'))
    metadata_store.TorrentMetadata.from_dict(dict(rnd_torrent(), title='lakes sheep', tags='video'))
    metadata_store.TorrentMetadata.from_dict(dict(rnd_torrent(), title='regular sheepish guy', tags='video'))
    autocomplete_terms = metadata_store.get_auto_complete_terms('sheep', 2)
    assert len(autocomplete_terms) == 2
    autocomplete_terms = metadata_store.get_auto_complete_terms('.', 2)

@db_session
def test_get_entries_for_infohashes(metadata_store):
    if False:
        i = 10
        return i + 15
    infohash1 = random_infohash()
    infohash2 = random_infohash()
    infohash3 = random_infohash()
    metadata_store.TorrentMetadata(title='title', infohash=infohash1, size=0, sign_with=TEST_PERSONAL_KEY)
    metadata_store.TorrentMetadata(title='title', infohash=infohash2, size=0, sign_with=TEST_PERSONAL_KEY)

    def count(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return len(metadata_store.get_entries_query(*args, **kwargs))
    assert count(infohash=infohash3) == 0
    assert count(infohash=infohash1) == 1
    assert count(infohash_set={infohash1, infohash2}) == 2
    assert count(infohash=infohash1, infohash_set={infohash1, infohash2}) == 2

@db_session
def test_get_entries(metadata_store):
    if False:
        return 10
    '\n    Test base method for getting torrents\n    '
    clock.clock = 0
    tlist = []
    keys = [*(default_eccrypto.generate_key('curve25519') for _ in range(4)), metadata_store.ChannelNode._my_key]
    for (ind, key) in enumerate(keys):
        metadata_store.ChannelMetadata(title='channel%d' % ind, subscribed=ind % 2 == 0, infohash=random_infohash(), num_entries=5, sign_with=key)
        tlist.extend([metadata_store.TorrentMetadata(title='torrent%d' % torrent_ind, infohash=random_infohash(), size=123, sign_with=key) for torrent_ind in range(5)])
    tlist[-1].xxx = 1
    tlist[-2].status = TODELETE
    torrents = metadata_store.get_entries(first=1, last=5)
    assert len(torrents) == 5
    count = metadata_store.get_entries_count(metadata_type=REGULAR_TORRENT)
    assert count == 25
    channel_pk = metadata_store.ChannelNode._my_key.pub().key_to_bin()[10:]
    args = dict(channel_pk=channel_pk, hide_xxx=True, exclude_deleted=True, metadata_type=REGULAR_TORRENT)
    torrents = metadata_store.get_entries_query(**args)[:]
    assert tlist[-5:-2] == list(torrents)[::-1]
    count = metadata_store.get_entries_count(**args)
    assert count == 3
    args = dict(sort_by='title', channel_pk=channel_pk, origin_id=0, metadata_type=REGULAR_TORRENT)
    torrents = metadata_store.get_entries(first=1, last=10, **args)
    assert len(torrents) == 5
    count = metadata_store.get_entries_count(**args)
    assert count == 5
    args = dict(sort_by='size', sort_desc=True, channel_pk=channel_pk, origin_id=0)
    torrents = metadata_store.get_entries(first=1, last=10, **args)
    assert torrents[0].metadata_type == CHANNEL_TORRENT
    args = dict(sort_by='size', sort_desc=False, channel_pk=channel_pk, origin_id=0)
    torrents = metadata_store.get_entries(first=1, last=10, **args)
    assert torrents[-1].metadata_type == CHANNEL_TORRENT
    args = dict(channel_pk=channel_pk, origin_id=0, attribute_ranges=(('timestamp', 3, 30),))
    torrents = metadata_store.get_entries(first=1, last=10, **args)
    assert sorted([t.timestamp for t in torrents]) == list(range(25, 30))
    args = dict(channel_pk=channel_pk, origin_id=0, attribute_ranges=(('timestamp < 3 and g.timestamp', 3, 30),))
    with pytest.raises(AttributeError):
        metadata_store.get_entries(**args)
    with db_session:
        entry = metadata_store.TorrentMetadata(id_=123, infohash=random_infohash())
    args = dict(channel_pk=channel_pk, id_=123)
    torrents = metadata_store.get_entries(first=1, last=10, **args)
    assert list(torrents) == [entry]
    with db_session:
        complete_chan = metadata_store.ChannelMetadata(infohash=random_infohash(), title='bla', local_version=222, timestamp=222)
        incomplete_chan = metadata_store.ChannelMetadata(infohash=random_infohash(), title='bla', local_version=222, timestamp=223)
        channels = metadata_store.get_entries(complete_channel=True)
        assert [complete_chan] == channels

@db_session
def test_get_entries_health_checked_after(metadata_store):
    if False:
        return 10
    t1 = metadata_store.TorrentMetadata(infohash=random_infohash())
    t1.health.last_check = int(time())
    t2 = metadata_store.TorrentMetadata(infohash=random_infohash())
    t2.health.last_check = t1.health.last_check - 10000
    torrents = metadata_store.get_entries(health_checked_after=t2.health.last_check + 1)
    assert torrents == [t1]

@db_session
def test_metadata_conflicting(metadata_store):
    if False:
        while True:
            i = 10
    tdict = dict(rnd_torrent(), title='lakes sheep', tags='video', infohash=b'\x00\xff')
    md = metadata_store.TorrentMetadata.from_dict(tdict)
    assert not md.metadata_conflicting(tdict)
    assert md.metadata_conflicting(dict(tdict, title='bla'))
    tdict.pop('title')
    assert not md.metadata_conflicting(tdict)

@db_session
def test_update_properties(metadata_store):
    if False:
        i = 10
        return i + 15
    '\n    Test the updating of several properties of a TorrentMetadata object\n    '
    metadata = metadata_store.TorrentMetadata(title='foo', infohash=random_infohash())
    orig_timestamp = metadata.timestamp
    assert metadata.update_properties({'status': 456}).status == 456
    assert orig_timestamp == metadata.timestamp
    assert metadata.update_properties({'title': 'bar'}).title == 'bar'
    assert metadata.timestamp > orig_timestamp

@db_session
def test_popular_torrens_with_metadata_type(metadata_store):
    if False:
        print('Hello World!')
    '\n    Test that `popular` argument cannot be combiner with `metadata_type` argument\n    '
    with pytest.raises(TypeError):
        metadata_store.get_entries(popular=True)
    metadata_store.get_entries(popular=True, metadata_type=REGULAR_TORRENT)
    with pytest.raises(TypeError):
        metadata_store.get_entries(popular=True, metadata_type=CHANNEL_TORRENT)
    with pytest.raises(TypeError):
        metadata_store.get_entries(popular=True, metadata_type=[REGULAR_TORRENT, CHANNEL_TORRENT])
WRONG_TRACKERS_OBJECTS = [["b'udp://tracker/announce'"], None]

@pytest.mark.parametrize('tracker', WRONG_TRACKERS_OBJECTS)
def test_tdef_to_metadata_dict_wrong_tracker(tracker):
    if False:
        i = 10
        return i + 15
    metadata_dict = tdef_to_metadata_dict(tdef=MagicMock(get_tracker=Mock(return_value=tracker)), category_filter=MagicMock())
    assert metadata_dict['tracker_info'] == ''