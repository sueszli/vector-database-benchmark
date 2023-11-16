import asyncio
import os
import random
from asyncio import Future
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from ipv8.util import succeed
from pony.orm import db_session
from tribler.core.components.gigachannel_manager.gigachannel_manager import GigaChannelManager
from tribler.core.components.libtorrent.torrentdef import TorrentDef
from tribler.core.components.metadata_store.db.orm_bindings.channel_node import NEW
from tribler.core.tests.tools.base_test import MockObject
from tribler.core.tests.tools.common import TORRENT_UBUNTU_FILE
from tribler.core.utilities.simpledefs import DownloadStatus
from tribler.core.utilities.utilities import random_infohash
update_metainfo = None

@pytest.fixture
def torrent_template():
    if False:
        return 10
    return {'title': '', 'infohash': b'', 'torrent_date': datetime(1970, 1, 1), 'tags': 'video'}

@pytest.fixture
async def personal_channel(metadata_store):
    global update_metainfo
    tdef = await TorrentDef.load(TORRENT_UBUNTU_FILE)
    with db_session:
        chan = metadata_store.ChannelMetadata.create_channel(title='my test chan', description='test')
        chan.add_torrent_to_channel(tdef, None)
        update_metainfo = chan.commit_channel_torrent()
        return chan

@pytest.fixture(name='gigachannel_manager')
async def gigachannel_manager_fixture(metadata_store):
    chanman = GigaChannelManager(state_dir=metadata_store.channels_dir.parent, metadata_store=metadata_store, download_manager=MagicMock(), notifier=MagicMock())
    yield chanman
    await chanman.shutdown()

async def test_regen_personal_channel_no_torrent(personal_channel, gigachannel_manager):
    """
    Test regenerating a non-existing personal channel torrent at startup
    """
    gigachannel_manager.download_manager.get_download = lambda _: None
    gigachannel_manager.regenerate_channel_torrent = MagicMock()
    await gigachannel_manager.check_and_regen_personal_channels()
    gigachannel_manager.regenerate_channel_torrent.assert_called_once()

async def test_regen_personal_channel_damaged_torrent(personal_channel, gigachannel_manager):
    """
    Test regenerating a damaged personal channel torrent at startup
    """
    complete = Future()

    async def mock_regen(*_, **__):
        complete.set_result(True)
    gigachannel_manager.check_and_regen_personal_channel_torrent = mock_regen
    gigachannel_manager.start()
    await complete

async def test_regenerate_channel_torrent(personal_channel, metadata_store, gigachannel_manager):
    with db_session:
        (chan_pk, chan_id) = (personal_channel.public_key, personal_channel.id_)
        channel_dir = Path(metadata_store.ChannelMetadata._channels_dir) / Path(personal_channel.dirname)
        for f in channel_dir.iterdir():
            f.unlink()
    assert await gigachannel_manager.regenerate_channel_torrent(chan_pk, chan_id + 1) is None
    gigachannel_manager.download_manager.get_downloads_by_name = lambda *_: [MagicMock()]
    downloads_to_remove = []

    async def mock_remove_download(download_obj, **_):
        downloads_to_remove.append(download_obj)
    gigachannel_manager.download_manager.remove_download = mock_remove_download
    metadata_store.ChannelMetadata.consolidate_channel_torrent = lambda *_: None
    assert await gigachannel_manager.regenerate_channel_torrent(chan_pk, chan_id) is None
    assert len(downloads_to_remove) == 1
    gigachannel_manager.updated_my_channel = MagicMock()
    metadata_store.ChannelMetadata.consolidate_channel_torrent = lambda *_: MagicMock()
    with patch('tribler.core.components.libtorrent.torrentdef.TorrentDef.load_from_dict'):
        await gigachannel_manager.regenerate_channel_torrent(chan_pk, chan_id)
        gigachannel_manager.updated_my_channel.assert_called_once()

async def test_updated_my_channel(personal_channel, gigachannel_manager, tmpdir):
    tdef = TorrentDef.load_from_dict(update_metainfo)
    gigachannel_manager.download_manager.start_download = AsyncMock()
    gigachannel_manager.download_manager.download_exists = lambda *_: False
    await gigachannel_manager.updated_my_channel(tdef)
    gigachannel_manager.download_manager.start_download.assert_called_once()

async def test_check_and_regen_personal_channel_torrent_wait(personal_channel, gigachannel_manager):
    await gigachannel_manager.check_and_regen_personal_channel_torrent(channel_pk=personal_channel.public_key, channel_id=personal_channel.id_, channel_download=MagicMock(wait_for_status=AsyncMock()), timeout=0.5)

async def test_check_and_regen_personal_channel_torrent_sleep(personal_channel, gigachannel_manager):

    async def mock_wait(*_):
        await asyncio.sleep(3)
    mock_regen = AsyncMock()
    with patch.object(GigaChannelManager, 'regenerate_channel_torrent', mock_regen):
        await gigachannel_manager.check_and_regen_personal_channel_torrent(channel_pk=personal_channel.public_key, channel_id=personal_channel.id_, channel_download=MagicMock(wait_for_status=mock_wait), timeout=0.5)
    mock_regen.assert_called_once()

async def test_check_channels_updates(personal_channel, gigachannel_manager, metadata_store):
    torrents_added = 0
    with db_session:
        my_channel = metadata_store.ChannelMetadata.get_my_channels().first()
        my_channel.local_version -= 1
        metadata_store.ChannelMetadata(title='bla1', public_key=b'123', signature=b'345', skip_key_check=True, timestamp=123, local_version=123, subscribed=True, infohash=random_infohash())
        metadata_store.ChannelMetadata(title='bla2', public_key=b'124', signature=b'346', skip_key_check=True, timestamp=123, local_version=122, subscribed=False, infohash=random_infohash())
        chan3 = metadata_store.ChannelMetadata(title='bla3', public_key=b'125', signature=b'347', skip_key_check=True, timestamp=123, local_version=122, subscribed=True, infohash=random_infohash())

        def mock_download_channel(chan1):
            if False:
                print('Hello World!')
            nonlocal torrents_added
            torrents_added += 1
            assert chan1 == chan3
        gigachannel_manager.download_channel = mock_download_channel

        @db_session
        def fake_get_metainfo(infohash, **_):
            if False:
                return 10
            return {'info': {'name': metadata_store.ChannelMetadata.get(infohash=infohash).dirname}}
        gigachannel_manager.download_manager.get_metainfo = fake_get_metainfo
        gigachannel_manager.download_manager.metainfo_requests = {}
        gigachannel_manager.download_manager.download_exists = lambda _: False
        gigachannel_manager.check_channels_updates()
        assert torrents_added == 1
        gigachannel_manager.download_manager = MockObject()
        gigachannel_manager.download_manager.download_exists = lambda _: True
        mock_download = MagicMock()
        mock_download.get_state.get_status = DownloadStatus.SEEDING
        gigachannel_manager.download_manager.get_download = lambda _: mock_download

        def mock_process_channel_dir(c, _):
            if False:
                return 10
            assert c == chan3
        gigachannel_manager.process_channel_dir = mock_process_channel_dir
        gigachannel_manager.check_channels_updates()
    await gigachannel_manager.process_queued_channels()
    assert not gigachannel_manager.channels_processing_queue

@db_session
def test_check_channel_updates_for_different_states(gigachannel_manager, metadata_store):
    if False:
        while True:
            i = 10

    def random_subscribed_channel():
        if False:
            while True:
                i = 10
        return metadata_store.ChannelMetadata(title=f'Channel {random.randint(0, 100)}', public_key=os.urandom(32), signature=os.urandom(32), skip_key_check=True, timestamp=123, local_version=122, subscribed=True, infohash=random_infohash())
    channel_with_metainfo = random_subscribed_channel()
    already_downloaded_channel = random_subscribed_channel()
    non_downloaded_channel = random_subscribed_channel()

    def mock_get_metainfo(infohash):
        if False:
            return 10
        return MagicMock() if infohash == channel_with_metainfo.infohash else None
    gigachannel_manager.download_manager.metainfo_requests = MagicMock(get=mock_get_metainfo)

    def mock_download_exists(infohash):
        if False:
            i = 10
            return i + 15
        return infohash == already_downloaded_channel.infohash
    gigachannel_manager.download_manager.download_exists = mock_download_exists
    gigachannel_manager.download_channel = MagicMock()

    def mock_get_download(infohash):
        if False:
            while True:
                i = 10
        if infohash != already_downloaded_channel.infohash:
            return None
        seeding_state = MagicMock(get_status=lambda : DownloadStatus.SEEDING)
        return MagicMock(get_state=lambda : seeding_state)
    gigachannel_manager.download_manager.get_download = mock_get_download
    gigachannel_manager.check_channels_updates()
    gigachannel_manager.download_channel.assert_called_once_with(non_downloaded_channel)
    assert len(gigachannel_manager.channels_processing_queue) == 1
    assert already_downloaded_channel.infohash in gigachannel_manager.channels_processing_queue

async def test_remove_cruft_channels(torrent_template, personal_channel, gigachannel_manager, metadata_store):
    remove_list = []
    with db_session:
        personal_channel = metadata_store.ChannelMetadata.get_my_channels().first()
        my_chan_old_infohash = personal_channel.infohash
        metadata_store.TorrentMetadata.from_dict(dict(torrent_template, origin_id=personal_channel.id_, status=NEW))
        personal_channel.commit_channel_torrent()
        chan2 = metadata_store.ChannelMetadata(title='bla1', infohash=b'123', public_key=b'123', signature=b'345', skip_key_check=True, timestamp=123, local_version=123, subscribed=True)
        chan3 = metadata_store.ChannelMetadata(title='bla2', infohash=b'124', public_key=b'124', signature=b'346', skip_key_check=True, timestamp=123, local_version=123, subscribed=False)

    class MockDownload(MockObject):

        def __init__(self, infohash, dirname):
            if False:
                print('Hello World!')
            self.infohash = infohash
            self.dirname = dirname
            self.tdef = MockObject()
            self.tdef.get_name_utf8 = lambda : self.dirname
            self.tdef.get_infohash = lambda : infohash

        def get_def(self):
            if False:
                while True:
                    i = 10
            a = MockObject()
            a.infohash = self.infohash
            a.get_name_utf8 = lambda : self.dirname
            a.get_infohash = lambda : self.infohash
            return a
    mock_dl_list = [MockDownload(my_chan_old_infohash, personal_channel.dirname), MockDownload(personal_channel.infohash, personal_channel.dirname), MockDownload(b'12331244', chan2.dirname), MockDownload(chan2.infohash, chan2.dirname), MockDownload(b'1231551', chan3.dirname), MockDownload(chan3.infohash, chan3.dirname), MockDownload(b'333', 'blabla')]

    def mock_get_channel_downloads(**_):
        if False:
            print('Hello World!')
        return mock_dl_list

    def mock_remove(infohash, remove_content=False):
        if False:
            return 10
        nonlocal remove_list
        d = Future()
        d.set_result(None)
        remove_list.append((infohash, remove_content))
        return d
    gigachannel_manager.download_manager.get_channel_downloads = mock_get_channel_downloads
    gigachannel_manager.download_manager.remove_download = mock_remove
    gigachannel_manager.remove_cruft_channels()
    await gigachannel_manager.process_queued_channels()
    assert remove_list == [(mock_dl_list[0], False), (mock_dl_list[2], False), (mock_dl_list[4], True), (mock_dl_list[5], True), (mock_dl_list[6], True)]
initiated_download = False

async def test_reject_malformed_channel(gigachannel_manager, metadata_store):
    global initiated_download
    with db_session:
        channel = metadata_store.ChannelMetadata(title='bla1', public_key=b'123', infohash=random_infohash())

    def mock_get_metainfo_bad(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        return succeed({b'info': {b'name': b'bla'}})

    def mock_get_metainfo_good(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        return succeed({b'info': {b'name': channel.dirname.encode('utf-8')}})
    initiated_download = False

    async def mock_download_from_tdef(*_, **__):
        global initiated_download
        initiated_download = True
        mock_dl = MockObject()
        mock_dl.future_finished = succeed(None)
        return mock_dl
    gigachannel_manager.download_manager.start_download = mock_download_from_tdef
    gigachannel_manager.download_manager.get_metainfo = mock_get_metainfo_bad
    await gigachannel_manager.download_channel(channel)
    assert not initiated_download
    with patch.object(TorrentDef, '__init__', lambda *_, **__: None):
        gigachannel_manager.download_manager.get_metainfo = mock_get_metainfo_good
        await gigachannel_manager.download_channel(channel)
        assert initiated_download