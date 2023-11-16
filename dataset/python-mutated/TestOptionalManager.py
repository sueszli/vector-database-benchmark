import copy
import pytest

@pytest.mark.usefixtures('resetSettings')
class TestOptionalManager:

    def testDbFill(self, site):
        if False:
            for i in range(10):
                print('nop')
        contents = site.content_manager.contents
        assert len(site.content_manager.hashfield) > 0
        assert contents.db.execute('SELECT COUNT(*) FROM file_optional WHERE is_downloaded = 1').fetchone()[0] == len(site.content_manager.hashfield)

    def testSetContent(self, site):
        if False:
            i = 10
            return i + 15
        contents = site.content_manager.contents
        new_content = copy.deepcopy(contents['content.json'])
        new_content['files_optional']['testfile'] = {'size': 1234, 'sha512': 'aaaabbbbcccc'}
        num_optional_files_before = contents.db.execute('SELECT COUNT(*) FROM file_optional').fetchone()[0]
        contents['content.json'] = new_content
        assert contents.db.execute('SELECT COUNT(*) FROM file_optional').fetchone()[0] > num_optional_files_before
        new_content = copy.deepcopy(contents['content.json'])
        del new_content['files_optional']['testfile']
        num_optional_files_before = contents.db.execute('SELECT COUNT(*) FROM file_optional').fetchone()[0]
        contents['content.json'] = new_content
        assert contents.db.execute('SELECT COUNT(*) FROM file_optional').fetchone()[0] < num_optional_files_before

    def testDeleteContent(self, site):
        if False:
            i = 10
            return i + 15
        contents = site.content_manager.contents
        num_optional_files_before = contents.db.execute('SELECT COUNT(*) FROM file_optional').fetchone()[0]
        del contents['content.json']
        assert contents.db.execute('SELECT COUNT(*) FROM file_optional').fetchone()[0] < num_optional_files_before

    def testVerifyFiles(self, site):
        if False:
            i = 10
            return i + 15
        contents = site.content_manager.contents
        new_content = copy.deepcopy(contents['content.json'])
        new_content['files_optional']['testfile'] = {'size': 1234, 'sha512': 'aaaabbbbcccc'}
        contents['content.json'] = new_content
        file_row = contents.db.execute("SELECT * FROM file_optional WHERE inner_path = 'testfile'").fetchone()
        assert not file_row['is_downloaded']
        site.storage.open('testfile', 'wb').write(b'A' * 1234)
        hashfield_len_before = len(site.content_manager.hashfield)
        site.storage.verifyFiles(quick_check=True)
        assert len(site.content_manager.hashfield) == hashfield_len_before + 1
        file_row = contents.db.execute("SELECT * FROM file_optional WHERE inner_path = 'testfile'").fetchone()
        assert file_row['is_downloaded']
        site.storage.delete('testfile')
        site.storage.verifyFiles(quick_check=True)
        file_row = contents.db.execute("SELECT * FROM file_optional WHERE inner_path = 'testfile'").fetchone()
        assert not file_row['is_downloaded']

    def testVerifyFilesSameHashId(self, site):
        if False:
            while True:
                i = 10
        contents = site.content_manager.contents
        new_content = copy.deepcopy(contents['content.json'])
        new_content['files_optional']['testfile1'] = {'size': 1234, 'sha512': 'aaaabbbbcccc'}
        new_content['files_optional']['testfile2'] = {'size': 2345, 'sha512': 'aaaabbbbdddd'}
        contents['content.json'] = new_content
        assert site.content_manager.hashfield.getHashId('aaaabbbbcccc') == site.content_manager.hashfield.getHashId('aaaabbbbdddd')
        site.storage.open('testfile1', 'wb').write(b'A' * 1234)
        site.storage.open('testfile2', 'wb').write(b'B' * 2345)
        site.storage.verifyFiles(quick_check=True)
        assert site.content_manager.isDownloaded('testfile1')
        assert site.content_manager.isDownloaded('testfile2')
        assert site.content_manager.hashfield.getHashId('aaaabbbbcccc') in site.content_manager.hashfield
        site.storage.delete('testfile1')
        site.storage.verifyFiles(quick_check=True)
        assert not site.content_manager.isDownloaded('testfile1')
        assert site.content_manager.isDownloaded('testfile2')
        assert site.content_manager.hashfield.getHashId('aaaabbbbdddd') in site.content_manager.hashfield

    def testIsPinned(self, site):
        if False:
            i = 10
            return i + 15
        assert not site.content_manager.isPinned('data/img/zerotalk-upvote.png')
        site.content_manager.setPin('data/img/zerotalk-upvote.png', True)
        assert site.content_manager.isPinned('data/img/zerotalk-upvote.png')
        assert len(site.content_manager.cache_is_pinned) == 1
        site.content_manager.cache_is_pinned = {}
        assert site.content_manager.isPinned('data/img/zerotalk-upvote.png')

    def testBigfilePieceReset(self, site):
        if False:
            for i in range(10):
                print('nop')
        site.bad_files = {'data/fake_bigfile.mp4|0-1024': 10, 'data/fake_bigfile.mp4|1024-2048': 10, 'data/fake_bigfile.mp4|2048-3064': 10}
        site.onFileDone('data/fake_bigfile.mp4|0-1024')
        assert site.bad_files['data/fake_bigfile.mp4|1024-2048'] == 1
        assert site.bad_files['data/fake_bigfile.mp4|2048-3064'] == 1

    def testOptionalDelete(self, site):
        if False:
            print('Hello World!')
        contents = site.content_manager.contents
        site.content_manager.setPin('data/img/zerotalk-upvote.png', True)
        site.content_manager.setPin('data/img/zeroid.png', False)
        new_content = copy.deepcopy(contents['content.json'])
        del new_content['files_optional']['data/img/zerotalk-upvote.png']
        del new_content['files_optional']['data/img/zeroid.png']
        assert site.storage.isFile('data/img/zerotalk-upvote.png')
        assert site.storage.isFile('data/img/zeroid.png')
        site.storage.writeJson('content.json', new_content)
        site.content_manager.loadContent('content.json', force=True)
        assert not site.storage.isFile('data/img/zeroid.png')
        assert site.storage.isFile('data/img/zerotalk-upvote.png')

    def testOptionalRename(self, site):
        if False:
            print('Hello World!')
        contents = site.content_manager.contents
        site.content_manager.setPin('data/img/zerotalk-upvote.png', True)
        new_content = copy.deepcopy(contents['content.json'])
        new_content['files_optional']['data/img/zerotalk-upvote-new.png'] = new_content['files_optional']['data/img/zerotalk-upvote.png']
        del new_content['files_optional']['data/img/zerotalk-upvote.png']
        assert site.storage.isFile('data/img/zerotalk-upvote.png')
        assert site.content_manager.isPinned('data/img/zerotalk-upvote.png')
        site.storage.writeJson('content.json', new_content)
        site.content_manager.loadContent('content.json', force=True)
        assert not site.storage.isFile('data/img/zerotalk-upvote.png')
        assert not site.content_manager.isPinned('data/img/zerotalk-upvote.png')
        assert site.content_manager.isPinned('data/img/zerotalk-upvote-new.png')
        assert site.storage.isFile('data/img/zerotalk-upvote-new.png')