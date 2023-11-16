import os
import zipfile
from datetime import datetime, timezone
from github import GithubException
from . import Framework

class FileLikeStub:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.dat = b'I wanted to come up with some clever phrase or something here to test with but my mind is blank.'
        self.file_length = len(self.dat)
        self.index = 0

    def read(self, size=-1):
        if False:
            return 10
        if size < 0 or size is None:
            start = self.index
            self.index = self.file_length
            return self.dat[start:]
        else:
            start = self.index
            end = start + size
            self.index = end
            return self.dat[start:end]
repo_name = 'RepoTest'
user = 'rickrickston123'
release_id = 28524234
author_id = 64711998
tag = 'v1.0'
create_date = datetime(2020, 7, 12, 7, 34, 42, tzinfo=timezone.utc)
publish_date = datetime(2020, 7, 14, 0, 58, 20, tzinfo=timezone.utc)

class GitRelease(Framework.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.new_tag = 'v1.25.2'
        self.content_path = 'content.txt'
        self.artifact_path = 'archive.zip'
        with open(self.content_path, 'w') as zip_content:
            zip_content.write('Pedro for president.')
        artifact = zipfile.ZipFile(self.artifact_path, 'w')
        artifact.write(self.content_path)
        artifact.close()
        self.repo = self.g.get_user(user).get_repo(repo_name)
        self.release = self.repo.get_release(release_id)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        if os.path.exists(self.content_path):
            os.remove(self.content_path)
        if os.path.exists(self.artifact_path):
            os.remove(self.artifact_path)
        super().tearDown()

    def setUpNewRelease(self):
        if False:
            i = 10
            return i + 15
        repo = self.repo
        commit_sha = repo.get_commits()[0].sha
        self.new_release = repo.create_git_tag_and_release(self.new_tag, 'tag message', 'release title', 'release message', commit_sha, 'commit')
        self.new_release_id = self.new_release.id

    def tearDownNewRelease(self):
        if False:
            print('Hello World!')
        try:
            new_release = self.repo.get_release(self.new_release_id)
            new_release.delete_release()
        except GithubException:
            pass

    def testAttributes(self):
        if False:
            return 10
        release = self.release
        self.assertEqual(release.id, release_id)
        self.assertEqual(release.tag_name, tag)
        self.assertEqual(release.target_commitish, 'master')
        self.assertEqual(release.upload_url, 'https://uploads.github.com/repos/{}/{}/releases/{}/assets{{?name,label}}'.format(user, repo_name, release_id))
        self.assertEqual(release.body, 'Body')
        self.assertEqual(release.title, 'Test')
        self.assertFalse(release.draft)
        self.assertFalse(release.prerelease)
        self.assertEqual(release.url, f'https://api.github.com/repos/{user}/{repo_name}/releases/{release_id}')
        self.assertEqual(release.author._rawData['login'], user)
        self.assertEqual(release.author.login, user)
        self.assertEqual(release.author.id, author_id)
        self.assertEqual(release.author.type, 'User')
        self.assertEqual(release.html_url, f'https://github.com/{user}/{repo_name}/releases/tag/{tag}')
        self.assertEqual(release.created_at, create_date)
        self.assertEqual(release.published_at, publish_date)
        self.assertEqual(release.tarball_url, f'https://api.github.com/repos/{user}/{repo_name}/tarball/{tag}')
        self.assertEqual(release.zipball_url, f'https://api.github.com/repos/{user}/{repo_name}/zipball/{tag}')
        self.assertEqual(repr(release), 'GitRelease(title="Test")')
        self.assertEqual(len(release.assets), 1)
        self.assertEqual(repr(release.assets[0]), f"""GitReleaseAsset(url="https://api.github.com/repos/{user}/{repo_name}/releases/assets/{release.raw_data['assets'][0]['id']}")""")

    def testGetRelease(self):
        if False:
            return 10
        release_by_id = self.release
        release_by_tag = self.repo.get_release(tag)
        self.assertEqual(release_by_id, release_by_tag)

    def testGetLatestRelease(self):
        if False:
            for i in range(10):
                print('nop')
        latest_release = self.repo.get_latest_release()
        self.assertEqual(latest_release.tag_name, tag)

    def testGetAssets(self):
        if False:
            return 10
        repo = self.repo
        release = self.release
        self.assertEqual(release.id, release_id)
        asset_list = [x for x in release.get_assets()]
        self.assertTrue(asset_list is not None)
        self.assertEqual(len(asset_list), 1)
        asset_id = asset_list[0].id
        asset = repo.get_release_asset(asset_id)
        self.assertTrue(asset is not None)
        self.assertEqual(asset.id, asset_id)

    def testDelete(self):
        if False:
            for i in range(10):
                print('nop')
        self.setUpNewRelease()
        self.new_release.delete_release()
        self.tearDownNewRelease()

    def testUpdate(self):
        if False:
            for i in range(10):
                print('nop')
        self.setUpNewRelease()
        release = self.new_release
        new_release = release.update_release('Updated Test', 'Updated Body')
        self.assertEqual(new_release.title, 'Updated Test')
        self.assertEqual(new_release.body, 'Updated Body')
        self.tearDownNewRelease()

    def testUploadAsset(self):
        if False:
            return 10
        self.setUpNewRelease()
        release = self.new_release
        self.assertEqual(release.id, self.new_release_id)
        release.upload_asset(self.artifact_path, 'unit test artifact', 'application/zip')
        self.tearDownNewRelease()

    def testUploadAssetWithName(self):
        if False:
            return 10
        self.setUpNewRelease()
        release = self.new_release
        r = release.upload_asset(self.artifact_path, name='foobar.zip', content_type='application/zip')
        self.assertEqual(r.name, 'foobar.zip')
        self.tearDownNewRelease()

    def testCreateGitTagAndRelease(self):
        if False:
            for i in range(10):
                print('nop')
        self.setUpNewRelease()
        release = self.new_release
        self.assertEqual(release.tag_name, self.new_tag)
        self.assertEqual(release.body, 'release message')
        self.assertEqual(release.title, 'release title')
        self.assertEqual(release.author._rawData['login'], user)
        self.assertEqual(release.html_url, f'https://github.com/{user}/{repo_name}/releases/tag/{self.new_tag}')
        self.tearDownNewRelease()

    def testUploadAssetFromMemory(self):
        if False:
            return 10
        self.setUpNewRelease()
        release = self.new_release
        content_size = os.path.getsize(self.content_path)
        with open(self.content_path, 'rb') as f:
            release.upload_asset_from_memory(f, content_size, name='file_name', content_type='text/plain', label='unit test artifact')
        asset_list = [x for x in release.get_assets()]
        self.assertTrue(asset_list is not None)
        self.assertEqual(len(asset_list), 1)
        self.tearDownNewRelease()

    def testUploadAssetFileLike(self):
        if False:
            print('Hello World!')
        self.setUpNewRelease()
        file_like = FileLikeStub()
        release = self.new_release
        release.upload_asset_from_memory(file_like, file_like.file_length, name='file_like', content_type='text/plain', label='another unit test artifact')
        asset_list = [x for x in release.get_assets()]
        self.assertTrue(asset_list is not None)
        self.assertEqual(len(asset_list), 1)
        self.tearDownNewRelease()