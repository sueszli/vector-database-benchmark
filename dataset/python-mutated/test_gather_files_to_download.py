"""Helpers: Download: gather_files_to_reload."""
from aiogithubapi.models.release import GitHubReleaseModel
from aiogithubapi.objects.repository.content import AIOGitHubAPIRepositoryTreeContent

def test_gather_files_to_download(repository):
    if False:
        for i in range(10):
            print('nop')
    repository.content.path.remote = ''
    repository.tree = [AIOGitHubAPIRepositoryTreeContent({'path': 'test/path/file.file', 'type': 'blob'}, 'test/test', 'main')]
    files = [x.path for x in repository.gather_files_to_download()]
    assert 'test/path/file.file' in files

def test_gather_plugin_files_from_root(repository_plugin):
    if False:
        print('Hello World!')
    repository_plugin.content.path.remote = ''
    repository_plugin.tree = [AIOGitHubAPIRepositoryTreeContent({'path': 'test.js', 'type': 'blob'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': 'dir', 'type': 'tree'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': 'aaaa.js', 'type': 'blob'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': 'dist/test.js', 'type': 'blob'}, 'test/test', 'main')]
    repository_plugin.update_filenames()
    files = [x.path for x in repository_plugin.gather_files_to_download()]
    assert 'test.js' in files
    assert 'dir' not in files
    assert 'aaaa.js' in files
    assert 'dist/test.js' not in files

def test_gather_plugin_files_from_dist(repository_plugin):
    if False:
        while True:
            i = 10
    repository = repository_plugin
    repository.content.path.remote = 'dist'
    repository.data.file_name = 'test.js'
    repository.tree = [AIOGitHubAPIRepositoryTreeContent({'path': 'test.js', 'type': 'blob'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': 'dist/image.png', 'type': 'blob'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': 'dist/test.js', 'type': 'blob'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': 'dist/subdir', 'type': 'tree'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': 'dist/subdir/file.file', 'type': 'blob'}, 'test/test', 'main')]
    files = [x.path for x in repository.gather_files_to_download()]
    assert 'test.js' not in files
    assert 'dist/image.png' in files
    assert 'dist/subdir/file.file' in files
    assert 'dist/subdir' not in files
    assert 'dist/test.js' in files

def test_gather_plugin_multiple_plugin_files_from_dist(repository_plugin):
    if False:
        print('Hello World!')
    repository = repository_plugin
    repository.content.path.remote = 'dist'
    repository.data.file_name = 'test.js'
    repository.tree = [AIOGitHubAPIRepositoryTreeContent({'path': 'test.js', 'type': 'blob'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': 'dist/test.js', 'type': 'blob'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': 'dist/something_other.js', 'type': 'blob'}, 'test/test', 'main')]
    files = [x.path for x in repository.gather_files_to_download()]
    assert 'test.js' not in files
    assert 'dist/test.js' in files
    assert 'dist/something_other.js' in files

def test_gather_plugin_files_from_release(repository_plugin):
    if False:
        while True:
            i = 10
    repository = repository_plugin
    repository.data.file_name = 'test.js'
    repository.data.releases = True
    release = GitHubReleaseModel({'tag_name': '3', 'assets': [{'name': 'test.js'}]})
    repository.releases.objects = [release]
    files = [x.name for x in repository.gather_files_to_download()]
    assert 'test.js' in files

def test_gather_plugin_files_from_release_multiple(repository_plugin):
    if False:
        while True:
            i = 10
    repository = repository_plugin
    repository.data.file_name = 'test.js'
    repository.data.releases = True
    repository.releases.objects = [GitHubReleaseModel({'tag_name': '3', 'assets': [{'name': 'test.js'}, {'name': 'test.png'}]})]
    files = [x.name for x in repository.gather_files_to_download()]
    assert 'test.js' in files
    assert 'test.png' in files

def test_gather_zip_release(repository_plugin):
    if False:
        print('Hello World!')
    repository = repository_plugin
    repository.data.file_name = 'test.zip'
    repository.repository_manifest.zip_release = True
    repository.repository_manifest.filename = 'test.zip'
    repository.releases.objects = [GitHubReleaseModel({'tag_name': '3', 'assets': [{'name': 'test.zip'}]})]
    files = [x.name for x in repository.gather_files_to_download()]
    assert 'test.zip' in files

def test_single_file_repo(repository):
    if False:
        print('Hello World!')
    repository.content.single = True
    repository.data.file_name = 'test.file'
    repository.tree = [AIOGitHubAPIRepositoryTreeContent({'path': 'test.file', 'type': 'blob'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': 'dir', 'type': 'tree'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': 'test.yaml', 'type': 'blob'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': 'readme.md', 'type': 'blob'}, 'test/test', 'main')]
    files = [x.path for x in repository.gather_files_to_download()]
    assert 'readme.md' not in files
    assert 'test.yaml' not in files
    assert 'test.file' in files

def test_gather_content_in_root_theme(repository_theme):
    if False:
        print('Hello World!')
    repository = repository_theme
    repository.repository_manifest.content_in_root = True
    repository.content.path.remote = ''
    repository.data.file_name = 'test.yaml'
    repository.tree = [AIOGitHubAPIRepositoryTreeContent({'path': 'test.yaml', 'type': 'blob'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': 'dir', 'type': 'tree'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': 'test2.yaml', 'type': 'blob'}, 'test/test', 'main')]
    files = [x.path for x in repository.gather_files_to_download()]
    assert 'test2.yaml' not in files
    assert 'test.yaml' in files

def test_gather_netdaemon_files_base(repository_netdaemon):
    if False:
        print('Hello World!')
    repository = repository_netdaemon
    repository.tree = [AIOGitHubAPIRepositoryTreeContent({'path': 'test.cs', 'type': 'blob'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': 'apps/test/test.cs', 'type': 'blob'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': 'apps/test/test.yaml', 'type': 'blob'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': '.github/file.file', 'type': 'blob'}, 'test/test', 'main')]
    files = [x.path for x in repository.gather_files_to_download()]
    assert '.github/file.file' not in files
    assert 'test.cs' not in files
    assert 'apps/test/test.cs' in files
    assert 'apps/test/test.yaml' in files

def test_gather_appdaemon_files_base(repository_appdaemon):
    if False:
        i = 10
        return i + 15
    repository = repository_appdaemon
    repository.tree = [AIOGitHubAPIRepositoryTreeContent({'path': 'test.py', 'type': 'blob'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': 'apps/test/test.py', 'type': 'blob'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': '.github/file.file', 'type': 'blob'}, 'test/test', 'main')]
    files = [x.path for x in repository.gather_files_to_download()]
    assert '.github/file.file' not in files
    assert 'test.py' not in files
    assert 'apps/test/test.py' in files

def test_gather_appdaemon_files_with_subdir(repository_appdaemon):
    if False:
        for i in range(10):
            print('nop')
    repository = repository_appdaemon
    repository.data.file_name = 'test.py'
    repository.tree = [AIOGitHubAPIRepositoryTreeContent({'path': 'test.py', 'type': 'blob'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': 'apps/test/test.py', 'type': 'blob'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': 'apps/test/core/test.py', 'type': 'blob'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': 'apps/test/devices/test.py', 'type': 'blob'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': 'apps/test/test/test.py', 'type': 'blob'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': '.github/file.file', 'type': 'blob'}, 'test/test', 'main')]
    files = [x.path for x in repository.gather_files_to_download()]
    assert '.github/file.file' not in files
    assert 'test.py' not in files
    assert 'apps/test/test.py' in files
    assert 'apps/test/devices/test.py' in files
    assert 'apps/test/test/test.py' in files
    assert 'apps/test/core/test.py' in files

def test_gather_plugin_multiple_files_in_root(repository_plugin):
    if False:
        return 10
    repository = repository_plugin
    repository.content.path.remote = ''
    repository.data.file_name = 'test.js'
    repository.tree = [AIOGitHubAPIRepositoryTreeContent({'path': 'test.js', 'type': 'blob'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': 'dep1.js', 'type': 'blob'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': 'dep2.js', 'type': 'blob'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': 'dep3.js', 'type': 'blob'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': 'info.md', 'type': 'blob'}, 'test/test', 'main')]
    files = [x.path for x in repository.gather_files_to_download()]
    assert 'test.js' in files
    assert 'dep1.js' in files
    assert 'dep2.js' in files
    assert 'dep3.js' in files
    assert 'info.md' not in files

def test_gather_plugin_different_card_name(repository_plugin):
    if False:
        for i in range(10):
            print('nop')
    repository = repository_plugin
    repository.content.path.remote = ''
    repository.data.file_name = 'card.js'
    repository.tree = [AIOGitHubAPIRepositoryTreeContent({'path': 'card.js', 'type': 'blob'}, 'test/test', 'main'), AIOGitHubAPIRepositoryTreeContent({'path': 'info.md', 'type': 'blob'}, 'test/test', 'main')]
    repository_plugin.update_filenames()
    files = [x.path for x in repository.gather_files_to_download()]
    assert 'card.js' in files
    assert 'info.md' not in files