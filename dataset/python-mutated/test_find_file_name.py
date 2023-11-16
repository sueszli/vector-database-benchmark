"""Helpers: Install: find_file_name."""
from aiogithubapi.models.release import GitHubReleaseModel
from aiogithubapi.objects.repository.content import AIOGitHubAPIRepositoryTreeContent

def test_find_file_name_base(repository_plugin):
    if False:
        for i in range(10):
            print('nop')
    repository_plugin.tree = [AIOGitHubAPIRepositoryTreeContent({'path': 'test.js', 'type': 'blob'}, 'test/test', 'main')]
    repository_plugin.update_filenames()
    assert repository_plugin.data.file_name == 'test.js'
    assert repository_plugin.content.path.remote == ''

def test_find_file_name_root(repository_plugin):
    if False:
        return 10
    repository_plugin.repository_manifest.content_in_root = True
    repository_plugin.tree = [AIOGitHubAPIRepositoryTreeContent({'path': 'test.js', 'type': 'blob'}, 'test/test', 'main')]
    repository_plugin.update_filenames()
    assert repository_plugin.data.file_name == 'test.js'
    assert repository_plugin.content.path.remote == ''

def test_find_file_name_dist(repository_plugin):
    if False:
        return 10
    repository_plugin.tree = [AIOGitHubAPIRepositoryTreeContent({'path': 'dist/test.js', 'type': 'blob'}, 'test/test', 'main')]
    repository_plugin.update_filenames()
    assert repository_plugin.data.file_name == 'test.js'
    assert repository_plugin.content.path.remote == 'dist'

def test_find_file_name_different_name(repository_plugin):
    if False:
        while True:
            i = 10
    repository_plugin.repository_manifest.filename = 'card.js'
    repository_plugin.tree = [AIOGitHubAPIRepositoryTreeContent({'path': 'card.js', 'type': 'blob'}, 'test/test', 'main')]
    repository_plugin.update_filenames()
    assert repository_plugin.data.file_name == 'card.js'
    assert repository_plugin.content.path.remote == ''

def test_find_file_release(repository_plugin):
    if False:
        return 10
    repository_plugin.releases.objects = [GitHubReleaseModel({'tag_name': '3', 'assets': [{'name': 'test.js'}]})]
    repository_plugin.update_filenames()
    assert repository_plugin.data.file_name == 'test.js'
    assert repository_plugin.content.path.remote == 'release'

def test_find_file_release_no_asset(repository_plugin):
    if False:
        print('Hello World!')
    repository_plugin.releases.objects = [GitHubReleaseModel({'tag_name': '3', 'assets': []})]
    repository_plugin.tree = [AIOGitHubAPIRepositoryTreeContent({'path': 'test.js', 'type': 'blob'}, 'test/test', 'main')]
    repository_plugin.update_filenames()
    assert repository_plugin.data.file_name == 'test.js'
    assert repository_plugin.content.path.remote == ''

def test_find_file_name_base_theme(repository_theme):
    if False:
        i = 10
        return i + 15
    repository_theme.tree = [AIOGitHubAPIRepositoryTreeContent({'path': 'themes/test.yaml', 'type': 'blob'}, 'test/test', 'main')]
    repository_theme.update_filenames()
    assert repository_theme.data.file_name == 'test.yaml'
    assert repository_theme.data.name == 'test'

def test_find_file_name_base_python_script(repository_python_script):
    if False:
        for i in range(10):
            print('nop')
    repository_python_script.tree = [AIOGitHubAPIRepositoryTreeContent({'path': 'python_scripts/test.py', 'type': 'blob'}, 'test/test', 'main')]
    repository_python_script.update_filenames()
    assert repository_python_script.data.file_name == 'test.py'
    assert repository_python_script.data.name == 'test'