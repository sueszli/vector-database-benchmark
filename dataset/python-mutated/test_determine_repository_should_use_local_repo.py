"""Tests around using locally cached cookiecutter template repositories."""
from pathlib import Path
import pytest
from cookiecutter import repository, exceptions

def test_finds_local_repo(tmp_path):
    if False:
        while True:
            i = 10
    'A valid local repository should be returned.'
    (project_dir, cleanup) = repository.determine_repo_dir('tests/fake-repo', abbreviations={}, clone_to_dir=str(tmp_path), checkout=None, no_input=True)
    assert 'tests/fake-repo' == project_dir
    assert not cleanup

def test_local_repo_with_no_context_raises(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    'A local repository without a cookiecutter.json should raise a     `RepositoryNotFound` exception.'
    template_path = str(Path('tests', 'fake-repo-bad'))
    with pytest.raises(exceptions.RepositoryNotFound) as err:
        repository.determine_repo_dir(template_path, abbreviations={}, clone_to_dir=str(tmp_path), checkout=None, no_input=True)
    assert str(err.value) == 'A valid repository for "{}" could not be found in the following locations:\n{}'.format(template_path, '\n'.join([template_path, str(tmp_path.joinpath('tests', 'fake-repo-bad'))]))

def test_local_repo_typo(tmp_path):
    if False:
        print('Hello World!')
    'An unknown local repository should raise a `RepositoryNotFound`     exception.'
    template_path = str(Path('tests', 'unknown-repo'))
    with pytest.raises(exceptions.RepositoryNotFound) as err:
        repository.determine_repo_dir(template_path, abbreviations={}, clone_to_dir=str(tmp_path), checkout=None, no_input=True)
    assert str(err.value) == 'A valid repository for "{}" could not be found in the following locations:\n{}'.format(template_path, '\n'.join([template_path, str(tmp_path.joinpath('tests', 'unknown-repo'))]))