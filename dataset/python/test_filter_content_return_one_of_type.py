"""Helpers: Filters: filter_content_return_one_of_type."""
# pylint: disable=missing-docstring
from aiogithubapi.objects.repository.content import AIOGitHubAPIRepositoryTreeContent

from custom_components.hacs.utils import filters


def test_valid_objects():
    tree = [
        AIOGitHubAPIRepositoryTreeContent(
            {"path": "test/file.file", "type": "blob"}, "test/test", "main"
        ),
        AIOGitHubAPIRepositoryTreeContent(
            {"path": "test/newfile.file", "type": "blob"}, "test/test", "main"
        ),
        AIOGitHubAPIRepositoryTreeContent(
            {"path": "test/file.png", "type": "blob"}, "test/test", "main"
        ),
    ]
    files = [
        x.filename
        for x in filters.filter_content_return_one_of_type(tree, "test", "file", "full_path")
    ]
    assert "file.file" in files
    assert "newfile.file" not in files
    assert "file.png" in files


def test_valid_list():
    tree = ["test/file.file", "test/newfile.file", "test/file.png"]

    files = filters.filter_content_return_one_of_type(tree, "test", "file")
    assert "test/file.file" in files
    assert "test/newfile.file" not in files
    assert "test/file.png" in files
