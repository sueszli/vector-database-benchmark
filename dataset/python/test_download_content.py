"""Helpers: Download: download_content."""
# pylint: disable=missing-docstring
import os

from aiogithubapi.objects.repository.content import AIOGitHubAPIRepositoryTreeContent
import pytest

from tests.sample_data import response_rate_limit_header


@pytest.mark.asyncio
async def test_download_content(repository, aresponses, tmp_path):
    aresponses.add(
        "raw.githubusercontent.com",
        "/test/test/main/test/path/file.file",
        "get",
        aresponses.Response(body="test", headers=response_rate_limit_header),
    )

    repository.content.path.remote = ""
    repository.content.path.local = tmp_path
    repository.tree = [
        AIOGitHubAPIRepositoryTreeContent(
            {"path": "test/path/file.file", "type": "blob"}, "test/test", "main"
        )
    ]

    await repository.download_content()
    assert os.path.exists(f"{repository.content.path.local}/test/path/file.file")


@pytest.mark.asyncio
async def test_download_content_integration(repository_integration, aresponses, hacs):
    aresponses.add(
        "raw.githubusercontent.com",
        aresponses.ANY,
        "get",
        aresponses.Response(body="", headers=response_rate_limit_header),
    )
    aresponses.add(
        "raw.githubusercontent.com",
        aresponses.ANY,
        "get",
        aresponses.Response(body="", headers=response_rate_limit_header),
    )
    aresponses.add(
        "raw.githubusercontent.com",
        aresponses.ANY,
        "get",
        aresponses.Response(body="", headers=response_rate_limit_header),
    )
    aresponses.add(
        "raw.githubusercontent.com",
        aresponses.ANY,
        "get",
        aresponses.Response(body="", headers=response_rate_limit_header),
    )
    repository_integration.data.domain = "test"
    repository_integration.content.path.local = repository_integration.localpath
    repository_integration.content.path.remote = "custom_components/test"
    integration_files = [
        "__init__.py",
        "sensor.py",
        "translations/en.json",
        "manifest.json",
    ]
    for integration_file in integration_files:
        repository_integration.tree.append(
            AIOGitHubAPIRepositoryTreeContent(
                {"path": f"custom_components/test/{integration_file}", "type": "blob"},
                "test/test",
                "main",
            )
        )
    await repository_integration.download_content()
    for path in repository_integration.tree:
        assert os.path.exists(f"{hacs.core.config_path}/{path.full_path}")
