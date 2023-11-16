"""Helpers: Information: get_repository."""
# pylint: disable=missing-docstring
import json

import aiohttp
import pytest

from custom_components.hacs.exceptions import HacsException

from tests.sample_data import (
    release_data,
    repository_data,
    response_rate_limit_header,
    response_rate_limit_header_with_limit,
)


@pytest.mark.asyncio
async def test_get_releases(aresponses, repository_integration):
    aresponses.add(
        "api.github.com",
        "/rate_limit",
        "get",
        aresponses.Response(body=b"{}", headers=response_rate_limit_header, status=200),
    )
    aresponses.add(
        "api.github.com",
        "/repos/test/test",
        "get",
        aresponses.Response(body=json.dumps(repository_data), headers=response_rate_limit_header),
    )
    aresponses.add(
        "api.github.com",
        "/rate_limit",
        "get",
        aresponses.Response(body=b"{}", headers=response_rate_limit_header, status=200),
    )
    aresponses.add(
        "api.github.com",
        "/repos/test/test/releases",
        "get",
        aresponses.Response(body=json.dumps(release_data), headers=response_rate_limit_header),
    )

    (
        repository_integration.repository_object,
        _,
    ) = await repository_integration.async_get_legacy_repository_object()
    releases = await repository_integration.get_releases()
    assert "3" in [x.tag_name for x in releases]


@pytest.mark.asyncio
async def test_get_releases_exception(aresponses, repository_integration):
    aresponses.add(
        "api.github.com",
        "/rate_limit",
        "get",
        aresponses.Response(body=b"{}", headers=response_rate_limit_header, status=200),
    )
    aresponses.add(
        "api.github.com",
        "/repos/test/test",
        "get",
        aresponses.Response(body=json.dumps(repository_data), headers=response_rate_limit_header),
    )
    aresponses.add(
        "api.github.com",
        "/rate_limit",
        "get",
        aresponses.Response(body=b"{}", headers=response_rate_limit_header_with_limit, status=403),
    )
    aresponses.add(
        "api.github.com",
        "/repos/test/test/releases",
        "get",
        aresponses.Response(
            body=json.dumps(release_data),
            headers=response_rate_limit_header_with_limit,
            status=403,
        ),
    )
    (
        repository_integration.repository_object,
        _,
    ) = await repository_integration.async_get_legacy_repository_object()
    with pytest.raises(HacsException):
        await repository_integration.get_releases()
