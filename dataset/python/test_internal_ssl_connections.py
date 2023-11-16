"""Tests for jupyterhub internal_ssl connections"""
import sys
import time
from unittest import mock

import pytest
from requests.exceptions import ConnectionError, SSLError

from ..utils import AnyTimeoutError
from .test_api import add_user
from .utils import async_requests

ssl_enabled = True

# possible errors raised by ssl failures
SSL_ERROR = (SSLError, ConnectionError)


async def wait_for_spawner(spawner, timeout=10):
    """Wait for an http server to show up

    polling at shorter intervals for early termination
    """
    deadline = time.monotonic() + timeout

    def wait():
        return spawner.server.wait_up(timeout=1, http=True)

    while time.monotonic() < deadline:
        status = await spawner.poll()
        assert status is None
        try:
            await wait()
        except AnyTimeoutError:
            continue
        else:
            break
    await wait()


async def test_connection_hub_wrong_certs(app):
    """Connecting to the internal hub url fails without correct certs"""
    with pytest.raises(SSL_ERROR):
        kwargs = {'verify': False}
        r = await async_requests.get(app.hub.url, **kwargs)
        r.raise_for_status()


async def test_connection_proxy_api_wrong_certs(app):
    """Connecting to the proxy api fails without correct certs"""
    with pytest.raises(SSL_ERROR):
        kwargs = {'verify': False}
        r = await async_requests.get(app.proxy.api_url, **kwargs)
        r.raise_for_status()


async def test_connection_notebook_wrong_certs(app):
    """Connecting to a notebook fails without correct certs"""
    with mock.patch.dict(
        app.config.LocalProcessSpawner,
        {'cmd': [sys.executable, '-m', 'jupyterhub.tests.mocksu']},
    ):
        user = add_user(app.db, app, name='foo')
        await user.spawn()
        await wait_for_spawner(user.spawner)
        spawner = user.spawner
        status = await spawner.poll()
        assert status is None

        with pytest.raises(SSL_ERROR):
            kwargs = {'verify': False}
            r = await async_requests.get(spawner.server.url, **kwargs)
            r.raise_for_status()
