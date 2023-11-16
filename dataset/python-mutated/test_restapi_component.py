from unittest.mock import MagicMock
import pytest
from tribler.core.components.bandwidth_accounting.bandwidth_accounting_component import BandwidthAccountingComponent
from tribler.core.components.database.database_component import DatabaseComponent
from tribler.core.components.exceptions import NoneComponent
from tribler.core.components.gigachannel.gigachannel_component import GigaChannelComponent
from tribler.core.components.ipv8.ipv8_component import Ipv8Component
from tribler.core.components.key.key_component import KeyComponent
from tribler.core.components.knowledge.knowledge_component import KnowledgeComponent
from tribler.core.components.libtorrent.libtorrent_component import LibtorrentComponent
from tribler.core.components.metadata_store.metadata_store_component import MetadataStoreComponent
from tribler.core.components.reporter.reported_error import ReportedError
from tribler.core.components.resource_monitor.resource_monitor_component import ResourceMonitorComponent
from tribler.core.components.restapi.rest.rest_endpoint import RESTEndpoint
from tribler.core.components.restapi.restapi_component import RESTComponent
from tribler.core.components.session import Session
from tribler.core.components.socks_servers.socks_servers_component import SocksServersComponent

async def test_rest_component(tribler_config):
    components = [KeyComponent(), RESTComponent(), Ipv8Component(), LibtorrentComponent(), ResourceMonitorComponent(), BandwidthAccountingComponent(), GigaChannelComponent(), KnowledgeComponent(), SocksServersComponent(), MetadataStoreComponent(), DatabaseComponent()]
    async with Session(tribler_config, components) as session:
        comp = session.get_instance(RESTComponent)
        assert comp.started_event.is_set() and (not comp.failed)
        assert comp.rest_manager
        comp._events_endpoint.on_tribler_exception = MagicMock()
        error = ReportedError(type='', text='text', event={})
        comp._core_exception_handler.report_callback(error)
        comp._events_endpoint.on_tribler_exception.assert_called_with(error)

@pytest.fixture
def endpoint_cls():
    if False:
        for i in range(10):
            print('nop')

    class Endpoint(RESTEndpoint):
        ...
    return Endpoint

@pytest.fixture
async def rest_component():
    component = RESTComponent()
    component.root_endpoint = MagicMock()
    return component

def test_maybe_add_check_args(rest_component, endpoint_cls):
    if False:
        i = 10
        return i + 15
    rest_component.maybe_add(endpoint_cls, NoneComponent())
    rest_component.root_endpoint.assert_not_called()
    rest_component.maybe_add(endpoint_cls, NoneComponent(), 'some arg')
    rest_component.root_endpoint.assert_not_called()

def test_maybe_add_check_kwargs(rest_component, endpoint_cls):
    if False:
        i = 10
        return i + 15
    rest_component.maybe_add(endpoint_cls, component=NoneComponent())
    rest_component.root_endpoint.assert_not_called()
    rest_component.maybe_add(endpoint_cls, component=NoneComponent(), another='kwarg')
    rest_component.root_endpoint.assert_not_called()

def test_maybe_add(rest_component, endpoint_cls):
    if False:
        return 10
    rest_component.maybe_add(endpoint_cls, 'arg')
    assert rest_component.root_endpoint.add_endpoint.called_once()