import uuid
import google.auth
import pytest
from ..routes.create_kms_route import create_route_to_windows_activation_host
from ..routes.delete import delete_route
from ..routes.list import list_routes
PROJECT = google.auth.default()[1]

def test_route_create_delete():
    if False:
        while True:
            i = 10
    route_name = 'test-route' + uuid.uuid4().hex[:10]
    route = create_route_to_windows_activation_host(PROJECT, 'global/networks/default', route_name)
    try:
        assert route.name == route_name
        assert route.dest_range == '35.190.247.13/32'
    finally:
        delete_route(PROJECT, route_name)
        for route in list_routes(PROJECT):
            if route.name == route_name:
                pytest.fail(f'Failed to delete test route {route_name}.')