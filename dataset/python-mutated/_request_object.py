"""Represents a request object.
"""

class RequestObject(object):

    def __init__(self, resource_type, operation_type, endpoint_override=None):
        if False:
            while True:
                i = 10
        self.resource_type = resource_type
        self.operation_type = operation_type
        self.endpoint_override = endpoint_override
        self.should_clear_session_token_on_session_read_failure = False
        self.use_preferred_locations = None
        self.location_index_to_route = None
        self.location_endpoint_to_route = None

    def route_to_location_with_preferred_location_flag(self, location_index, use_preferred_locations):
        if False:
            print('Hello World!')
        self.location_index_to_route = location_index
        self.use_preferred_locations = use_preferred_locations
        self.location_endpoint_to_route = None

    def route_to_location(self, location_endpoint):
        if False:
            while True:
                i = 10
        self.location_index_to_route = None
        self.use_preferred_locations = None
        self.location_endpoint_to_route = location_endpoint

    def clear_route_to_location(self):
        if False:
            while True:
                i = 10
        self.location_index_to_route = None
        self.use_preferred_locations = None
        self.location_endpoint_to_route = None