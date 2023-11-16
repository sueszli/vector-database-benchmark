"""Implements the abstraction to resolve target location for geo-replicated
DatabaseAccount with multiple writable and readable locations.
"""
import collections
import time
from . import documents
from . import http_constants

class EndpointOperationType(object):
    NoneType = 'None'
    ReadType = 'Read'
    WriteType = 'Write'

def get_endpoint_by_location(locations):
    if False:
        print('Hello World!')
    endpoints_by_location = collections.OrderedDict()
    parsed_locations = []
    for location in locations:
        if not location['name']:
            continue
        try:
            region_uri = location['databaseAccountEndpoint']
            parsed_locations.append(location['name'])
            endpoints_by_location.update({location['name']: region_uri})
        except Exception as e:
            raise e
    return (endpoints_by_location, parsed_locations)

class LocationCache(object):

    def current_time_millis(self):
        if False:
            while True:
                i = 10
        return int(round(time.time() * 1000))

    def __init__(self, preferred_locations, default_endpoint, enable_endpoint_discovery, use_multiple_write_locations, refresh_time_interval_in_ms):
        if False:
            print('Hello World!')
        self.preferred_locations = preferred_locations
        self.default_endpoint = default_endpoint
        self.enable_endpoint_discovery = enable_endpoint_discovery
        self.use_multiple_write_locations = use_multiple_write_locations
        self.enable_multiple_writable_locations = False
        self.write_endpoints = [self.default_endpoint]
        self.read_endpoints = [self.default_endpoint]
        self.location_unavailability_info_by_endpoint = {}
        self.refresh_time_interval_in_ms = refresh_time_interval_in_ms
        self.last_cache_update_time_stamp = 0
        self.available_read_endpoint_by_locations = {}
        self.available_write_endpoint_by_locations = {}
        self.available_write_locations = []
        self.available_read_locations = []

    def check_and_update_cache(self):
        if False:
            print('Hello World!')
        if self.location_unavailability_info_by_endpoint and self.current_time_millis() - self.last_cache_update_time_stamp > self.refresh_time_interval_in_ms:
            self.update_location_cache()

    def get_write_endpoints(self):
        if False:
            return 10
        self.check_and_update_cache()
        return self.write_endpoints

    def get_read_endpoints(self):
        if False:
            return 10
        self.check_and_update_cache()
        return self.read_endpoints

    def get_write_endpoint(self):
        if False:
            for i in range(10):
                print('nop')
        return self.get_write_endpoints()[0]

    def get_read_endpoint(self):
        if False:
            for i in range(10):
                print('nop')
        return self.get_read_endpoints()[0]

    def mark_endpoint_unavailable_for_read(self, endpoint):
        if False:
            return 10
        self.mark_endpoint_unavailable(endpoint, EndpointOperationType.ReadType)

    def mark_endpoint_unavailable_for_write(self, endpoint):
        if False:
            return 10
        self.mark_endpoint_unavailable(endpoint, EndpointOperationType.WriteType)

    def perform_on_database_account_read(self, database_account):
        if False:
            print('Hello World!')
        self.update_location_cache(database_account._WritableLocations, database_account._ReadableLocations, database_account._EnableMultipleWritableLocations)

    def get_ordered_write_endpoints(self):
        if False:
            while True:
                i = 10
        return self.available_write_locations

    def get_ordered_read_endpoints(self):
        if False:
            i = 10
            return i + 15
        return self.available_read_locations

    def resolve_service_endpoint(self, request):
        if False:
            i = 10
            return i + 15
        if request.location_endpoint_to_route:
            return request.location_endpoint_to_route
        location_index = int(request.location_index_to_route) if request.location_index_to_route else 0
        use_preferred_locations = request.use_preferred_locations if request.use_preferred_locations is not None else True
        if not use_preferred_locations or (documents._OperationType.IsWriteOperation(request.operation_type) and (not self.can_use_multiple_write_locations_for_request(request))):
            if self.enable_endpoint_discovery and self.available_write_locations:
                location_index = min(location_index % 2, len(self.available_write_locations) - 1)
                write_location = self.available_write_locations[location_index]
                return self.available_write_endpoint_by_locations[write_location]
            return self.default_endpoint
        endpoints = self.get_write_endpoints() if documents._OperationType.IsWriteOperation(request.operation_type) else self.get_read_endpoints()
        return endpoints[location_index % len(endpoints)]

    def should_refresh_endpoints(self):
        if False:
            i = 10
            return i + 15
        most_preferred_location = self.preferred_locations[0] if self.preferred_locations else None
        if self.enable_endpoint_discovery:
            should_refresh = self.use_multiple_write_locations and (not self.enable_multiple_writable_locations)
            if most_preferred_location:
                if self.available_read_endpoint_by_locations:
                    most_preferred_read_endpoint = self.available_read_endpoint_by_locations[most_preferred_location]
                    if most_preferred_read_endpoint and most_preferred_read_endpoint != self.read_endpoints[0]:
                        return True
                else:
                    return True
            if not self.can_use_multiple_write_locations():
                if self.is_endpoint_unavailable(self.write_endpoints[0], EndpointOperationType.WriteType):
                    return True
                return should_refresh
            if most_preferred_location:
                most_preferred_write_endpoint = self.available_write_endpoint_by_locations[most_preferred_location]
                if most_preferred_write_endpoint:
                    should_refresh |= most_preferred_write_endpoint != self.write_endpoints[0]
                    return should_refresh
                return True
            return should_refresh
        return False

    def clear_stale_endpoint_unavailability_info(self):
        if False:
            while True:
                i = 10
        new_location_unavailability_info = {}
        if self.location_unavailability_info_by_endpoint:
            for unavailable_endpoint in self.location_unavailability_info_by_endpoint:
                unavailability_info = self.location_unavailability_info_by_endpoint[unavailable_endpoint]
                if not (unavailability_info and self.current_time_millis() - unavailability_info['lastUnavailabilityCheckTimeStamp'] > self.refresh_time_interval_in_ms):
                    new_location_unavailability_info[unavailable_endpoint] = self.location_unavailability_info_by_endpoint[unavailable_endpoint]
        self.location_unavailability_info_by_endpoint = new_location_unavailability_info

    def is_endpoint_unavailable(self, endpoint, expected_available_operations):
        if False:
            while True:
                i = 10
        unavailability_info = self.location_unavailability_info_by_endpoint[endpoint] if endpoint in self.location_unavailability_info_by_endpoint else None
        if expected_available_operations == EndpointOperationType.NoneType or not unavailability_info or expected_available_operations not in unavailability_info['operationType']:
            return False
        if self.current_time_millis() - unavailability_info['lastUnavailabilityCheckTimeStamp'] > self.refresh_time_interval_in_ms:
            return False
        return True

    def mark_endpoint_unavailable(self, unavailable_endpoint, unavailable_operation_type):
        if False:
            print('Hello World!')
        unavailability_info = self.location_unavailability_info_by_endpoint[unavailable_endpoint] if unavailable_endpoint in self.location_unavailability_info_by_endpoint else None
        current_time = self.current_time_millis()
        if not unavailability_info:
            self.location_unavailability_info_by_endpoint[unavailable_endpoint] = {'lastUnavailabilityCheckTimeStamp': current_time, 'operationType': set([unavailable_operation_type])}
        else:
            unavailable_operations = set([unavailable_operation_type]).union(unavailability_info['operationType'])
            self.location_unavailability_info_by_endpoint[unavailable_endpoint] = {'lastUnavailabilityCheckTimeStamp': current_time, 'operationType': unavailable_operations}
        self.update_location_cache()

    def get_preferred_locations(self):
        if False:
            print('Hello World!')
        return self.preferred_locations

    def update_location_cache(self, write_locations=None, read_locations=None, enable_multiple_writable_locations=None):
        if False:
            print('Hello World!')
        if enable_multiple_writable_locations:
            self.enable_multiple_writable_locations = enable_multiple_writable_locations
        self.clear_stale_endpoint_unavailability_info()
        if self.enable_endpoint_discovery:
            if read_locations:
                (self.available_read_endpoint_by_locations, self.available_read_locations) = get_endpoint_by_location(read_locations)
            if write_locations:
                (self.available_write_endpoint_by_locations, self.available_write_locations) = get_endpoint_by_location(write_locations)
        self.write_endpoints = self.get_preferred_available_endpoints(self.available_write_endpoint_by_locations, self.available_write_locations, EndpointOperationType.WriteType, self.default_endpoint)
        self.read_endpoints = self.get_preferred_available_endpoints(self.available_read_endpoint_by_locations, self.available_read_locations, EndpointOperationType.ReadType, self.write_endpoints[0])
        self.last_cache_update_timestamp = self.current_time_millis()

    def get_preferred_available_endpoints(self, endpoints_by_location, orderedLocations, expected_available_operation, fallback_endpoint):
        if False:
            print('Hello World!')
        endpoints = []
        if self.enable_endpoint_discovery and endpoints_by_location:
            if self.can_use_multiple_write_locations() or expected_available_operation == EndpointOperationType.ReadType:
                unavailable_endpoints = []
                if self.preferred_locations:
                    for location in self.preferred_locations:
                        endpoint = endpoints_by_location[location] if location in endpoints_by_location else None
                        if endpoint:
                            if self.is_endpoint_unavailable(endpoint, expected_available_operation):
                                unavailable_endpoints.append(endpoint)
                            else:
                                endpoints.append(endpoint)
                if not endpoints:
                    endpoints.append(fallback_endpoint)
                endpoints.extend(unavailable_endpoints)
            else:
                for location in orderedLocations:
                    if location and location in endpoints_by_location:
                        endpoints.append(endpoints_by_location[location])
        if not endpoints:
            endpoints.append(fallback_endpoint)
        return endpoints

    def can_use_multiple_write_locations(self):
        if False:
            i = 10
            return i + 15
        return self.use_multiple_write_locations and self.enable_multiple_writable_locations

    def can_use_multiple_write_locations_for_request(self, request):
        if False:
            i = 10
            return i + 15
        return self.can_use_multiple_write_locations() and (request.resource_type == http_constants.ResourceType.Document or (request.resource_type == http_constants.ResourceType.StoredProcedure and request.operation_type == documents._OperationType.ExecuteJavaScript))