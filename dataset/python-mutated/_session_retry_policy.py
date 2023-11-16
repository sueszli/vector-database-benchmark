"""Internal class for session read/write unavailable retry policy implementation
in the Azure Cosmos database service.
"""
import logging
from azure.cosmos.documents import _OperationType
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(levelname)s:%(message)s')
log_handler = logging.StreamHandler()
log_handler.setFormatter(log_formatter)
logger.addHandler(log_handler)

class _SessionRetryPolicy(object):
    """The session retry policy used to handle read/write session unavailability.
    """
    Max_retry_attempt_count = 1
    Retry_after_in_milliseconds = 0

    def __init__(self, endpoint_discovery_enable, global_endpoint_manager, *args):
        if False:
            while True:
                i = 10
        self.global_endpoint_manager = global_endpoint_manager
        self._max_retry_attempt_count = _SessionRetryPolicy.Max_retry_attempt_count
        self.session_token_retry_count = 0
        self.retry_after_in_milliseconds = _SessionRetryPolicy.Retry_after_in_milliseconds
        self.endpoint_discovery_enable = endpoint_discovery_enable
        self.request = args[0] if args else None
        if self.request:
            self.can_use_multiple_write_locations = self.global_endpoint_manager.can_use_multiple_write_locations(self.request)
            self.request.clear_route_to_location()
            self.location_endpoint = self.global_endpoint_manager.resolve_service_endpoint(self.request)
            self.request.route_to_location(self.location_endpoint)

    def ShouldRetry(self, _exception):
        if False:
            return 10
        'Returns true if the request should retry based on the passed-in exception.\n\n        :param exceptions.CosmosHttpResponseError _exception:\n        :returns: a boolean stating whether the request should be retried\n        :rtype: bool\n        '
        self.session_token_retry_count += 1
        self.request.clear_route_to_location()
        if not self.endpoint_discovery_enable:
            return False
        if self.can_use_multiple_write_locations:
            if _OperationType.IsReadOnlyOperation(self.request.operation_type):
                endpoints = self.global_endpoint_manager.get_ordered_read_endpoints()
            else:
                endpoints = self.global_endpoint_manager.get_ordered_write_endpoints()
            if self.session_token_retry_count > len(endpoints):
                return False
            self.request.route_to_location_with_preferred_location_flag(self.session_token_retry_count - 1, self.session_token_retry_count > self._max_retry_attempt_count)
            self.request.should_clear_session_token_on_session_read_failure = self.session_token_retry_count == len(endpoints)
            self.location_endpoint = self.global_endpoint_manager.resolve_service_endpoint(self.request)
            self.request.route_to_location(self.location_endpoint)
            return True
        if self.session_token_retry_count > self._max_retry_attempt_count:
            return False
        self.request.route_to_location_with_preferred_location_flag(self.session_token_retry_count - 1, False)
        self.request.should_clear_session_token_on_session_read_failure = True
        self.location_endpoint = self.global_endpoint_manager.resolve_service_endpoint(self.request)
        self.request.route_to_location(self.location_endpoint)
        return True