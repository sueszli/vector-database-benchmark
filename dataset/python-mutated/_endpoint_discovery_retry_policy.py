"""Internal class for endpoint discovery retry policy implementation in the
Azure Cosmos database service.
"""
import logging
from azure.cosmos.documents import _OperationType
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(levelname)s:%(message)s')
log_handler = logging.StreamHandler()
log_handler.setFormatter(log_formatter)
logger.addHandler(log_handler)

class EndpointDiscoveryRetryPolicy(object):
    """The endpoint discovery retry policy class used for geo-replicated database accounts
       to handle the write forbidden exceptions due to writable/readable location changes
       (say, after a failover).
    """
    Max_retry_attempt_count = 120
    Retry_after_in_milliseconds = 1000

    def __init__(self, connection_policy, global_endpoint_manager, *args):
        if False:
            return 10
        self.global_endpoint_manager = global_endpoint_manager
        self._max_retry_attempt_count = EndpointDiscoveryRetryPolicy.Max_retry_attempt_count
        self.failover_retry_count = 0
        self.retry_after_in_milliseconds = EndpointDiscoveryRetryPolicy.Retry_after_in_milliseconds
        self.connection_policy = connection_policy
        self.request = args[0] if args else None
        if self.request:
            self.request.clear_route_to_location()
            self.location_endpoint = self.global_endpoint_manager.resolve_service_endpoint(self.request)
            self.request.route_to_location(self.location_endpoint)

    def ShouldRetry(self, exception):
        if False:
            for i in range(10):
                print('nop')
        'Returns true if the request should retry based on the passed-in exception.\n\n        :param exceptions.CosmosHttpResponseError exception:\n        :returns: a boolean stating whether the request should be retried\n        :rtype: bool\n        '
        if not self.connection_policy.EnableEndpointDiscovery:
            return False
        if self.failover_retry_count >= self.Max_retry_attempt_count:
            return False
        self.failover_retry_count += 1
        if self.location_endpoint:
            if _OperationType.IsReadOnlyOperation(self.request.operation_type):
                self.global_endpoint_manager.mark_endpoint_unavailable_for_read(self.location_endpoint)
            else:
                self.global_endpoint_manager.mark_endpoint_unavailable_for_write(self.location_endpoint)
        self.global_endpoint_manager.refresh_needed = True
        self.request.clear_route_to_location()
        self.request.route_to_location_with_preferred_location_flag(self.failover_retry_count, False)
        self.location_endpoint = self.global_endpoint_manager.resolve_service_endpoint(self.request)
        self.request.route_to_location(self.location_endpoint)
        return True