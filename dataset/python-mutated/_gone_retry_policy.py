"""Internal class for connection reset retry policy implementation in the Azure
Cosmos database service.
"""

class PartitionKeyRangeGoneRetryPolicy(object):

    def __init__(self, client, *args):
        if False:
            for i in range(10):
                print('nop')
        self.retry_after_in_milliseconds = 1000
        self.refresh_partition_key_range_cache = True
        self.args = args
        self.client = client
        self.exception = None

    def ShouldRetry(self, exception):
        if False:
            print('Hello World!')
        'Returns true if the request should retry based on the passed-in exception.\n\n        :param (exceptions.CosmosHttpResponseError instance) exception:\n        :returns: a boolean stating whether the request should be retried\n        :rtype: bool\n\n        '
        self.exception = exception
        if self.refresh_partition_key_range_cache:
            self.client.refresh_routing_map_provider()
            self.refresh_partition_key_range_cache = False
        return False