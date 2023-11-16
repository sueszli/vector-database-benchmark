"""Create throughput properties in the Azure Cosmos DB SQL API service.
"""

class ThroughputProperties(object):
    """Represents the throughput properties in an Azure Cosmos DB SQL API container.

    To read and update throughput properties, use the associated methods on the :class:`Container`.
    If configuring auto-scale, `auto_scale_max_throughput` needs to be set and
    `auto_scale_increment_percent` can also be set in conjunction with it.
    The value of `offer_throughput` will not be allowed to be set in conjunction with the auto-scale settings.

    :keyword int offer_throughput: The provisioned throughput in request units per second as a number.
    :keyword int auto_scale_max_throughput: The max auto-scale throughput. It should have a valid throughput
     value between 1000 and 1000000 inclusive, in increments of 1000.
    :keyword int auto_scale_increment_percent: is the % from the base selected RU it increases at a given time,
     the increment percent should be greater than or equal to zero.
    """

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.offer_throughput = args[0] if args else kwargs.get('offer_throughput')
        self.properties = args[1] if len(args) > 1 else kwargs.get('properties')
        self.auto_scale_max_throughput = kwargs.get('auto_scale_max_throughput')
        self.auto_scale_increment_percent = kwargs.get('auto_scale_increment_percent')
Offer = ThroughputProperties