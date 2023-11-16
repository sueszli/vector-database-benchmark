from typing import Optional, List
from .CloudClusterResponse import CloudClusterResponse
from .ClusterPrinterStatus import ClusterPrinterStatus

class CloudClusterWithConfigResponse(CloudClusterResponse):
    """Class representing a cloud connected cluster."""

    def __init__(self, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.configuration = self.parseModel(ClusterPrinterStatus, kwargs.get('host_printer'))
        super().__init__(**kwargs)