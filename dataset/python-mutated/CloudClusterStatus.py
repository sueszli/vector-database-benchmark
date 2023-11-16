from datetime import datetime
from typing import List, Dict, Union, Any
from ..BaseModel import BaseModel
from .ClusterPrinterStatus import ClusterPrinterStatus
from .ClusterPrintJobStatus import ClusterPrintJobStatus

class CloudClusterStatus(BaseModel):

    def __init__(self, printers: List[Union[ClusterPrinterStatus, Dict[str, Any]]], print_jobs: List[Union[ClusterPrintJobStatus, Dict[str, Any]]], generated_time: Union[str, datetime], **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        'Creates a new cluster status model object.\n\n        :param printers: The latest status of each printer in the cluster.\n        :param print_jobs: The latest status of each print job in the cluster.\n        :param generated_time: The datetime when the object was generated on the server-side.\n        '
        self.generated_time = self.parseDate(generated_time)
        self.printers = self.parseModels(ClusterPrinterStatus, printers)
        self.print_jobs = self.parseModels(ClusterPrintJobStatus, print_jobs)
        super().__init__(**kwargs)