from typing import Optional
from ..BaseModel import BaseModel

class ClusterPrintJobConstraints(BaseModel):
    """Class representing a cloud cluster print job constraint"""

    def __init__(self, require_printer_name: Optional[str]=None, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        "Creates a new print job constraint.\n\n        :param require_printer_name: Unique name of the printer that this job should be printed on.\n        Should be one of the unique_name field values in the cluster, e.g. 'ultimakersystem-ccbdd30044ec'\n        "
        self.require_printer_name = require_printer_name
        super().__init__(**kwargs)