from typing import Dict, Any
from ..BaseModel import BaseModel

class PrinterSystemStatus(BaseModel):
    """Class representing the system status of a printer."""

    def __init__(self, guid: str, firmware: str, hostname: str, name: str, platform: str, variant: str, hardware: Dict[str, Any], **kwargs) -> None:
        if False:
            print('Hello World!')
        self.guid = guid
        self.firmware = firmware
        self.hostname = hostname
        self.name = name
        self.platform = platform
        self.variant = variant
        self.hardware = hardware
        super().__init__(**kwargs)