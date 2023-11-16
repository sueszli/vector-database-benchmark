from typing import Union, Dict, Any, List
from ..BaseModel import BaseModel
from .ClusterPrinterMaterialStationSlot import ClusterPrinterMaterialStationSlot

class ClusterPrinterMaterialStation(BaseModel):
    """Class representing the data of a Material Station in the cluster."""

    def __init__(self, status: str, supported: bool=False, material_slots: List[Union[ClusterPrinterMaterialStationSlot, Dict[str, Any]]]=None, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        'Creates a new Material Station status.\n\n        :param status: The status of the material station.\n        :param: supported: Whether the material station is supported on this machine or not.\n        :param material_slots: The active slots configurations of this material station.\n        '
        self.status = status
        self.supported = supported
        self.material_slots = self.parseModels(ClusterPrinterMaterialStationSlot, material_slots) if material_slots else []
        super().__init__(**kwargs)