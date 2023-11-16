from typing import Optional
from .ClusterPrintCoreConfiguration import ClusterPrintCoreConfiguration

class ClusterPrinterMaterialStationSlot(ClusterPrintCoreConfiguration):
    """Class representing the data of a single slot in the material station."""

    def __init__(self, slot_index: int, compatible: bool, material_remaining: float, material_empty: Optional[bool]=False, **kwargs) -> None:
        if False:
            print('Hello World!')
        'Create a new material station slot object.\n\n        :param slot_index: The index of the slot in the material station (ranging 0 to 5).\n        :param compatible: Whether the configuration is compatible with the print core.\n        :param material_remaining: How much material is remaining on the spool (between 0 and 1, or -1 for missing data).\n        :param material_empty: Whether the material spool is too empty to be used.\n        '
        self.slot_index = slot_index
        self.compatible = compatible
        self.material_remaining = material_remaining
        self.material_empty = material_empty
        super().__init__(**kwargs)