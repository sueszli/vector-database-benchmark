from typing import Optional
from ..BaseModel import BaseModel

class ClusterPrintJobConfigurationChange(BaseModel):
    """Model for the types of changes that are needed before a print job can start"""

    def __init__(self, type_of_change: str, target_id: str, origin_id: str, index: Optional[int]=None, target_name: Optional[str]=None, origin_name: Optional[str]=None, **kwargs) -> None:
        if False:
            return 10
        'Creates a new print job constraint.\n\n        :param type_of_change: The type of configuration change, one of: "material", "print_core_change"\n        :param index: The hotend slot or extruder index to change\n        :param target_id: Target material guid or hotend id\n        :param origin_id: Original/current material guid or hotend id\n        :param target_name: Target material name or hotend id\n        :param origin_name: Original/current material name or hotend id\n        '
        self.type_of_change = type_of_change
        self.index = index
        self.target_id = target_id
        self.origin_id = origin_id
        self.target_name = target_name
        self.origin_name = origin_name
        super().__init__(**kwargs)