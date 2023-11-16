from typing import Union, Dict, Optional, Any
from cura.PrinterOutput.Models.ExtruderConfigurationModel import ExtruderConfigurationModel
from cura.PrinterOutput.Models.ExtruderOutputModel import ExtruderOutputModel
from .ClusterPrinterConfigurationMaterial import ClusterPrinterConfigurationMaterial
from ..BaseModel import BaseModel

class ClusterPrintCoreConfiguration(BaseModel):
    """Class representing a cloud cluster printer configuration

    Also used for representing slots in a Material Station (as from Cura's perspective these are the same).
    """

    def __init__(self, extruder_index: int, material: Union[None, Dict[str, Any], ClusterPrinterConfigurationMaterial]=None, print_core_id: Optional[str]=None, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Creates a new cloud cluster printer configuration object\n\n        :param extruder_index: The position of the extruder on the machine as list index. Numbered from left to right.\n        :param material: The material of a configuration object in a cluster printer. May be in a dict or an object.\n        :param nozzle_diameter: The diameter of the print core at this position in millimeters, e.g. '0.4'.\n        :param print_core_id: The type of print core inserted at this position, e.g. 'AA 0.4'.\n        "
        self.extruder_index = extruder_index
        self.material = self.parseModel(ClusterPrinterConfigurationMaterial, material) if material else None
        self.print_core_id = print_core_id
        super().__init__(**kwargs)

    def updateOutputModel(self, model: ExtruderOutputModel) -> None:
        if False:
            while True:
                i = 10
        'Updates the given output model.\n\n        :param model: The output model to update.\n        '
        if self.print_core_id is not None:
            model.updateHotendID(self.print_core_id)
        if self.material:
            active_material = model.activeMaterial
            if active_material is None or active_material.guid != self.material.guid:
                material = self.material.createOutputModel()
                model.updateActiveMaterial(material)
        else:
            model.updateActiveMaterial(None)

    def createConfigurationModel(self) -> ExtruderConfigurationModel:
        if False:
            print('Hello World!')
        'Creates a configuration model'
        model = ExtruderConfigurationModel(position=self.extruder_index)
        self.updateConfigurationModel(model)
        return model

    def updateConfigurationModel(self, model: ExtruderConfigurationModel) -> ExtruderConfigurationModel:
        if False:
            for i in range(10):
                print('nop')
        'Creates a configuration model'
        model.setHotendID(self.print_core_id)
        if self.material:
            model.setMaterial(self.material.createOutputModel())
        return model