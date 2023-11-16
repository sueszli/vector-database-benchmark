from typing import List, Optional, Union, Dict, Any
from cura.PrinterOutput.Models.PrinterConfigurationModel import PrinterConfigurationModel
from .ClusterBuildPlate import ClusterBuildPlate
from .ClusterPrintJobConfigurationChange import ClusterPrintJobConfigurationChange
from .ClusterPrintJobImpediment import ClusterPrintJobImpediment
from .ClusterPrintCoreConfiguration import ClusterPrintCoreConfiguration
from .ClusterPrintJobConstraints import ClusterPrintJobConstraints
from ..UM3PrintJobOutputModel import UM3PrintJobOutputModel
from ..ConfigurationChangeModel import ConfigurationChangeModel
from ..BaseModel import BaseModel
from ...ClusterOutputController import ClusterOutputController

class ClusterPrintJobStatus(BaseModel):
    """Model for the status of a single print job in a cluster."""

    def __init__(self, created_at: str, force: bool, machine_variant: str, name: str, started: bool, status: str, time_total: int, uuid: str, configuration: List[Union[Dict[str, Any], ClusterPrintCoreConfiguration]], constraints: Optional[Union[Dict[str, Any], ClusterPrintJobConstraints]]=None, last_seen: Optional[float]=None, network_error_count: Optional[int]=None, owner: Optional[str]=None, printer_uuid: Optional[str]=None, time_elapsed: Optional[int]=None, assigned_to: Optional[str]=None, deleted_at: Optional[str]=None, printed_on_uuid: Optional[str]=None, configuration_changes_required: List[Union[Dict[str, Any], ClusterPrintJobConfigurationChange]]=None, build_plate: Union[Dict[str, Any], ClusterBuildPlate]=None, compatible_machine_families: Optional[List[str]]=None, impediments_to_printing: List[Union[Dict[str, Any], ClusterPrintJobImpediment]]=None, preview_url: Optional[str]=None, **kwargs) -> None:
        if False:
            while True:
                i = 10
        "Creates a new cloud print job status model.\n\n        :param assigned_to: The name of the printer this job is assigned to while being queued.\n        :param configuration: The required print core configurations of this print job.\n        :param constraints: Print job constraints object.\n        :param created_at: The timestamp when the job was created in Cura Connect.\n        :param force: Allow this job to be printed despite of mismatching configurations.\n        :param last_seen: The number of seconds since this job was checked.\n        :param machine_variant: The machine type that this job should be printed on.Coincides with the machine_type field\n        of the printer object.\n        :param name: The name of the print job. Usually the name of the .gcode file.\n        :param network_error_count: The number of errors encountered when requesting data for this print job.\n        :param owner: The name of the user who added the print job to Cura Connect.\n        :param printer_uuid: UUID of the printer that the job is currently printing on or assigned to.\n        :param started: Whether the job has started printing or not.\n        :param status: The status of the print job.\n        :param time_elapsed: The remaining printing time in seconds.\n        :param time_total: The total printing time in seconds.\n        :param uuid: UUID of this print job. Should be used for identification purposes.\n        :param deleted_at: The time when this print job was deleted.\n        :param printed_on_uuid: UUID of the printer used to print this job.\n        :param configuration_changes_required: List of configuration changes the printer this job is associated with\n        needs to make in order to be able to print this job\n        :param build_plate: The build plate (type) this job needs to be printed on.\n        :param compatible_machine_families: Family names of machines suitable for this print job\n        :param impediments_to_printing: A list of reasons that prevent this job from being printed on the associated\n        printer\n        :param preview_url: URL to the preview image (same as wou;d've been included in the ufp).\n        "
        self.assigned_to = assigned_to
        self.configuration = self.parseModels(ClusterPrintCoreConfiguration, configuration)
        self.constraints = self.parseModel(ClusterPrintJobConstraints, constraints) if constraints else None
        self.created_at = created_at
        self.force = force
        self.last_seen = last_seen
        self.machine_variant = machine_variant
        self.name = name
        self.network_error_count = network_error_count
        self.owner = owner
        self.printer_uuid = printer_uuid
        self.started = started
        self.status = status
        self.time_elapsed = time_elapsed
        self.time_total = time_total
        self.uuid = uuid
        self.deleted_at = deleted_at
        self.printed_on_uuid = printed_on_uuid
        self.preview_url = preview_url
        self.configuration_changes_required = self.parseModels(ClusterPrintJobConfigurationChange, configuration_changes_required) if configuration_changes_required else []
        self.build_plate = self.parseModel(ClusterBuildPlate, build_plate) if build_plate else None
        self.compatible_machine_families = compatible_machine_families if compatible_machine_families is not None else []
        self.impediments_to_printing = self.parseModels(ClusterPrintJobImpediment, impediments_to_printing) if impediments_to_printing else []
        super().__init__(**kwargs)

    def createOutputModel(self, controller: ClusterOutputController) -> UM3PrintJobOutputModel:
        if False:
            for i in range(10):
                print('nop')
        'Creates an UM3 print job output model based on this cloud cluster print job.\n\n        :param printer: The output model of the printer\n        '
        model = UM3PrintJobOutputModel(controller, self.uuid, self.name)
        self.updateOutputModel(model)
        return model

    def _createConfigurationModel(self) -> PrinterConfigurationModel:
        if False:
            return 10
        'Creates a new configuration model'
        extruders = [extruder.createConfigurationModel() for extruder in self.configuration or ()]
        configuration = PrinterConfigurationModel()
        configuration.setExtruderConfigurations(extruders)
        configuration.setPrinterType(self.machine_variant)
        return configuration

    def updateOutputModel(self, model: UM3PrintJobOutputModel) -> None:
        if False:
            while True:
                i = 10
        'Updates an UM3 print job output model based on this cloud cluster print job.\n\n        :param model: The model to update.\n        '
        model.updateConfiguration(self._createConfigurationModel())
        model.updateTimeTotal(self.time_total)
        if self.time_elapsed is not None:
            model.updateTimeElapsed(self.time_elapsed)
        if self.owner is not None:
            model.updateOwner(self.owner)
        model.updateState(self.status)
        model.setCompatibleMachineFamilies(self.compatible_machine_families)
        status_set_by_impediment = False
        for impediment in self.impediments_to_printing:
            if impediment.severity == 'UNFIXABLE':
                status_set_by_impediment = True
                model.updateState('error')
                break
        if not status_set_by_impediment:
            model.updateState(self.status)
        model.updateConfigurationChanges([ConfigurationChangeModel(type_of_change=change.type_of_change, index=change.index if change.index else 0, target_name=change.target_name if change.target_name else '', origin_name=change.origin_name if change.origin_name else '') for change in self.configuration_changes_required])