import json
import math
import os
from typing import Dict, List, Optional, TYPE_CHECKING
from PyQt6.QtCore import QObject, pyqtSignal, pyqtProperty, pyqtSlot, QTimer
from UM.Logger import Logger
from UM.Qt.Duration import Duration
from UM.Scene.SceneNode import SceneNode
from UM.i18n import i18nCatalog
from UM.MimeTypeDatabase import MimeTypeDatabase, MimeTypeNotFoundError
from UM.OutputDevice.OutputDevice import OutputDevice
from UM.OutputDevice.ProjectOutputDevice import ProjectOutputDevice
if TYPE_CHECKING:
    from cura.CuraApplication import CuraApplication
catalog = i18nCatalog('cura')

class PrintInformation(QObject):
    """A class for processing the print times per build plate and managing the job name

    This class also combines the current machine name and the filename of the first loaded mesh into a job name.
    This job name is requested by the JobSpecs qml file.
    """
    UNTITLED_JOB_NAME = 'Untitled'

    def __init__(self, application: 'CuraApplication', parent=None) -> None:
        if False:
            print('Hello World!')
        super().__init__(parent)
        self._application = application
        self.initializeCuraMessagePrintTimeProperties()
        self.slice_uuid: Optional[str] = None
        self._material_lengths = {}
        self._material_weights = {}
        self._material_costs = {}
        self._material_names = {}
        self._pre_sliced = False
        self._backend = self._application.getBackend()
        if self._backend:
            self._backend.printDurationMessage.connect(self._onPrintDurationMessage)
        self._application.getController().getScene().sceneChanged.connect(self._onSceneChangedDelayed)
        self._change_timer = QTimer()
        self._change_timer.setInterval(100)
        self._change_timer.setSingleShot(True)
        self._change_timer.timeout.connect(self._onSceneChanged)
        self._is_user_specified_job_name = False
        self._base_name = ''
        self._abbr_machine = ''
        self._job_name = ''
        self._active_build_plate = 0
        self._initVariablesByBuildPlate(self._active_build_plate)
        self._multi_build_plate_model = self._application.getMultiBuildPlateModel()
        self._application.globalContainerStackChanged.connect(self._updateJobName)
        self._application.globalContainerStackChanged.connect(self.setToZeroPrintInformation)
        self._application.fileLoaded.connect(self.setBaseName)
        self._application.workspaceLoaded.connect(self.setProjectName)
        self._application.getOutputDeviceManager().writeStarted.connect(self._onOutputStart)
        self._application.getMachineManager().rootMaterialChanged.connect(self._onActiveMaterialsChanged)
        self._application.getInstance().getPreferences().preferenceChanged.connect(self._onPreferencesChanged)
        self._multi_build_plate_model.activeBuildPlateChanged.connect(self._onActiveBuildPlateChanged)
        self._material_amounts = []
        self._onActiveMaterialsChanged()

    def initializeCuraMessagePrintTimeProperties(self) -> None:
        if False:
            i = 10
            return i + 15
        self._current_print_time = {}
        self._print_time_message_translations = {'inset_0': catalog.i18nc('@tooltip', 'Outer Wall'), 'inset_x': catalog.i18nc('@tooltip', 'Inner Walls'), 'skin': catalog.i18nc('@tooltip', 'Skin'), 'infill': catalog.i18nc('@tooltip', 'Infill'), 'support_infill': catalog.i18nc('@tooltip', 'Support Infill'), 'support_interface': catalog.i18nc('@tooltip', 'Support Interface'), 'support': catalog.i18nc('@tooltip', 'Support'), 'skirt': catalog.i18nc('@tooltip', 'Skirt'), 'prime_tower': catalog.i18nc('@tooltip', 'Prime Tower'), 'travel': catalog.i18nc('@tooltip', 'Travel'), 'retract': catalog.i18nc('@tooltip', 'Retractions'), 'none': catalog.i18nc('@tooltip', 'Other')}
        self._print_times_per_feature = {}

    def _initPrintTimesPerFeature(self, build_plate_number: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._print_times_per_feature[build_plate_number] = {}
        for key in self._print_time_message_translations.keys():
            self._print_times_per_feature[build_plate_number][key] = Duration(None, self)

    def _initVariablesByBuildPlate(self, build_plate_number: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        if build_plate_number not in self._print_times_per_feature:
            self._initPrintTimesPerFeature(build_plate_number)
        if self._active_build_plate not in self._material_lengths:
            self._material_lengths[self._active_build_plate] = []
        if self._active_build_plate not in self._material_weights:
            self._material_weights[self._active_build_plate] = []
        if self._active_build_plate not in self._material_costs:
            self._material_costs[self._active_build_plate] = []
        if self._active_build_plate not in self._material_names:
            self._material_names[self._active_build_plate] = []
        if self._active_build_plate not in self._current_print_time:
            self._current_print_time[self._active_build_plate] = Duration(parent=self)
    currentPrintTimeChanged = pyqtSignal()
    preSlicedChanged = pyqtSignal()

    @pyqtProperty(bool, notify=preSlicedChanged)
    def preSliced(self) -> bool:
        if False:
            return 10
        return self._pre_sliced

    def setPreSliced(self, pre_sliced: bool) -> None:
        if False:
            return 10
        if self._pre_sliced != pre_sliced:
            self._pre_sliced = pre_sliced
            self._updateJobName()
            self.preSlicedChanged.emit()

    @pyqtProperty(QObject, notify=currentPrintTimeChanged)
    def currentPrintTime(self) -> Duration:
        if False:
            return 10
        return self._current_print_time[self._active_build_plate]
    materialLengthsChanged = pyqtSignal()

    @pyqtProperty('QVariantList', notify=materialLengthsChanged)
    def materialLengths(self):
        if False:
            for i in range(10):
                print('nop')
        return self._material_lengths[self._active_build_plate]
    materialWeightsChanged = pyqtSignal()

    @pyqtProperty('QVariantList', notify=materialWeightsChanged)
    def materialWeights(self):
        if False:
            while True:
                i = 10
        return self._material_weights[self._active_build_plate]
    materialCostsChanged = pyqtSignal()

    @pyqtProperty('QVariantList', notify=materialCostsChanged)
    def materialCosts(self):
        if False:
            print('Hello World!')
        return self._material_costs[self._active_build_plate]
    materialNamesChanged = pyqtSignal()

    @pyqtProperty('QVariantList', notify=materialNamesChanged)
    def materialNames(self):
        if False:
            while True:
                i = 10
        return self._material_names[self._active_build_plate]

    def printTimes(self) -> Dict[str, Duration]:
        if False:
            i = 10
            return i + 15
        return self._print_times_per_feature[self._active_build_plate]

    def _onPrintDurationMessage(self, build_plate_number: int, print_times_per_feature: Dict[str, int], material_amounts: List[float]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._updateTotalPrintTimePerFeature(build_plate_number, print_times_per_feature)
        self.currentPrintTimeChanged.emit()
        self._material_amounts = material_amounts
        self._calculateInformation(build_plate_number)

    def _updateTotalPrintTimePerFeature(self, build_plate_number: int, print_times_per_feature: Dict[str, int]) -> None:
        if False:
            while True:
                i = 10
        total_estimated_time = 0
        if build_plate_number not in self._print_times_per_feature:
            self._initPrintTimesPerFeature(build_plate_number)
        for (feature, time) in print_times_per_feature.items():
            if feature not in self._print_times_per_feature[build_plate_number]:
                self._print_times_per_feature[build_plate_number][feature] = Duration(parent=self)
            duration = self._print_times_per_feature[build_plate_number][feature]
            if time != time:
                duration.setDuration(0)
                Logger.warning('Received NaN for print duration message')
                continue
            total_estimated_time += time
            duration.setDuration(time)
        if build_plate_number not in self._current_print_time:
            self._current_print_time[build_plate_number] = Duration(None, self)
        self._current_print_time[build_plate_number].setDuration(total_estimated_time)

    def _calculateInformation(self, build_plate_number: int) -> None:
        if False:
            print('Hello World!')
        global_stack = self._application.getGlobalContainerStack()
        if global_stack is None:
            return
        self._material_lengths[build_plate_number] = []
        self._material_weights[build_plate_number] = []
        self._material_costs[build_plate_number] = []
        self._material_names[build_plate_number] = []
        try:
            material_preference_values = json.loads(self._application.getInstance().getPreferences().getValue('cura/material_settings'))
        except json.JSONDecodeError:
            Logger.warning('Material preference values are corrupt. Will revert to defaults!')
            material_preference_values = {}
        for (index, extruder_stack) in enumerate(global_stack.extruderList):
            if index >= len(self._material_amounts):
                continue
            amount = self._material_amounts[index]
            density = extruder_stack.getMetaDataEntry('properties', {}).get('density', 0)
            material = extruder_stack.material
            radius = extruder_stack.getProperty('material_diameter', 'value') / 2
            weight = float(amount) * float(density) / 1000
            cost = 0.0
            material_guid = material.getMetaDataEntry('GUID')
            material_name = material.getName()
            if material_guid in material_preference_values:
                material_values = material_preference_values[material_guid]
                if material_values and 'spool_weight' in material_values:
                    weight_per_spool = float(material_values['spool_weight'])
                else:
                    weight_per_spool = float(extruder_stack.getMetaDataEntry('properties', {}).get('weight', 0))
                cost_per_spool = float(material_values['spool_cost'] if material_values and 'spool_cost' in material_values else 0)
                if weight_per_spool != 0:
                    cost = cost_per_spool * weight / weight_per_spool
                else:
                    cost = 0
            if radius != 0:
                length = round(amount / (math.pi * radius ** 2) / 1000, 2)
            else:
                length = 0
            self._material_weights[build_plate_number].append(weight)
            self._material_lengths[build_plate_number].append(length)
            self._material_costs[build_plate_number].append(cost)
            self._material_names[build_plate_number].append(material_name)
        self.materialLengthsChanged.emit()
        self.materialWeightsChanged.emit()
        self.materialCostsChanged.emit()
        self.materialNamesChanged.emit()

    def _onPreferencesChanged(self, preference: str) -> None:
        if False:
            return 10
        if preference != 'cura/material_settings':
            return
        for build_plate_number in range(self._multi_build_plate_model.maxBuildPlate + 1):
            self._calculateInformation(build_plate_number)

    def _onActiveBuildPlateChanged(self) -> None:
        if False:
            i = 10
            return i + 15
        new_active_build_plate = self._multi_build_plate_model.activeBuildPlate
        if new_active_build_plate != self._active_build_plate:
            self._active_build_plate = new_active_build_plate
            self._updateJobName()
            self._initVariablesByBuildPlate(self._active_build_plate)
            self.materialLengthsChanged.emit()
            self.materialWeightsChanged.emit()
            self.materialCostsChanged.emit()
            self.materialNamesChanged.emit()
            self.currentPrintTimeChanged.emit()

    def _onActiveMaterialsChanged(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        for build_plate_number in range(self._multi_build_plate_model.maxBuildPlate + 1):
            self._calculateInformation(build_plate_number)

    @pyqtSlot(str, bool)
    def setJobName(self, name: str, is_user_specified_job_name=False) -> None:
        if False:
            i = 10
            return i + 15
        self._is_user_specified_job_name = is_user_specified_job_name
        self._job_name = name
        self._base_name = name.replace(self._abbr_machine + '_', '')
        if name == '':
            self._is_user_specified_job_name = False
        self.jobNameChanged.emit()
    jobNameChanged = pyqtSignal()

    @pyqtProperty(str, notify=jobNameChanged)
    def jobName(self):
        if False:
            return 10
        return self._job_name

    def _updateJobName(self) -> None:
        if False:
            while True:
                i = 10
        if self._base_name == '':
            self._job_name = self.UNTITLED_JOB_NAME
            self._is_user_specified_job_name = False
            self._application.getController().getScene().clearMetaData()
            self.jobNameChanged.emit()
            return
        base_name = self._base_name
        self._defineAbbreviatedMachineName()
        if not self._is_user_specified_job_name:
            if self._application.getInstance().getPreferences().getValue('cura/jobname_prefix') and (not self._pre_sliced):
                if base_name.startswith(self._abbr_machine + '_'):
                    self._job_name = base_name
                else:
                    self._job_name = self._abbr_machine + '_' + base_name
            else:
                self._job_name = base_name
        if self._multi_build_plate_model.maxBuildPlate > 0:
            connector = '_#'
            suffix = connector + str(self._active_build_plate + 1)
            if connector in self._job_name:
                self._job_name = self._job_name.split(connector)[0]
            if self._active_build_plate != 0:
                self._job_name += suffix
        self.jobNameChanged.emit()

    @pyqtSlot(str)
    def setProjectName(self, name: str) -> None:
        if False:
            return 10
        self.setBaseName(name, is_project_file=True)
    baseNameChanged = pyqtSignal()

    def setBaseName(self, base_name: str, is_project_file: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._is_user_specified_job_name = False
        name = os.path.basename(base_name)
        check_name = os.path.splitext(name)[0]
        filename_parts = os.path.basename(base_name).split('.')
        is_gcode = False
        if len(filename_parts) > 1:
            is_gcode = 'gcode' in filename_parts[1:]
        is_empty = check_name == ''
        if is_gcode or is_project_file or (is_empty or (self._base_name == '' and self._base_name != check_name)):
            data = ''
            try:
                mime_type = MimeTypeDatabase.getMimeTypeForFile(name)
                data = mime_type.stripExtension(name)
            except MimeTypeNotFoundError:
                Logger.warning(f'Unsupported Mime Type Database file extension {name}')
            if data is not None and check_name is not None:
                self._base_name = data
            else:
                self._base_name = ''
            OLD_CURA_PROJECT_EXT = '.curaproject'
            if self._base_name.lower().endswith(OLD_CURA_PROJECT_EXT):
                self._base_name = self._base_name[:len(self._base_name) - len(OLD_CURA_PROJECT_EXT)]
            OLD_CURA_PROJECT_3MF_EXT = '.curaproject.3mf'
            while self._base_name.lower().endswith(OLD_CURA_PROJECT_3MF_EXT):
                self._base_name = self._base_name[:len(self._base_name) - len(OLD_CURA_PROJECT_3MF_EXT)]
            self._updateJobName()

    @pyqtProperty(str, fset=setBaseName, notify=baseNameChanged)
    def baseName(self):
        if False:
            for i in range(10):
                print('nop')
        return self._base_name

    def _defineAbbreviatedMachineName(self) -> None:
        if False:
            return 10
        'Creates an abbreviated machine name from the currently active machine name.\n\n        Called each time the global stack is switched.\n        '
        global_container_stack = self._application.getGlobalContainerStack()
        if not global_container_stack:
            self._abbr_machine = ''
            return
        active_machine_type_name = global_container_stack.definition.getName()
        self._abbr_machine = self._application.getMachineManager().getAbbreviatedMachineName(active_machine_type_name)

    @pyqtSlot(result='QVariantMap')
    def getFeaturePrintTimes(self) -> Dict[str, Duration]:
        if False:
            print('Hello World!')
        result = {}
        if self._active_build_plate not in self._print_times_per_feature:
            self._initPrintTimesPerFeature(self._active_build_plate)
        for (feature, time) in self._print_times_per_feature[self._active_build_plate].items():
            if feature in self._print_time_message_translations:
                result[self._print_time_message_translations[feature]] = time
            else:
                result[feature] = time
        return result

    def setToZeroPrintInformation(self, build_plate: Optional[int]=None) -> None:
        if False:
            return 10
        if build_plate is None:
            build_plate = self._active_build_plate
        temp_message = {}
        if build_plate not in self._print_times_per_feature:
            self._print_times_per_feature[build_plate] = {}
        for key in self._print_times_per_feature[build_plate].keys():
            temp_message[key] = 0
        temp_material_amounts = [0.0]
        self._onPrintDurationMessage(build_plate, temp_message, temp_material_amounts)

    def _onSceneChangedDelayed(self, scene_node: SceneNode) -> None:
        if False:
            while True:
                i = 10
        if not isinstance(scene_node, SceneNode) or not scene_node.callDecoration('isSliceable') or (not scene_node.callDecoration('getBuildPlateNumber') == self._active_build_plate):
            return
        self._change_timer.start()

    def _onSceneChanged(self) -> None:
        if False:
            return 10
        'Listen to scene changes to check if we need to reset the print information'
        self.setToZeroPrintInformation(self._active_build_plate)

    def _onOutputStart(self, output_device: OutputDevice) -> None:
        if False:
            i = 10
            return i + 15
        "If this is a sort of output 'device' (like local or online file storage, rather than a printer),\n           the user could have altered the file-name, and thus the project name should be altered as well."
        if isinstance(output_device, ProjectOutputDevice):
            new_name = output_device.getLastOutputName()
            if new_name is not None:
                self.setJobName(os.path.splitext(os.path.basename(new_name))[0])