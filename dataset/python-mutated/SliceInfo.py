import json
import os
import platform
import time
from typing import cast, Optional, Set, TYPE_CHECKING
from PyQt6.QtCore import pyqtSlot, QObject
from PyQt6.QtNetwork import QNetworkRequest
from UM.Extension import Extension
from UM.Scene.Iterator.DepthFirstIterator import DepthFirstIterator
from UM.i18n import i18nCatalog
from UM.Logger import Logger
from UM.PluginRegistry import PluginRegistry
from UM.Qt.Duration import DurationFormat
from cura import ApplicationMetadata
if TYPE_CHECKING:
    from PyQt6.QtNetwork import QNetworkReply
catalog = i18nCatalog('cura')

class SliceInfo(QObject, Extension):
    """This Extension runs in the background and sends several bits of information to the UltiMaker servers.

    The data is only sent when the user in question gave permission to do so. All data is anonymous and
    no model files are being sent (Just a SHA256 hash of the model).
    """
    info_url = 'https://stats.ultimaker.com/api/cura'

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        QObject.__init__(self, parent)
        Extension.__init__(self)
        from cura.CuraApplication import CuraApplication
        self._application = CuraApplication.getInstance()
        self._application.getOutputDeviceManager().writeStarted.connect(self._onWriteStarted)
        self._application.getPreferences().addPreference('info/send_slice_info', True)
        self._application.getPreferences().addPreference('info/asked_send_slice_info', False)
        self._more_info_dialog = None
        self._example_data_content = None
        self._application.initializationFinished.connect(self._onAppInitialized)

    def _onAppInitialized(self):
        if False:
            while True:
                i = 10
        if self._more_info_dialog is None:
            self._more_info_dialog = self._createDialog('MoreInfoWindow.qml')

    def messageActionTriggered(self, message_id, action_id):
        if False:
            return 10
        'Perform action based on user input.\n\n        Note that clicking "Disable" won\'t actually disable the data sending, but rather take the user to preferences where they can disable it.\n        '
        self._application.getPreferences().setValue('info/asked_send_slice_info', True)
        if action_id == 'MoreInfo':
            self.showMoreInfoDialog()
        self.send_slice_info_message.hide()

    def showMoreInfoDialog(self):
        if False:
            for i in range(10):
                print('nop')
        if self._more_info_dialog is None:
            self._more_info_dialog = self._createDialog('MoreInfoWindow.qml')
        self._more_info_dialog.show()

    def _createDialog(self, qml_name):
        if False:
            print('Hello World!')
        Logger.log('d', 'Creating dialog [%s]', qml_name)
        file_path = os.path.join(PluginRegistry.getInstance().getPluginPath(self.getPluginId()), qml_name)
        dialog = self._application.createQmlComponent(file_path, {'manager': self})
        return dialog

    @pyqtSlot(result=str)
    def getExampleData(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        if self._example_data_content is None:
            plugin_path = PluginRegistry.getInstance().getPluginPath(self.getPluginId())
            if not plugin_path:
                Logger.log('e', 'Could not get plugin path!', self.getPluginId())
                return None
            file_path = os.path.join(plugin_path, 'example_data.html')
            if file_path:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self._example_data_content = f.read()
                except EnvironmentError as e:
                    Logger.error(f'Unable to read example slice info data to show to the user: {e}')
                    self._example_data_content = '<i>' + catalog.i18nc('@text', 'Unable to read example data file.') + '</i>'
        return self._example_data_content

    @pyqtSlot(bool)
    def setSendSliceInfo(self, enabled: bool):
        if False:
            for i in range(10):
                print('nop')
        self._application.getPreferences().setValue('info/send_slice_info', enabled)

    def _getUserModifiedSettingKeys(self) -> list:
        if False:
            print('Hello World!')
        machine_manager = self._application.getMachineManager()
        global_stack = machine_manager.activeMachine
        user_modified_setting_keys = set()
        for stack in [global_stack] + global_stack.extruderList:
            all_keys = stack.userChanges.getAllKeys() | stack.qualityChanges.getAllKeys()
            user_modified_setting_keys |= all_keys
        return list(sorted(user_modified_setting_keys))

    def _onWriteStarted(self, output_device):
        if False:
            while True:
                i = 10
        try:
            if not self._application.getPreferences().getValue('info/send_slice_info'):
                Logger.log('d', "'info/send_slice_info' is turned off.")
                return
            machine_manager = self._application.getMachineManager()
            print_information = self._application.getPrintInformation()
            user_profile = self._application.getCuraAPI().account.userProfile
            global_stack = machine_manager.activeMachine
            data = dict()
            data['time_stamp'] = time.time()
            data['schema_version'] = 0
            data['cura_version'] = self._application.getVersion()
            data['cura_build_type'] = ApplicationMetadata.CuraBuildType
            org_id = user_profile.get('organization_id', None) if user_profile else None
            data['is_logged_in'] = self._application.getCuraAPI().account.isLoggedIn
            data['organization_id'] = org_id if org_id else None
            data['subscriptions'] = user_profile.get('subscriptions', []) if user_profile else []
            data['slice_uuid'] = print_information.slice_uuid
            active_mode = self._application.getPreferences().getValue('cura/active_mode')
            if active_mode == 0:
                data['active_mode'] = 'recommended'
            else:
                data['active_mode'] = 'custom'
            data['camera_view'] = self._application.getPreferences().getValue('general/camera_perspective_mode')
            if data['camera_view'] == 'orthographic':
                data['camera_view'] = 'orthogonal'
            definition_changes = global_stack.definitionChanges
            machine_settings_changed_by_user = False
            if definition_changes.getId() != 'empty':
                if definition_changes.getAllKeys():
                    machine_settings_changed_by_user = True
            data['machine_settings_changed_by_user'] = machine_settings_changed_by_user
            data['language'] = self._application.getPreferences().getValue('general/language')
            data['os'] = {'type': platform.system(), 'version': platform.version()}
            data['active_machine'] = {'definition_id': global_stack.definition.getId(), 'manufacturer': global_stack.definition.getMetaDataEntry('manufacturer', '')}
            data['extruders'] = []
            extruders = global_stack.extruderList
            extruders = sorted(extruders, key=lambda extruder: extruder.getMetaDataEntry('position'))
            for extruder in extruders:
                extruder_dict = dict()
                extruder_dict['active'] = machine_manager.activeStack == extruder
                extruder_dict['material'] = {'GUID': extruder.material.getMetaData().get('GUID', ''), 'type': extruder.material.getMetaData().get('material', ''), 'brand': extruder.material.getMetaData().get('brand', '')}
                extruder_position = int(extruder.getMetaDataEntry('position', '0'))
                if len(print_information.materialLengths) > extruder_position:
                    extruder_dict['material_used'] = print_information.materialLengths[extruder_position]
                extruder_dict['variant'] = extruder.variant.getName()
                extruder_dict['nozzle_size'] = extruder.getProperty('machine_nozzle_size', 'value')
                extruder_settings = dict()
                extruder_settings['wall_line_count'] = extruder.getProperty('wall_line_count', 'value')
                extruder_settings['retraction_enable'] = extruder.getProperty('retraction_enable', 'value')
                extruder_settings['infill_sparse_density'] = extruder.getProperty('infill_sparse_density', 'value')
                extruder_settings['infill_pattern'] = extruder.getProperty('infill_pattern', 'value')
                extruder_settings['gradual_infill_steps'] = extruder.getProperty('gradual_infill_steps', 'value')
                extruder_settings['default_material_print_temperature'] = extruder.getProperty('default_material_print_temperature', 'value')
                extruder_settings['material_print_temperature'] = extruder.getProperty('material_print_temperature', 'value')
                extruder_dict['extruder_settings'] = extruder_settings
                data['extruders'].append(extruder_dict)
            data['intent_category'] = global_stack.getIntentCategory()
            data['quality_profile'] = global_stack.quality.getMetaData().get('quality_type')
            data['user_modified_setting_keys'] = self._getUserModifiedSettingKeys()
            data['models'] = []
            for node in DepthFirstIterator(self._application.getController().getScene().getRoot()):
                if node.callDecoration('isSliceable'):
                    model = dict()
                    model['hash'] = node.getMeshData().getHash()
                    bounding_box = node.getBoundingBox()
                    if not bounding_box:
                        continue
                    model['bounding_box'] = {'minimum': {'x': bounding_box.minimum.x, 'y': bounding_box.minimum.y, 'z': bounding_box.minimum.z}, 'maximum': {'x': bounding_box.maximum.x, 'y': bounding_box.maximum.y, 'z': bounding_box.maximum.z}}
                    model['transformation'] = {'data': str(node.getWorldTransformation(copy=False).getData()).replace('\n', '')}
                    extruder_position = node.callDecoration('getActiveExtruderPosition')
                    model['extruder'] = 0 if extruder_position is None else int(extruder_position)
                    model_settings = dict()
                    model_stack = node.callDecoration('getStack')
                    if model_stack:
                        model_settings['support_enabled'] = model_stack.getProperty('support_enable', 'value')
                        model_settings['support_extruder_nr'] = int(model_stack.getExtruderPositionValueWithDefault('support_extruder_nr'))
                        model_settings['infill_mesh'] = model_stack.getProperty('infill_mesh', 'value')
                        model_settings['cutting_mesh'] = model_stack.getProperty('cutting_mesh', 'value')
                        model_settings['support_mesh'] = model_stack.getProperty('support_mesh', 'value')
                        model_settings['anti_overhang_mesh'] = model_stack.getProperty('anti_overhang_mesh', 'value')
                        model_settings['wall_line_count'] = model_stack.getProperty('wall_line_count', 'value')
                        model_settings['retraction_enable'] = model_stack.getProperty('retraction_enable', 'value')
                        model_settings['infill_sparse_density'] = model_stack.getProperty('infill_sparse_density', 'value')
                        model_settings['infill_pattern'] = model_stack.getProperty('infill_pattern', 'value')
                        model_settings['gradual_infill_steps'] = model_stack.getProperty('gradual_infill_steps', 'value')
                    model['model_settings'] = model_settings
                    if node.source_mime_type is None:
                        model['mime_type'] = ''
                    else:
                        model['mime_type'] = node.source_mime_type.name
                    data['models'].append(model)
            print_times = print_information.printTimes()
            data['print_times'] = {'travel': int(print_times['travel'].getDisplayString(DurationFormat.Format.Seconds)), 'support': int(print_times['support'].getDisplayString(DurationFormat.Format.Seconds)), 'infill': int(print_times['infill'].getDisplayString(DurationFormat.Format.Seconds)), 'total': int(print_information.currentPrintTime.getDisplayString(DurationFormat.Format.Seconds))}
            print_settings = dict()
            print_settings['layer_height'] = global_stack.getProperty('layer_height', 'value')
            print_settings['support_enabled'] = global_stack.getProperty('support_enable', 'value')
            print_settings['support_extruder_nr'] = int(global_stack.getExtruderPositionValueWithDefault('support_extruder_nr'))
            print_settings['adhesion_type'] = global_stack.getProperty('adhesion_type', 'value')
            print_settings['wall_line_count'] = global_stack.getProperty('wall_line_count', 'value')
            print_settings['retraction_enable'] = global_stack.getProperty('retraction_enable', 'value')
            print_settings['prime_tower_enable'] = global_stack.getProperty('prime_tower_enable', 'value')
            print_settings['infill_sparse_density'] = global_stack.getProperty('infill_sparse_density', 'value')
            print_settings['infill_pattern'] = global_stack.getProperty('infill_pattern', 'value')
            print_settings['gradual_infill_steps'] = global_stack.getProperty('gradual_infill_steps', 'value')
            print_settings['print_sequence'] = global_stack.getProperty('print_sequence', 'value')
            data['print_settings'] = print_settings
            data['output_to'] = type(output_device).__name__
            time_setup = 0.0
            time_backend = 0.0
            if not print_information.preSliced:
                backend_info = self._application.getBackend().resetAndReturnLastSliceTimeStats()
                time_start_process = backend_info['time_start_process']
                time_send_message = backend_info['time_send_message']
                time_end_slice = backend_info['time_end_slice']
                if time_start_process and time_send_message and time_end_slice:
                    time_setup = time_send_message - time_start_process
                    time_backend = time_end_slice - time_send_message
            data['engine_stats'] = {'is_presliced': int(print_information.preSliced), 'time_setup': int(round(time_setup)), 'time_backend': int(round(time_backend))}
            binary_data = json.dumps(data).encode('utf-8')
            network_manager = self._application.getHttpRequestManager()
            network_manager.post(self.info_url, data=binary_data, callback=self._onRequestFinished, error_callback=self._onRequestError)
        except Exception:
            Logger.logException('e', 'Exception raised while sending slice info.')

    def _onRequestFinished(self, reply: 'QNetworkReply') -> None:
        if False:
            for i in range(10):
                print('nop')
        status_code = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
        if status_code == 200:
            Logger.log('i', 'SliceInfo sent successfully')
            return
        data = reply.readAll().data().decode('utf-8')
        Logger.log('e', 'SliceInfo request failed, status code %s, data: %s', status_code, data)

    def _onRequestError(self, reply: 'QNetworkReply', error: 'QNetworkReply.NetworkError') -> None:
        if False:
            i = 10
            return i + 15
        Logger.log('e', 'Got error for SliceInfo request: %s', reply.errorString())