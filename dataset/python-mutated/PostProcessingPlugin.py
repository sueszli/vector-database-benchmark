import configparser
import importlib.util
import io
import os.path
import pkgutil
import sys
from typing import Dict, Type, TYPE_CHECKING, List, Optional, cast
from PyQt6.QtCore import QObject, pyqtProperty, pyqtSignal, pyqtSlot
from UM.Application import Application
from UM.Extension import Extension
from UM.Logger import Logger
from UM.PluginRegistry import PluginRegistry
from UM.Resources import Resources
from UM.Trust import Trust, TrustBasics
from UM.i18n import i18nCatalog
from cura import ApplicationMetadata
from cura.CuraApplication import CuraApplication
i18n_catalog = i18nCatalog('cura')
if TYPE_CHECKING:
    from .Script import Script

class PostProcessingPlugin(QObject, Extension):
    """Extension type plugin that enables pre-written scripts to post process g-code files."""

    def __init__(self, parent=None) -> None:
        if False:
            i = 10
            return i + 15
        QObject.__init__(self, parent)
        Extension.__init__(self)
        self.setMenuName(i18n_catalog.i18nc('@item:inmenu', 'Post Processing'))
        self.addMenuItem(i18n_catalog.i18nc('@item:inmenu', 'Modify G-Code'), self.showPopup)
        self._view = None
        self._loaded_scripts = {}
        self._script_labels = {}
        self._script_list = []
        self._selected_script_index = -1
        self._global_container_stack = Application.getInstance().getGlobalContainerStack()
        if self._global_container_stack:
            self._global_container_stack.metaDataChanged.connect(self._restoreScriptInforFromMetadata)
        Application.getInstance().getOutputDeviceManager().writeStarted.connect(self.execute)
        Application.getInstance().globalContainerStackChanged.connect(self._onGlobalContainerStackChanged)
        CuraApplication.getInstance().mainWindowChanged.connect(self._createView)
    selectedIndexChanged = pyqtSignal()

    @pyqtProperty(str, notify=selectedIndexChanged)
    def selectedScriptDefinitionId(self) -> Optional[str]:
        if False:
            return 10
        try:
            return self._script_list[self._selected_script_index].getDefinitionId()
        except IndexError:
            return ''

    @pyqtProperty(str, notify=selectedIndexChanged)
    def selectedScriptStackId(self) -> Optional[str]:
        if False:
            print('Hello World!')
        try:
            return self._script_list[self._selected_script_index].getStackId()
        except IndexError:
            return ''

    def execute(self, output_device) -> None:
        if False:
            i = 10
            return i + 15
        'Execute all post-processing scripts on the gcode.'
        scene = Application.getInstance().getController().getScene()
        if not hasattr(scene, 'gcode_dict'):
            return
        gcode_dict = getattr(scene, 'gcode_dict')
        if not gcode_dict:
            return
        active_build_plate_id = CuraApplication.getInstance().getMultiBuildPlateModel().activeBuildPlate
        gcode_list = gcode_dict[active_build_plate_id]
        if not gcode_list:
            return
        if ';POSTPROCESSED' not in gcode_list[0]:
            for script in self._script_list:
                try:
                    gcode_list = script.execute(gcode_list)
                except Exception:
                    Logger.logException('e', 'Exception in post-processing script.')
            if len(self._script_list):
                gcode_list[0] += ';POSTPROCESSED\n'
                pp_name_list = Application.getInstance().getGlobalContainerStack().getMetaDataEntry('post_processing_scripts')
                for pp_name in pp_name_list.split('\n'):
                    pp_name = pp_name.split(']')
                    gcode_list[0] += ';  ' + str(pp_name[0]) + ']\n'
            gcode_dict[active_build_plate_id] = gcode_list
            setattr(scene, 'gcode_dict', gcode_dict)
        else:
            Logger.log('e', 'Already post processed')

    @pyqtSlot(int)
    def setSelectedScriptIndex(self, index: int) -> None:
        if False:
            while True:
                i = 10
        if self._selected_script_index != index:
            self._selected_script_index = index
            self.selectedIndexChanged.emit()

    @pyqtProperty(int, notify=selectedIndexChanged)
    def selectedScriptIndex(self) -> int:
        if False:
            while True:
                i = 10
        return self._selected_script_index

    @pyqtSlot(int, int)
    def moveScript(self, index: int, new_index: int) -> None:
        if False:
            return 10
        if new_index < 0 or new_index > len(self._script_list) - 1:
            return
        else:
            (self._script_list[new_index], self._script_list[index]) = (self._script_list[index], self._script_list[new_index])
            self.scriptListChanged.emit()
            self.selectedIndexChanged.emit()
            self._propertyChanged()

    @pyqtSlot(int)
    def removeScriptByIndex(self, index: int) -> None:
        if False:
            while True:
                i = 10
        'Remove a script from the active script list by index.'
        self._script_list.pop(index)
        if len(self._script_list) - 1 < self._selected_script_index:
            self._selected_script_index = len(self._script_list) - 1
        self.scriptListChanged.emit()
        self.selectedIndexChanged.emit()
        self._propertyChanged()

    def loadAllScripts(self) -> None:
        if False:
            while True:
                i = 10
        'Load all scripts from all paths where scripts can be found.\n\n        This should probably only be done on init.\n        '
        if self._loaded_scripts:
            return
        for path in set([os.path.join(Resources.getStoragePath(r), 'scripts') for r in [Resources.Resources, Resources.Preferences]]):
            if not os.path.isdir(path):
                try:
                    os.makedirs(path)
                except OSError:
                    Logger.log('w', 'Unable to create a folder for scripts: ' + path)
        resource_folders = [PluginRegistry.getInstance().getPluginPath('PostProcessingPlugin'), Resources.getStoragePath(Resources.Preferences)]
        resource_folders.extend(Resources.getAllPathsForType(Resources.Resources))
        for root in resource_folders:
            if root is None:
                continue
            path = os.path.join(root, 'scripts')
            if not os.path.isdir(path):
                continue
            self.loadScripts(path)

    def loadScripts(self, path: str) -> None:
        if False:
            return 10
        'Load all scripts from provided path.\n\n        This should probably only be done on init.\n        :param path: Path to check for scripts.\n        '
        if ApplicationMetadata.IsEnterpriseVersion:
            install_prefix = os.path.abspath(CuraApplication.getInstance().getInstallPrefix())
            try:
                is_in_installation_path = os.path.commonpath([install_prefix, path]).startswith(install_prefix)
            except ValueError:
                is_in_installation_path = False
            if not is_in_installation_path:
                TrustBasics.removeCached(path)
        scripts = pkgutil.iter_modules(path=[path])
        'Load all scripts in the scripts folders'
        for (loader, script_name, ispkg) in scripts:
            if script_name not in sys.modules:
                try:
                    file_path = os.path.join(path, script_name + '.py')
                    if not self._isScriptAllowed(file_path):
                        Logger.warning('Skipped loading post-processing script {}: not trusted'.format(file_path))
                        continue
                    spec = importlib.util.spec_from_file_location(__name__ + '.' + script_name, file_path)
                    if spec is None:
                        continue
                    loaded_script = importlib.util.module_from_spec(spec)
                    if spec.loader is None:
                        continue
                    spec.loader.exec_module(loaded_script)
                    sys.modules[script_name] = loaded_script
                    loaded_class = getattr(loaded_script, script_name)
                    temp_object = loaded_class()
                    Logger.log('d', 'Begin loading of script: %s', script_name)
                    try:
                        setting_data = temp_object.getSettingData()
                        if 'name' in setting_data and 'key' in setting_data:
                            self._script_labels[setting_data['key']] = setting_data['name']
                            self._loaded_scripts[setting_data['key']] = loaded_class
                        else:
                            Logger.log('w', 'Script %s.py has no name or key', script_name)
                            self._script_labels[script_name] = script_name
                            self._loaded_scripts[script_name] = loaded_class
                    except AttributeError:
                        Logger.log('e', 'Script %s.py is not a recognised script type. Ensure it inherits Script', script_name)
                    except NotImplementedError:
                        Logger.log('e', 'Script %s.py has no implemented settings', script_name)
                except Exception as e:
                    Logger.logException('e', 'Exception occurred while loading post processing plugin: {error_msg}'.format(error_msg=str(e)))
    loadedScriptListChanged = pyqtSignal()

    @pyqtProperty('QVariantList', notify=loadedScriptListChanged)
    def loadedScriptList(self) -> List[str]:
        if False:
            while True:
                i = 10
        return sorted(list(self._loaded_scripts.keys()))

    @pyqtSlot(str, result=str)
    def getScriptLabelByKey(self, key: str) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        return self._script_labels.get(key)
    scriptListChanged = pyqtSignal()

    @pyqtProperty('QStringList', notify=scriptListChanged)
    def scriptList(self) -> List[str]:
        if False:
            print('Hello World!')
        script_list = [script.getSettingData()['key'] for script in self._script_list]
        return script_list

    @pyqtSlot(str)
    def addScriptToList(self, key: str) -> None:
        if False:
            print('Hello World!')
        Logger.log('d', 'Adding script %s to list.', key)
        new_script = self._loaded_scripts[key]()
        new_script.initialize()
        self._script_list.append(new_script)
        self.setSelectedScriptIndex(len(self._script_list) - 1)
        self.scriptListChanged.emit()
        self._propertyChanged()

    def _restoreScriptInforFromMetadata(self):
        if False:
            return 10
        self.loadAllScripts()
        new_stack = self._global_container_stack
        if new_stack is None:
            return
        self._script_list.clear()
        if not new_stack.getMetaDataEntry('post_processing_scripts'):
            self.scriptListChanged.emit()
            self.setSelectedScriptIndex(-1)
            return
        self._script_list.clear()
        scripts_list_strs = new_stack.getMetaDataEntry('post_processing_scripts')
        for script_str in scripts_list_strs.split('\n'):
            if not script_str:
                continue
            script_str = script_str.replace('\\\\\\n', '\n').replace('\\\\\\\\', '\\\\')
            script_parser = configparser.ConfigParser(interpolation=None)
            script_parser.optionxform = str
            try:
                script_parser.read_string(script_str)
            except configparser.Error as e:
                Logger.error('Stored post-processing scripts have syntax errors: {err}'.format(err=str(e)))
                continue
            for (script_name, settings) in script_parser.items():
                if script_name == 'DEFAULT':
                    continue
                if script_name not in self._loaded_scripts:
                    Logger.log('e', 'Unknown post-processing script {script_name} was encountered in this global stack.'.format(script_name=script_name))
                    continue
                new_script = self._loaded_scripts[script_name]()
                new_script.initialize()
                for (setting_key, setting_value) in settings.items():
                    if new_script._instance is not None:
                        new_script._instance.setProperty(setting_key, 'value', setting_value)
                self._script_list.append(new_script)
        self.setSelectedScriptIndex(0)
        self.selectedIndexChanged.emit()
        self.scriptListChanged.emit()
        self._propertyChanged()

    def _onGlobalContainerStackChanged(self) -> None:
        if False:
            i = 10
            return i + 15
        'When the global container stack is changed, swap out the list of active scripts.'
        if self._global_container_stack:
            self._global_container_stack.metaDataChanged.disconnect(self._restoreScriptInforFromMetadata)
        self._global_container_stack = Application.getInstance().getGlobalContainerStack()
        if self._global_container_stack:
            self._global_container_stack.metaDataChanged.connect(self._restoreScriptInforFromMetadata)
        self._restoreScriptInforFromMetadata()

    @pyqtSlot()
    def writeScriptsToStack(self) -> None:
        if False:
            while True:
                i = 10
        script_list_strs = []
        for script in self._script_list:
            parser = configparser.ConfigParser(interpolation=None)
            parser.optionxform = str
            script_name = script.getSettingData()['key']
            parser.add_section(script_name)
            for key in script.getSettingData()['settings']:
                value = script.getSettingValueByKey(key)
                parser[script_name][key] = str(value)
            serialized = io.StringIO()
            parser.write(serialized)
            serialized.seek(0)
            script_str = serialized.read()
            script_str = script_str.replace('\\\\', '\\\\\\\\').replace('\n', '\\\\\\n')
            script_list_strs.append(script_str)
        script_list_string = '\n'.join(script_list_strs)
        if self._global_container_stack is None:
            return
        self._global_container_stack.metaDataChanged.disconnect(self._restoreScriptInforFromMetadata)
        if 'post_processing_scripts' not in self._global_container_stack.getMetaData():
            self._global_container_stack.setMetaDataEntry('post_processing_scripts', '')
        self._global_container_stack.setMetaDataEntry('post_processing_scripts', script_list_string)
        self._global_container_stack.metaDataChanged.connect(self._restoreScriptInforFromMetadata)

    def _createView(self) -> None:
        if False:
            print('Hello World!')
        'Creates the view used by show popup.\n\n        The view is saved because of the fairly aggressive garbage collection.\n        '
        Logger.log('d', 'Creating post processing plugin view.')
        self.loadAllScripts()
        path = os.path.join(cast(str, PluginRegistry.getInstance().getPluginPath('PostProcessingPlugin')), 'PostProcessingPlugin.qml')
        self._view = CuraApplication.getInstance().createQmlComponent(path, {'manager': self})
        if self._view is None:
            Logger.log('e', 'Not creating PostProcessing button near save button because the QML component failed to be created.')
            return
        Logger.log('d', 'Post processing view created.')
        CuraApplication.getInstance().addAdditionalComponent('saveButton', self._view.findChild(QObject, 'postProcessingSaveAreaButton'))

    def showPopup(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Show the (GUI) popup of the post processing plugin.'
        if self._view is None:
            self._createView()
            if self._view is None:
                Logger.log('e', 'Not creating PostProcessing window since the QML component failed to be created.')
                return
        self._view.show()

    def _propertyChanged(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Property changed: trigger re-slice\n\n        To do this we use the global container stack propertyChanged.\n        Re-slicing is necessary for setting changes in this plugin, because the changes\n        are applied only once per "fresh" gcode\n        '
        global_container_stack = Application.getInstance().getGlobalContainerStack()
        if global_container_stack is not None:
            global_container_stack.propertyChanged.emit('post_processing_plugin', 'value')

    @staticmethod
    def _isScriptAllowed(file_path: str) -> bool:
        if False:
            print('Hello World!')
        'Checks whether the given file is allowed to be loaded'
        if not ApplicationMetadata.IsEnterpriseVersion:
            return True
        dir_path = os.path.split(file_path)[0]
        plugin_path = PluginRegistry.getInstance().getPluginPath('PostProcessingPlugin')
        assert plugin_path is not None
        bundled_path = os.path.join(plugin_path, 'scripts')
        if dir_path == bundled_path:
            return True
        trust_instance = Trust.getInstanceOrNone()
        if trust_instance is not None and Trust.signatureFileExistsFor(file_path):
            if trust_instance.signedFileCheck(file_path):
                return True
        return False