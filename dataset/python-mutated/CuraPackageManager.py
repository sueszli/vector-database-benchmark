import glob
import os
from pathlib import Path
from typing import Any, cast, Dict, List, Set, Tuple, TYPE_CHECKING, Optional
from UM.Logger import Logger
from UM.PluginRegistry import PluginRegistry
from cura.CuraApplication import CuraApplication
from cura.Settings.GlobalStack import GlobalStack
from UM.PackageManager import PackageManager
from UM.Resources import Resources
from UM.i18n import i18nCatalog
from urllib.parse import unquote_plus
catalog = i18nCatalog('cura')
if TYPE_CHECKING:
    from UM.Qt.QtApplication import QtApplication
    from PyQt6.QtCore import QObject

class CuraPackageManager(PackageManager):

    def __init__(self, application: 'QtApplication', parent: Optional['QObject']=None) -> None:
        if False:
            return 10
        super().__init__(application, parent)
        self._local_packages: Optional[List[Dict[str, Any]]] = None
        self._local_packages_ids: Optional[Set[str]] = None
        self.installedPackagesChanged.connect(self._updateLocalPackages)

    def _updateLocalPackages(self) -> None:
        if False:
            print('Hello World!')
        self._local_packages = self.getAllLocalPackages()
        self._local_packages_ids = set((pkg['package_id'] for pkg in self._local_packages))

    @property
    def local_packages(self) -> List[Dict[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        'locally installed packages, lazy execution'
        if self._local_packages is None:
            self._updateLocalPackages()
        return cast(List[Dict[str, Any]], self._local_packages)

    @property
    def local_packages_ids(self) -> Set[str]:
        if False:
            while True:
                i = 10
        'locally installed packages, lazy execution'
        if self._local_packages_ids is None:
            self._updateLocalPackages()
        return cast(Set[str], self._local_packages_ids)

    def initialize(self) -> None:
        if False:
            while True:
                i = 10
        self._installation_dirs_dict['materials'] = Resources.getStoragePath(CuraApplication.ResourceTypes.MaterialInstanceContainer)
        self._installation_dirs_dict['qualities'] = Resources.getStoragePath(CuraApplication.ResourceTypes.QualityInstanceContainer)
        self._installation_dirs_dict['variants'] = Resources.getStoragePath(CuraApplication.ResourceTypes.VariantInstanceContainer)
        self._installation_dirs_dict['images'] = Resources.getStoragePath(CuraApplication.ResourceTypes.ImageFiles)
        self._installation_dirs_dict['intents'] = Resources.getStoragePath(CuraApplication.ResourceTypes.IntentInstanceContainer)
        self._installation_dirs_dict['intent'] = Resources.getStoragePath(CuraApplication.ResourceTypes.IntentInstanceContainer)
        super().initialize()

    def isMaterialBundled(self, file_name: str, guid: str):
        if False:
            return 10
        ' Check if there is a bundled material name with file_name and guid '
        for path in Resources.getSecureSearchPaths():
            paths = [Path(p) for p in glob.glob(path + '/**/*.xml.fdm_material', recursive=True)]
            for material in paths:
                if material.name == file_name:
                    Logger.info(f'Found bundled material: {material.name}. Located in path: {str(material)}')
                    with open(material, encoding='utf-8') as f:
                        parsed_guid = PluginRegistry.getInstance().getPluginObject('XmlMaterialProfile').getMetadataFromSerialized(f.read(), 'GUID')
                        if guid == parsed_guid:
                            return True
        return False

    def getMaterialFilePackageId(self, file_name: str, guid: str) -> str:
        if False:
            i = 10
            return i + 15
        'Get the id of the installed material package that contains file_name'
        file_name = unquote_plus(file_name)
        for material_package in [f for f in os.scandir(self._installation_dirs_dict['materials']) if f.is_dir()]:
            package_id = material_package.name
            for (root, _, file_names) in os.walk(material_package.path):
                if file_name not in file_names:
                    continue
                with open(os.path.join(root, file_name), encoding='utf-8') as f:
                    parsed_guid = PluginRegistry.getInstance().getPluginObject('XmlMaterialProfile').getMetadataFromSerialized(f.read(), 'GUID')
                    if guid == parsed_guid:
                        return package_id
        Logger.error('Could not find package_id for file: {} with GUID: {} '.format(file_name, guid))
        Logger.error(f'Bundled paths searched: {list(Resources.getSecureSearchPaths())}')
        return ''

    def getMachinesUsingPackage(self, package_id: str) -> Tuple[List[Tuple[GlobalStack, str, str]], List[Tuple[GlobalStack, str, str]]]:
        if False:
            return 10
        'Returns a list of where the package is used\n\n        It loops through all the package contents and see if some of the ids are used.\n\n        :param package_id: package id to search for\n        :return: empty if it is never used, otherwise a list consisting of 3-tuples\n        '
        ids = self.getPackageContainerIds(package_id)
        container_stacks = self._application.getContainerRegistry().findContainerStacks()
        global_stacks = [container_stack for container_stack in container_stacks if isinstance(container_stack, GlobalStack)]
        machine_with_materials = []
        machine_with_qualities = []
        for container_id in ids:
            for global_stack in global_stacks:
                for (extruder_nr, extruder_stack) in enumerate(global_stack.extruderList):
                    if container_id in (extruder_stack.material.getId(), extruder_stack.material.getMetaData().get('base_file')):
                        machine_with_materials.append((global_stack, str(extruder_nr), container_id))
                    if container_id == extruder_stack.quality.getId():
                        machine_with_qualities.append((global_stack, str(extruder_nr), container_id))
        return (machine_with_materials, machine_with_qualities)

    def getAllLocalPackages(self) -> List[Dict[str, Any]]:
        if False:
            while True:
                i = 10
        ' Returns an unordered list of all the package_info of installed, to be installed, or bundled packages'
        packages: List[Dict[str, Any]] = []
        for packages_to_add in self.getAllInstalledPackagesInfo().values():
            packages.extend(packages_to_add)
        return packages