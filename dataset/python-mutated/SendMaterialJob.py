import os
from typing import Dict, TYPE_CHECKING, Set, List
from PyQt6.QtNetwork import QNetworkReply, QNetworkRequest
from UM.Job import Job
from UM.Logger import Logger
from cura.CuraApplication import CuraApplication
from cura.Utils.Threading import call_on_qt_thread
from ..Models.Http.ClusterMaterial import ClusterMaterial
from ..Models.LocalMaterial import LocalMaterial
from ..Messages.MaterialSyncMessage import MaterialSyncMessage
import time
import threading
if TYPE_CHECKING:
    from .LocalClusterOutputDevice import LocalClusterOutputDevice

class SendMaterialJob(Job):
    """Asynchronous job to send material profiles to the printer.

    This way it won't freeze up the interface while sending those materials.
    """

    def __init__(self, device: 'LocalClusterOutputDevice') -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.device = device
        self._send_material_thread = threading.Thread(target=self._sendMissingMaterials)
        self._send_material_thread.setDaemon(True)
        self._remote_materials = {}

    def run(self) -> None:
        if False:
            i = 10
            return i + 15
        'Send the request to the printer and register a callback'
        self.device.getMaterials(on_finished=self._onGetMaterials)

    def _onGetMaterials(self, materials: List[ClusterMaterial]) -> None:
        if False:
            i = 10
            return i + 15
        'Callback for when the remote materials were returned.'
        remote_materials_by_guid = {material.guid: material for material in materials}
        self._remote_materials = remote_materials_by_guid
        self._send_material_thread.start()

    def _sendMissingMaterials(self) -> None:
        if False:
            return 10
        'Determine which materials should be updated and send them to the printer.\n\n        :param remote_materials_by_guid: The remote materials by GUID.\n        '
        local_materials_by_guid = self._getLocalMaterials()
        if len(local_materials_by_guid) == 0:
            Logger.log('d', 'There are no local materials to synchronize with the printer.')
            return
        material_ids_to_send = self._determineMaterialsToSend(local_materials_by_guid, self._remote_materials)
        if len(material_ids_to_send) == 0:
            Logger.log('d', 'There are no remote materials to update.')
            return
        self._sendMaterials(material_ids_to_send)

    @staticmethod
    def _determineMaterialsToSend(local_materials: Dict[str, LocalMaterial], remote_materials: Dict[str, ClusterMaterial]) -> Set[str]:
        if False:
            i = 10
            return i + 15
        "From the local and remote materials, determine which ones should be synchronized.\n\n        Makes a Set of id's containing only the id's of the materials that are not on the printer yet or the ones that\n        are newer in Cura.\n        :param local_materials: The local materials by GUID.\n        :param remote_materials: The remote materials by GUID.\n        "
        return {local_material.id for (guid, local_material) in local_materials.items() if guid not in remote_materials.keys() or local_material.version > remote_materials[guid].version}

    def _sendMaterials(self, materials_to_send: Set[str]) -> None:
        if False:
            return 10
        "Send the materials to the printer.\n\n        The given materials will be loaded from disk en sent to to printer.\n        The given id's will be matched with filenames of the locally stored materials.\n        :param materials_to_send: A set with id's of materials that must be sent.\n        "
        container_registry = CuraApplication.getInstance().getContainerRegistry()
        all_materials = container_registry.findInstanceContainersMetadata(type='material')
        all_base_files = {material['base_file'] for material in all_materials if 'base_file' in material}
        if 'empty_material' in all_base_files:
            all_base_files.remove('empty_material')
        for root_material_id in all_base_files:
            if root_material_id not in materials_to_send:
                continue
            file_path = container_registry.getContainerFilePathById(root_material_id)
            if not file_path:
                Logger.log('w', 'Cannot get file path for material container [%s]', root_material_id)
                continue
            file_name = os.path.basename(file_path)
            self._sendMaterialFile(file_path, file_name, root_material_id)
            time.sleep(1)

    @call_on_qt_thread
    def _sendMaterialFile(self, file_path: str, file_name: str, material_id: str) -> None:
        if False:
            while True:
                i = 10
        'Send a single material file to the printer.\n\n        Also add the material signature file if that is available.\n        :param file_path: The path of the material file.\n        :param file_name: The name of the material file.\n        :param material_id: The ID of the material in the file.\n        '
        parts = []
        try:
            with open(file_path, 'rb') as f:
                parts.append(self.device.createFormPart('name="file"; filename="{file_name}"'.format(file_name=file_name), f.read()))
        except FileNotFoundError:
            Logger.error('Unable to send material {material_id}, since it has been deleted in the meanwhile.'.format(material_id=material_id))
            return
        except EnvironmentError as e:
            Logger.error(f"Unable to send material {material_id}. We can't open that file for reading: {str(e)}")
            return
        signature_file_path = '{}.sig'.format(file_path)
        if os.path.exists(signature_file_path):
            signature_file_name = os.path.basename(signature_file_path)
            with open(signature_file_path, 'rb') as f:
                parts.append(self.device.createFormPart('name="signature_file"; filename="{file_name}"'.format(file_name=signature_file_name), f.read()))
        self.device.postFormWithParts(target='/cluster-api/v1/materials/', parts=parts, on_finished=self._sendingFinished)

    def _sendingFinished(self, reply: QNetworkReply) -> None:
        if False:
            while True:
                i = 10
        'Check a reply from an upload to the printer and log an error when the call failed'
        if reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute) != 200:
            Logger.log('w', 'Error while syncing material: %s', reply.errorString())
            return
        body = reply.readAll().data().decode('utf8')
        if 'not added' in body:
            return
        MaterialSyncMessage(self.device).show()

    @staticmethod
    def _getLocalMaterials() -> Dict[str, LocalMaterial]:
        if False:
            for i in range(10):
                print('nop')
        'Retrieves a list of local materials\n\n        Only the new newest version of the local materials is returned\n        :return: a dictionary of LocalMaterial objects by GUID\n        '
        result = {}
        all_materials = CuraApplication.getInstance().getContainerRegistry().findInstanceContainersMetadata(type='material')
        all_base_files = [material for material in all_materials if material['id'] == material.get('base_file') and material.get('visible', True)]
        for material_metadata in all_base_files:
            try:
                material_metadata['version'] = int(material_metadata['version'])
                local_material = LocalMaterial(**material_metadata)
                local_material.id = material_metadata['id']
                if local_material.GUID not in result or local_material.GUID not in result or local_material.version > result[local_material.GUID].version:
                    result[local_material.GUID] = local_material
            except KeyError:
                Logger.logException('w', 'Local material {} has missing values.'.format(material_metadata['id']))
            except ValueError:
                Logger.logException('w', 'Local material {} has invalid values.'.format(material_metadata['id']))
            except TypeError:
                Logger.logException('w', 'Local material {} has invalid values.'.format(material_metadata['id']))
        return result