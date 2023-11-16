"""Methods for returning the correct file system class to the client."""
from __future__ import annotations
from core import feconf
from core import utils
from core.domain import image_services
from core.platform import models
from typing import Dict, List, Optional
MYPY = False
if MYPY:
    from mypy_imports import app_identity_services
    from mypy_imports import storage_services
    from proto_files import text_classifier_pb2
storage_services = models.Registry.import_storage_services()
app_identity_services = models.Registry.import_app_identity_services()
CHANGE_LIST_SAVE: List[Dict[str, str]] = [{'cmd': 'save'}]
ALLOWED_ENTITY_NAMES: List[str] = [feconf.ENTITY_TYPE_EXPLORATION, feconf.ENTITY_TYPE_BLOG_POST, feconf.ENTITY_TYPE_TOPIC, feconf.ENTITY_TYPE_SKILL, feconf.ENTITY_TYPE_STORY, feconf.ENTITY_TYPE_QUESTION, feconf.ENTITY_TYPE_USER]
ALLOWED_SUGGESTION_IMAGE_CONTEXTS: List[str] = [feconf.IMAGE_CONTEXT_QUESTION_SUGGESTIONS, feconf.IMAGE_CONTEXT_EXPLORATION_SUGGESTIONS]

class GeneralFileSystem:
    """The parent class which is inherited by GcsFileSystem.

    Attributes:
        entity_name: str. The name of the entity (eg: exploration, topic etc).
        entity_id: str. The ID of the corresponding entity.
    """

    def __init__(self, entity_name: str, entity_id: str) -> None:
        if False:
            return 10
        'Constructs a GeneralFileSystem object.\n\n        Args:\n            entity_name: str. The name of the entity\n                (eg: exploration, topic etc).\n            entity_id: str. The ID of the corresponding entity.\n        '
        self._validate_entity_parameters(entity_name, entity_id)
        self._assets_path = '%s/%s/assets' % (entity_name, entity_id)

    def _validate_entity_parameters(self, entity_name: str, entity_id: str) -> None:
        if False:
            while True:
                i = 10
        'Checks whether the entity_id and entity_name passed in are valid.\n\n        Args:\n            entity_name: str. The name of the entity\n                (eg: exploration, topic etc).\n            entity_id: str. The ID of the corresponding entity.\n\n        Raises:\n            ValidationError. When parameters passed in are invalid.\n        '
        if entity_name not in ALLOWED_ENTITY_NAMES and entity_name not in ALLOWED_SUGGESTION_IMAGE_CONTEXTS:
            raise utils.ValidationError('Invalid entity_name received: %s.' % entity_name)
        if not isinstance(entity_id, str):
            raise utils.ValidationError('Invalid entity_id received: %s' % entity_id)
        if entity_id == '':
            raise utils.ValidationError('Entity id cannot be empty')

    @property
    def assets_path(self) -> str:
        if False:
            print('Hello World!')
        'Returns the path of the parent folder of assets.\n\n        Returns:\n            str. The path.\n        '
        return self._assets_path

class GcsFileSystem(GeneralFileSystem):
    """Wrapper for a file system based on GCS.

    This implementation ignores versioning.
    """

    def __init__(self, entity_name: str, entity_id: str) -> None:
        if False:
            print('Hello World!')
        self._bucket_name = app_identity_services.get_gcs_resource_bucket_name()
        super().__init__(entity_name, entity_id)

    def _get_gcs_file_url(self, filepath: str) -> str:
        if False:
            while True:
                i = 10
        "Returns the constructed GCS file URL.\n\n        Args:\n            filepath: str. The path to the relevant file within the entity's\n                assets folder.\n\n        Returns:\n            str. The GCS file URL.\n        "
        gcs_file_url = '%s/%s' % (self._assets_path, filepath)
        return gcs_file_url

    def _check_filepath(self, filepath: str) -> None:
        if False:
            print('Hello World!')
        "Raises an error if a filepath is invalid.\n\n        Args:\n            filepath: str. The path to the relevant file within the entity's\n                assets folder.\n\n        Raises:\n            OSError. Invalid filepath.\n        "
        base_dir = utils.vfs_construct_path('/', self.assets_path, 'assets')
        absolute_path = utils.vfs_construct_path(base_dir, filepath)
        normalized_path = utils.vfs_normpath(absolute_path)
        if not normalized_path.startswith(base_dir):
            raise IOError('Invalid filepath: %s' % filepath)

    def isfile(self, filepath: str) -> bool:
        if False:
            print('Hello World!')
        "Checks if the file with the given filepath exists in the GCS.\n\n        Args:\n            filepath: str. The path to the relevant file within the entity's\n                assets folder.\n\n        Returns:\n            bool. Whether the file exists in GCS.\n        "
        self._check_filepath(filepath)
        return storage_services.isfile(self._bucket_name, self._get_gcs_file_url(filepath))

    def get(self, filepath: str) -> bytes:
        if False:
            return 10
        "Gets a file as an unencoded stream of raw bytes.\n\n        Args:\n            filepath: str. The path to the relevant file within the entity's\n                assets folder.\n\n        Returns:\n            bytes. A stream of raw bytes if the file exists.\n\n        Raises:\n            OSError. Given file does not exist.\n        "
        if self.isfile(filepath):
            return storage_services.get(self._bucket_name, self._get_gcs_file_url(filepath))
        else:
            raise IOError('File %s not found.' % filepath)

    def commit(self, filepath: str, raw_bytes: bytes, mimetype: Optional[str]=None) -> None:
        if False:
            i = 10
            return i + 15
        "Commit raw_bytes to the relevant file in the entity's assets folder.\n\n        Args:\n            filepath: str. The path to the relevant file within the entity's\n                assets folder.\n            raw_bytes: bytes. The content to be stored in the file.\n            mimetype: Optional[str]. The content-type of the cloud file.\n        "
        self._check_filepath(filepath)
        storage_services.commit(self._bucket_name, self._get_gcs_file_url(filepath), raw_bytes, mimetype)

    def delete(self, filepath: str) -> None:
        if False:
            print('Hello World!')
        "Deletes a file and the metadata associated with it.\n\n        Args:\n            filepath: str. The path to the relevant file within the entity's\n                assets folder.\n\n        Raises:\n            OSError. Given file does not exist.\n        "
        if self.isfile(filepath):
            storage_services.delete(self._bucket_name, self._get_gcs_file_url(filepath))
        else:
            raise IOError('File does not exist: %s' % filepath)

    def copy(self, source_assets_path: str, filepath: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Copy images from source_path.\n\n        Args:\n            source_assets_path: str. The path to the source entity's assets\n                folder.\n            filepath: str. The path to the relevant file within the entity's\n                assets folder.\n        "
        source_file_url = '%s/%s' % (source_assets_path, filepath)
        storage_services.copy(self._bucket_name, source_file_url, self._get_gcs_file_url(filepath))

    def listdir(self, dir_name: str) -> List[str]:
        if False:
            while True:
                i = 10
        "Lists all files in a directory.\n\n        Args:\n            dir_name: str. The directory whose files should be listed. This\n                should not start with '/' or end with '/'.\n\n        Returns:\n            list(str). A lexicographically-sorted list of filenames.\n\n        Raises:\n            OSError. The directory name starts or ends with '/'.\n        "
        self._check_filepath(dir_name)
        if dir_name.startswith('/') or dir_name.endswith('/'):
            raise IOError('The dir_name should not start with / or end with / : %s' % dir_name)
        if dir_name and (not dir_name.endswith('/')):
            dir_name += '/'
        assets_path = '%s/' % self._assets_path
        prefix = utils.vfs_construct_path(self._assets_path, dir_name)
        blobs_in_dir = storage_services.listdir(self._bucket_name, prefix)
        return [blob.name.replace(assets_path, '') for blob in blobs_in_dir]

def save_original_and_compressed_versions_of_image(filename: str, entity_type: str, entity_id: str, original_image_content: bytes, filename_prefix: str, image_is_compressible: bool) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Saves the three versions of the image file.\n\n    Args:\n        filename: str. The name of the image file.\n        entity_type: str. The type of the entity.\n        entity_id: str. The id of the entity.\n        original_image_content: bytes. The content of the original image.\n        filename_prefix: str. The string to prefix to the filename.\n        image_is_compressible: bool. Whether the image can be compressed or\n            not.\n    '
    filepath = '%s/%s' % (filename_prefix, filename)
    filename_wo_filetype = filename[:filename.rfind('.')]
    filetype = filename[filename.rfind('.') + 1:]
    compressed_image_filename = '%s_compressed.%s' % (filename_wo_filetype, filetype)
    compressed_image_filepath = '%s/%s' % (filename_prefix, compressed_image_filename)
    micro_image_filename = '%s_micro.%s' % (filename_wo_filetype, filetype)
    micro_image_filepath = '%s/%s' % (filename_prefix, micro_image_filename)
    fs = GcsFileSystem(entity_type, entity_id)
    if image_is_compressible:
        compressed_image_content = image_services.compress_image(original_image_content, 0.8)
        micro_image_content = image_services.compress_image(original_image_content, 0.7)
    else:
        compressed_image_content = original_image_content
        micro_image_content = original_image_content
    mimetype = 'image/svg+xml' if filetype == 'svg' else 'image/%s' % filetype
    if not fs.isfile(filepath):
        fs.commit(filepath, original_image_content, mimetype=mimetype)
    if not fs.isfile(compressed_image_filepath):
        fs.commit(compressed_image_filepath, compressed_image_content, mimetype=mimetype)
    if not fs.isfile(micro_image_filepath):
        fs.commit(micro_image_filepath, micro_image_content, mimetype=mimetype)

def save_classifier_data(exp_id: str, job_id: str, classifier_data_proto: text_classifier_pb2.TextClassifierFrozenModel) -> None:
    if False:
        while True:
            i = 10
    'Store classifier model data in a file.\n\n    Args:\n        exp_id: str. The id of the exploration.\n        job_id: str. The id of the classifier training job model.\n        classifier_data_proto: Object. Protobuf object of the classifier data\n            to be stored.\n    '
    filepath = '%s-classifier-data.pb.xz' % job_id
    fs = GcsFileSystem(feconf.ENTITY_TYPE_EXPLORATION, exp_id)
    content = utils.compress_to_zlib(classifier_data_proto.SerializeToString())
    fs.commit(filepath, content, mimetype='application/octet-stream')

def delete_classifier_data(exp_id: str, job_id: str) -> None:
    if False:
        while True:
            i = 10
    'Delete the classifier data from file.\n\n    Args:\n        exp_id: str. The id of the exploration.\n        job_id: str. The id of the classifier training job model.\n    '
    filepath = '%s-classifier-data.pb.xz' % job_id
    fs = GcsFileSystem(feconf.ENTITY_TYPE_EXPLORATION, exp_id)
    if fs.isfile(filepath):
        fs.delete(filepath)

def copy_images(source_entity_type: str, source_entity_id: str, destination_entity_type: str, destination_entity_id: str, filenames: List[str]) -> None:
    if False:
        i = 10
        return i + 15
    'Copy images from source to destination.\n\n    Args:\n        source_entity_type: str. The entity type of the source.\n        source_entity_id: str. The type of the source entity.\n        destination_entity_id: str. The id of the destination entity.\n        destination_entity_type: str. The entity type of the destination.\n        filenames: list(str). The list of filenames to copy.\n    '
    source_fs = GcsFileSystem(source_entity_type, source_entity_id)
    destination_fs = GcsFileSystem(destination_entity_type, destination_entity_id)
    for filename in filenames:
        filename_wo_filetype = filename[:filename.rfind('.')]
        filetype = filename[filename.rfind('.') + 1:]
        compressed_image_filename = '%s_compressed.%s' % (filename_wo_filetype, filetype)
        micro_image_filename = '%s_micro.%s' % (filename_wo_filetype, filetype)
        destination_fs.copy(source_fs.assets_path, 'image/%s' % filename)
        destination_fs.copy(source_fs.assets_path, 'image/%s' % compressed_image_filename)
        destination_fs.copy(source_fs.assets_path, 'image/%s' % micro_image_filename)