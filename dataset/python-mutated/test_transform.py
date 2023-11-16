import pathlib
import yaml
from metadata_service.models import transform
from metadata_service.models.generated.ConnectorMetadataDefinitionV0 import ConnectorMetadataDefinitionV0

def get_all_dict_key_paths(dict_to_traverse, key_path=''):
    if False:
        for i in range(10):
            print('nop')
    'Get all paths to keys in a dict.\n\n    Args:\n        dict_to_traverse (dict): A dict.\n\n    Returns:\n        list: List of paths to keys in the dict. e.g ["data.name", "data.version", "data.meta.url""]\n    '
    if not isinstance(dict_to_traverse, dict):
        return [key_path]
    key_paths = []
    for (key, value) in dict_to_traverse.items():
        new_key_path = f'{key_path}.{key}' if key_path else key
        key_paths += get_all_dict_key_paths(value, new_key_path)
    return key_paths

def have_same_keys(dict1, dict2):
    if False:
        while True:
            i = 10
    'Check if two dicts have the same keys.\n\n    Args:\n        dict1 (dict): A dict.\n        dict2 (dict): A dict.\n\n    Returns:\n        bool: True if the dicts have the same keys, False otherwise.\n    '
    return set(get_all_dict_key_paths(dict1)) == set(get_all_dict_key_paths(dict2))

def test_transform_to_json_does_not_mutate_keys(valid_metadata_upload_files, valid_metadata_yaml_files):
    if False:
        i = 10
        return i + 15
    all_valid_metadata_files = valid_metadata_upload_files + valid_metadata_yaml_files
    for file_path in all_valid_metadata_files:
        metadata_file_path = pathlib.Path(file_path)
        original_yaml_text = metadata_file_path.read_text()
        metadata_yaml_dict = yaml.safe_load(original_yaml_text)
        metadata = ConnectorMetadataDefinitionV0.parse_obj(metadata_yaml_dict)
        metadata_json_dict = transform.to_json_sanitized_dict(metadata)
        new_yaml_text = yaml.safe_dump(metadata_json_dict, sort_keys=False)
        new_yaml_dict = yaml.safe_load(new_yaml_text)
        assert have_same_keys(metadata_yaml_dict, new_yaml_dict)