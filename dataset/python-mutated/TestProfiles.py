from unittest.mock import MagicMock
import configparser
import os
import os.path
import pytest
from UM.FastConfigParser import FastConfigParser
from cura.CuraApplication import CuraApplication
from UM.Settings.DefinitionContainer import DefinitionContainer
from UM.Settings.InstanceContainer import InstanceContainer
from UM.VersionUpgradeManager import VersionUpgradeManager

def collectAllQualities():
    if False:
        print('Hello World!')
    result = []
    for (root, directories, filenames) in os.walk(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'quality'))):
        for filename in filenames:
            result.append(os.path.join(root, filename))
    return result

def collecAllDefinitionIds():
    if False:
        for i in range(10):
            print('nop')
    result = []
    for (root, directories, filenames) in os.walk(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'definitions'))):
        for filename in filenames:
            result.append(os.path.basename(filename).split('.')[0])
    return result

def collectAllSettingIds():
    if False:
        return 10
    VersionUpgradeManager._VersionUpgradeManager__instance = VersionUpgradeManager(MagicMock())
    CuraApplication._initializeSettingDefinitions()
    definition_container = DefinitionContainer('whatever')
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'definitions', 'fdmprinter.def.json'), encoding='utf-8') as data:
        definition_container.deserialize(data.read())
    return definition_container.getAllKeys()

def collectAllVariants():
    if False:
        for i in range(10):
            print('nop')
    result = []
    for (root, directories, filenames) in os.walk(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'variants'))):
        for filename in filenames:
            result.append(os.path.join(root, filename))
    return result

def collectAllIntents():
    if False:
        return 10
    result = []
    for (root, directories, filenames) in os.walk(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'intent'))):
        for filename in filenames:
            result.append(os.path.join(root, filename))
    return result
all_definition_ids = collecAllDefinitionIds()
quality_filepaths = collectAllQualities()
all_setting_ids = collectAllSettingIds()
variant_filepaths = collectAllVariants()
intent_filepaths = collectAllIntents()

def test_uniqueID():
    if False:
        i = 10
        return i + 15
    "Check if the ID's from the qualities, variants & intents are unique."
    all_paths = quality_filepaths + variant_filepaths + intent_filepaths
    all_ids = {}
    for path in all_paths:
        profile_id = os.path.basename(path)
        profile_id = profile_id.replace('.inst.cfg', '')
        if profile_id not in all_ids:
            all_ids[profile_id] = []
        all_ids[profile_id].append(path)
    duplicated_ids_with_paths = {profile_id: paths for (profile_id, paths) in all_ids.items() if len(paths) > 1}
    if len(duplicated_ids_with_paths.keys()) == 0:
        return
    assert False, "Duplicate profile ID's were detected! Ensure that every profile ID is unique: %s" % duplicated_ids_with_paths

@pytest.mark.parametrize('file_name', quality_filepaths)
def test_validateQualityProfiles(file_name):
    if False:
        i = 10
        return i + 15
    'Attempt to load all the quality profiles.'
    try:
        with open(file_name, encoding='utf-8') as data:
            serialized = data.read()
            result = InstanceContainer._readAndValidateSerialized(serialized)
            assert InstanceContainer.getConfigurationTypeFromSerialized(serialized) == 'quality'
            assert result['general']['definition'] in all_definition_ids, 'The quality profile %s links to an unknown definition (%s)' % (file_name, result['general']['definition'])
            assert result['metadata'].get('quality_type', None) is not None
    except Exception as e:
        assert False, f'Got an Exception while reading the file [{file_name}]: {e}'

@pytest.mark.parametrize('file_name', intent_filepaths)
def test_validateIntentProfiles(file_name):
    if False:
        for i in range(10):
            print('nop')
    try:
        with open(file_name, encoding='utf-8') as f:
            serialized = f.read()
            result = InstanceContainer._readAndValidateSerialized(serialized)
            assert InstanceContainer.getConfigurationTypeFromSerialized(serialized) == 'intent', 'The intent folder must only contain intent profiles.'
            assert result['general']['definition'] in all_definition_ids, 'The definition for this intent profile must exist.'
            assert result['metadata'].get('intent_category', None) is not None, 'All intent profiles must have some intent category.'
            assert result['metadata'].get('quality_type', None) is not None, 'All intent profiles must be linked to some quality type.'
            assert result['metadata'].get('material', None) is not None, 'All intent profiles must be linked to some material.'
            assert result['metadata'].get('variant', None) is not None, 'All intent profiles must be linked to some variant.'
    except Exception as e:
        assert False, 'Got an exception while reading the file {file_name}: {err}'.format(file_name=file_name, err=str(e))

@pytest.mark.parametrize('file_name', variant_filepaths)
def test_validateVariantProfiles(file_name):
    if False:
        return 10
    'Attempt to load all the variant profiles.'
    try:
        with open(file_name, encoding='utf-8') as data:
            serialized = data.read()
            result = InstanceContainer._readAndValidateSerialized(serialized)
            assert InstanceContainer.getConfigurationTypeFromSerialized(serialized) == 'variant', "The profile %s should be of type variant, but isn't" % file_name
            assert result['general']['definition'] in all_definition_ids, "The profile %s isn't associated with a definition" % file_name
            if 'values' in result:
                variant_setting_keys = set(result['values'])
                variant_setting_keys = {key for key in variant_setting_keys if not key.startswith('#')}
                has_unknown_settings = not variant_setting_keys.issubset(all_setting_ids)
                if has_unknown_settings:
                    assert False, 'The following setting(s) %s are defined in the variant %s, but not in fdmprinter.def.json' % ([key for key in variant_setting_keys if key not in all_setting_ids], file_name)
    except Exception as e:
        assert False, 'Got an exception while reading the file {file_name}: {err}'.format(file_name=file_name, err=str(e))

@pytest.mark.parametrize('file_name', quality_filepaths + variant_filepaths + intent_filepaths)
def test_versionUpToDate(file_name):
    if False:
        print('Hello World!')
    try:
        with open(file_name, encoding='utf-8') as data:
            parser = FastConfigParser(data.read())
            assert 'general' in parser
            assert 'version' in parser['general']
            assert int(parser['general']['version']) == InstanceContainer.Version, 'The version of this profile is not up to date!'
            assert 'metadata' in parser
            assert 'setting_version' in parser['metadata']
            assert int(parser['metadata']['setting_version']) == CuraApplication.SettingVersion, 'The version of this profile is not up to date!'
    except Exception as e:
        assert False, 'Got an exception while reading the file {file_name}: {err}'.format(file_name=file_name, err=str(e))