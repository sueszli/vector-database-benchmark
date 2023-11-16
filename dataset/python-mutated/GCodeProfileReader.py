import re
import json
from typing import Optional
from UM.Settings.ContainerFormatError import ContainerFormatError
from UM.Settings.InstanceContainer import InstanceContainer
from UM.Logger import Logger
from UM.i18n import i18nCatalog
from cura.ReaderWriters.ProfileReader import ProfileReader, NoProfileException
catalog = i18nCatalog('cura')

class GCodeProfileReader(ProfileReader):
    """A class that reads profile data from g-code files.

    It reads the profile data from g-code files and stores it in a new profile.
    This class currently does not process the rest of the g-code in any way.
    """
    version = 3
    'The file format version of the serialized g-code.\n\n    It can only read settings with the same version as the version it was\n    written with. If the file format is changed in a way that breaks reverse\n    compatibility, increment this version number!\n    '
    escape_characters = {re.escape('\\\\'): '\\', re.escape('\\n'): '\n', re.escape('\\r'): '\r'}
    'Dictionary that defines how characters are escaped when embedded in\n\n    g-code.\n\n    Note that the keys of this dictionary are regex strings. The values are\n    not.\n    '

    def read(self, file_name):
        if False:
            for i in range(10):
                print('nop')
        'Reads a g-code file, loading the profile from it.\n\n        :param file_name: The name of the file to read the profile from.\n        :return: The profile that was in the specified file, if any. If the\n            specified file was no g-code or contained no parsable profile,\n            None  is returned.\n        '
        Logger.log('i', 'Attempting to read a profile from the g-code')
        if file_name.split('.')[-1] != 'gcode':
            return None
        prefix = ';SETTING_' + str(GCodeProfileReader.version) + ' '
        prefix_length = len(prefix)
        serialized = ''
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith(prefix):
                        serialized += line[prefix_length:-1]
        except IOError as e:
            Logger.log('e', 'Unable to open file %s for reading: %s', file_name, str(e))
            return None
        serialized = unescapeGcodeComment(serialized)
        serialized = serialized.strip()
        if not serialized:
            Logger.log('w', 'No custom profile to import from this g-code: %s', file_name)
            raise NoProfileException()
        try:
            json_data = json.loads(serialized)
        except Exception as e:
            Logger.log('e', 'Could not parse serialized JSON data from g-code %s, error: %s', file_name, e)
            return None
        profiles = []
        global_profile = readQualityProfileFromString(json_data['global_quality'])
        if global_profile.getMetaDataEntry('extruder', None) is not None:
            global_profile.setMetaDataEntry('extruder', None)
        profiles.append(global_profile)
        for profile_string in json_data.get('extruder_quality', []):
            profiles.append(readQualityProfileFromString(profile_string))
        return profiles

def unescapeGcodeComment(string: str) -> str:
    if False:
        i = 10
        return i + 15
    'Unescape a string which has been escaped for use in a gcode comment.\n\n    :param string: The string to unescape.\n    :return: The unescaped string.\n    '
    pattern = re.compile('|'.join(GCodeProfileReader.escape_characters.keys()))
    return pattern.sub(lambda m: GCodeProfileReader.escape_characters[re.escape(m.group(0))], string)

def readQualityProfileFromString(profile_string) -> Optional[InstanceContainer]:
    if False:
        for i in range(10):
            print('nop')
    'Read in a profile from a serialized string.\n\n    :param profile_string: The profile data in serialized form.\n    :return: The resulting Profile object or None if it could not be read.\n    '
    profile = InstanceContainer('')
    try:
        profile.deserialize(profile_string)
    except ContainerFormatError as e:
        Logger.log('e', 'Corrupt profile in this g-code file: %s', str(e))
        return None
    except Exception as e:
        Logger.log('e', 'Unable to serialise the profile: %s', str(e))
        return None
    return profile