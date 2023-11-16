from UM.Logger import Logger
from cura.ReaderWriters.ProfileWriter import ProfileWriter
import zipfile

class CuraProfileWriter(ProfileWriter):
    """Writes profiles to Cura's own profile format with config files."""

    def write(self, path, profiles):
        if False:
            i = 10
            return i + 15
        "Writes a profile to the specified file path.\n\n        :param path: :type{string} The file to output to.\n        :param profiles: :type{Profile} :type{List} The profile(s) to write to that file.\n        :return: True if the writing was successful, or\n                 False if it wasn't.\n        "
        if type(profiles) != list:
            profiles = [profiles]
        stream = open(path, 'wb')
        archive = zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED)
        try:
            for profile in profiles:
                serialized = profile.serialize()
                profile_file = zipfile.ZipInfo(profile.getId())
                archive.writestr(profile_file, serialized)
        except Exception as e:
            Logger.log('e', 'Failed to write profile to %s: %s', path, str(e))
            return False
        finally:
            archive.close()
        return True