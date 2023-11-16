from UM.PluginObject import PluginObject

class NoProfileException(Exception):
    pass

class ProfileReader(PluginObject):
    """A type of plug-ins that reads profiles from a file.

    The profile is then stored as instance container of the type user profile.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()

    def read(self, file_name):
        if False:
            while True:
                i = 10
        'Read profile data from a file and return a filled profile.\n\n        :return: :type{Profile|Profile[]} The profile that was obtained from the file or a list of Profiles.\n        '
        raise NotImplementedError('Profile reader plug-in was not correctly implemented. The read function was not implemented.')