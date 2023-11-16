from UM.PluginObject import PluginObject

class ProfileWriter(PluginObject):
    """Base class for profile writer plugins.

    This class defines a write() function to write profiles to files with.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        "Initialises the profile writer.\n\n        This currently doesn't do anything since the writer is basically static.\n        "
        super().__init__()

    def write(self, path, profiles):
        if False:
            print('Hello World!')
        "Writes a profile to the specified file path.\n\n        The profile writer may write its own file format to the specified file.\n\n        :param path: :type{string} The file to output to.\n        :param profiles: :type{Profile} or :type{List} The profile(s) to write to the file.\n        :return: True if the writing was successful, or False  if it wasn't.\n        "
        raise NotImplementedError('Profile writer plugin was not correctly implemented. No write was specified.')