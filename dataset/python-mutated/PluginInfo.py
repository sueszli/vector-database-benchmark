from UM.PluginObject import PluginObject

class PluginInfo(PluginObject):
    __instance = None

    def __init__(self, *args, **kwags):
        if False:
            i = 10
            return i + 15
        if PluginInfo.__instance is not None:
            raise RuntimeError("Try to create singleton '%s' more than once" % self.__class__.__name__)
        super().__init__(*args, **kwags)
        PluginInfo.__instance = self

    @classmethod
    def getInstance(cls, *args, **kwargs) -> 'PluginInfo':
        if False:
            return 10
        return cls.__instance