from viztracer.vizplugin import VizPluginBase

class DummyVizPlugin(VizPluginBase):

    def support_version(self):
        if False:
            i = 10
            return i + 15
        return '0.10.5'

def get_vizplugin(arg):
    if False:
        i = 10
        return i + 15
    return DummyVizPlugin()