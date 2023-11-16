def _pyi_rthook():
    if False:
        for i in range(10):
            print('nop')
    import os
    import sys
    os.environ['GST_REGISTRY_FORK'] = 'no'
    gst_plugin_paths = [sys._MEIPASS, os.path.join(sys._MEIPASS, 'gst-plugins')]
    os.environ['GST_PLUGIN_PATH'] = os.pathsep.join(gst_plugin_paths)
    os.environ['GST_REGISTRY'] = os.path.join(sys._MEIPASS, 'registry.bin')
    os.environ['GST_PLUGIN_SYSTEM_PATH'] = ''
_pyi_rthook()
del _pyi_rthook