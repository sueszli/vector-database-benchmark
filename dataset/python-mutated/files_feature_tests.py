from hatch.template import File
from hatch.utils.fs import Path

class TestsPackageRoot(File):

    def __init__(self, template_config: dict, plugin_config: dict):
        if False:
            return 10
        super().__init__(Path('tests', '__init__.py'))