"""
Defines Scene Graph tree UI
"""
from .SceneGraphUIBase import SceneGraphUIBase

class SceneGraphUI(SceneGraphUIBase):

    def __init__(self, parent, editor):
        if False:
            print('Hello World!')
        SceneGraphUIBase.__init__(self, parent, editor)

    def populateExtraMenu(self):
        if False:
            return 10
        pass