from os import path
from apps.core.task.coretaskstate import TaskDefinition, AdvanceVerificationOptions

class RenderingTaskDefinition(TaskDefinition):

    def __init__(self):
        if False:
            while True:
                i = 10
        TaskDefinition.__init__(self)
        self.resolution = [0, 0]
        self.renderer = None
        self.options = None
        self.main_scene_file = ''
        self.output_format = ''

    def is_valid(self):
        if False:
            return 10
        (is_valid, err) = super(RenderingTaskDefinition, self).is_valid()
        if is_valid and (not path.exists(self.main_scene_file)):
            return (False, 'Main scene file {} is not properly set'.format(self.main_scene_file))
        return (is_valid, err)

    def add_to_resources(self):
        if False:
            return 10
        super(RenderingTaskDefinition, self).add_to_resources()
        self.resources.add(path.normpath(self.main_scene_file))

class AdvanceRenderingVerificationOptions(AdvanceVerificationOptions):

    def __init__(self):
        if False:
            print('Hello World!')
        AdvanceVerificationOptions.__init__(self)
        self.probability = 0.01