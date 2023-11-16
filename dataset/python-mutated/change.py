from buildbot.process.properties import Properties
from buildbot.test.fake.state import State

class Change(State):
    project = ''
    repository = ''
    branch = ''
    category = ''
    codebase = ''
    properties = {}

    def __init__(self, **kw):
        if False:
            return 10
        super().__init__(**kw)
        props = Properties()
        props.update(self.properties, 'test')
        self.properties = props