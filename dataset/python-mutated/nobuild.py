from waflib import Task

def build(bld):
    if False:
        return 10

    def run(self):
        if False:
            return 10
        for x in self.outputs:
            x.write('')
    for (name, cls) in Task.classes.items():
        cls.run = run