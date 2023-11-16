from listenerlibrary import listenerlibrary

class multiple_listenerlibrary:

    def __init__(self, fail=False):
        if False:
            for i in range(10):
                print('nop')
        self.instances = [listenerlibrary(), listenerlibrary()]
        if fail:

            class NoVersionListener:

                def events_should_be_empty(self):
                    if False:
                        print('Hello World!')
                    pass
            self.instances.append(NoVersionListener())
        self.ROBOT_LIBRARY_LISTENER = self.instances

    def events_should_be(self, *expected):
        if False:
            print('Hello World!')
        for inst in self.instances:
            inst.events_should_be(*expected)

    def events_should_be_empty(self):
        if False:
            return 10
        for inst in self.instances:
            inst.events_should_be_empty()