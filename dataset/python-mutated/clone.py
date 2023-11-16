from .port import Port, Element

class PortClone(Port):

    def __init__(self, parent, direction, master, name, key):
        if False:
            print('Hello World!')
        Element.__init__(self, parent)
        self.master_port = master
        self.name = name
        self.key = key
        self.multiplicity = 1

    def __getattr__(self, item):
        if False:
            print('Hello World!')
        return getattr(self.master_port, item)

    def add_clone(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def remove_clone(self, port):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()