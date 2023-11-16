from torch._inductor.codegen import cpp, wrapper
from torch._inductor.scheduler import BaseScheduling
from torch._inductor.virtualized import V

class ExtensionWrapperCodegen(wrapper.WrapperCodeGen):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()

class ExtensionScheduling(BaseScheduling):

    def __init__(self, scheduler):
        if False:
            return 10
        self.scheduler = scheduler
        self._scheduling = cpp.CppScheduling(scheduler)

    def can_fuse_vertical(self, node1, node2):
        if False:
            return 10
        return True

    def can_fuse_horizontal(self, node1, node2):
        if False:
            print('Hello World!')
        return True

    def group_fn(self, sizes):
        if False:
            i = 10
            return i + 15
        return tuple((tuple(map(V.graph.sizevars.simplify, s)) for s in sizes))

    def codegen_template(self, template_node, epilogue_nodes):
        if False:
            print('Hello World!')
        pass

    def codegen_nodes(self, nodes):
        if False:
            i = 10
            return i + 15
        self._scheduling.codegen_nodes(nodes)

    def codegen_sync(self):
        if False:
            return 10
        pass

    def flush(self):
        if False:
            print('Hello World!')
        self._scheduling.flush()