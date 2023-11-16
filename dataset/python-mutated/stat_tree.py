import queue

class StatTree(object):

    def __init__(self, root_node):
        if False:
            while True:
                i = 10
        assert isinstance(root_node, StatNode)
        self.root_node = root_node

    def get_same_level_max_node_depth(self, query_node):
        if False:
            i = 10
            return i + 15
        if query_node.name == self.root_node.name:
            return 0
        same_level_depth = max([child.depth for child in query_node.parent.children])
        return same_level_depth

    def update_stat_nodes_granularity(self):
        if False:
            for i in range(10):
                print('nop')
        q = queue.Queue()
        q.put(self.root_node)
        while not q.empty():
            node = q.get()
            node.granularity = self.get_same_level_max_node_depth(node)
            for child in node.children:
                q.put(child)

    def get_collected_stat_nodes(self, query_granularity):
        if False:
            return 10
        self.update_stat_nodes_granularity()
        collected_nodes = []
        stack = list()
        stack.append(self.root_node)
        while len(stack) > 0:
            node = stack.pop()
            for child in reversed(node.children):
                stack.append(child)
            if node.depth == query_granularity:
                collected_nodes.append(node)
            if node.depth < query_granularity <= node.granularity:
                collected_nodes.append(node)
        return collected_nodes

class StatNode(object):

    def __init__(self, name=str(), parent=None):
        if False:
            return 10
        self._name = name
        self._input_shape = None
        self._output_shape = None
        self._parameter_quantity = 0
        self._inference_memory = 0
        self._MAdd = 0
        self._Memory = (0, 0)
        self._Flops = 0
        self._duration = 0
        self._duration_percent = 0
        self._granularity = 1
        self._depth = 1
        self.parent = parent
        self.children = list()

    @property
    def name(self):
        if False:
            while True:
                i = 10
        return self._name

    @name.setter
    def name(self, name):
        if False:
            print('Hello World!')
        self._name = name

    @property
    def granularity(self):
        if False:
            for i in range(10):
                print('nop')
        return self._granularity

    @granularity.setter
    def granularity(self, g):
        if False:
            for i in range(10):
                print('nop')
        self._granularity = g

    @property
    def depth(self):
        if False:
            i = 10
            return i + 15
        d = self._depth
        if len(self.children) > 0:
            d += max([child.depth for child in self.children])
        return d

    @property
    def input_shape(self):
        if False:
            i = 10
            return i + 15
        if len(self.children) == 0:
            return self._input_shape
        else:
            return self.children[0].input_shape

    @input_shape.setter
    def input_shape(self, input_shape):
        if False:
            return 10
        assert isinstance(input_shape, (list, tuple))
        self._input_shape = input_shape

    @property
    def output_shape(self):
        if False:
            i = 10
            return i + 15
        if len(self.children) == 0:
            return self._output_shape
        else:
            return self.children[-1].output_shape

    @output_shape.setter
    def output_shape(self, output_shape):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(output_shape, (list, tuple))
        self._output_shape = output_shape

    @property
    def parameter_quantity(self):
        if False:
            return 10
        total_parameter_quantity = self._parameter_quantity
        for child in self.children:
            total_parameter_quantity += child.parameter_quantity
        return total_parameter_quantity

    @parameter_quantity.setter
    def parameter_quantity(self, parameter_quantity):
        if False:
            print('Hello World!')
        assert parameter_quantity >= 0
        self._parameter_quantity = parameter_quantity

    @property
    def inference_memory(self):
        if False:
            print('Hello World!')
        total_inference_memory = self._inference_memory
        for child in self.children:
            total_inference_memory += child.inference_memory
        return total_inference_memory

    @inference_memory.setter
    def inference_memory(self, inference_memory):
        if False:
            i = 10
            return i + 15
        self._inference_memory = inference_memory

    @property
    def MAdd(self):
        if False:
            print('Hello World!')
        total_MAdd = self._MAdd
        for child in self.children:
            total_MAdd += child.MAdd
        return total_MAdd

    @MAdd.setter
    def MAdd(self, MAdd):
        if False:
            return 10
        self._MAdd = MAdd

    @property
    def Flops(self):
        if False:
            while True:
                i = 10
        total_Flops = self._Flops
        for child in self.children:
            total_Flops += child.Flops
        return total_Flops

    @Flops.setter
    def Flops(self, Flops):
        if False:
            return 10
        self._Flops = Flops

    @property
    def Memory(self):
        if False:
            for i in range(10):
                print('nop')
        total_Memory = self._Memory
        for child in self.children:
            total_Memory[0] += child.Memory[0]
            total_Memory[1] += child.Memory[1]
            print(total_Memory)
        return total_Memory

    @Memory.setter
    def Memory(self, Memory):
        if False:
            return 10
        assert isinstance(Memory, (list, tuple))
        self._Memory = Memory

    @property
    def duration(self):
        if False:
            i = 10
            return i + 15
        total_duration = self._duration
        for child in self.children:
            total_duration += child.duration
        return total_duration

    @duration.setter
    def duration(self, duration):
        if False:
            while True:
                i = 10
        self._duration = duration

    def find_child_index(self, child_name):
        if False:
            print('Hello World!')
        assert isinstance(child_name, str)
        index = -1
        for i in range(len(self.children)):
            if child_name == self.children[i].name:
                index = i
        return index

    def add_child(self, node):
        if False:
            while True:
                i = 10
        assert isinstance(node, StatNode)
        if self.find_child_index(node.name) == -1:
            self.children.append(node)