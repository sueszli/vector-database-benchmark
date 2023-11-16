import inspect

class Input:
    name = None
    default = None
    typ = None
    source = None

    def __init__(self, name, default=inspect.Parameter.empty, typ=None):
        if False:
            return 10
        self.name = name
        self.default = default
        self.typ = typ

    def get_value(self):
        if False:
            return 10
        if not self.is_linked():
            return self.default
        return self.source.get_value()

    def has_default(self):
        if False:
            return 10
        return not self.default == inspect.Parameter.empty

    def link(self, output):
        if False:
            for i in range(10):
                print('nop')
        if not issubclass(type(output), Output):
            return
        self.source = output

    def is_linked(self):
        if False:
            i = 10
            return i + 15
        return self.source != None

    def unlink(self):
        if False:
            for i in range(10):
                print('nop')
        Input.link(self, None)

class Output:
    name = None
    typ = None
    destinations = None
    value = None

    def __init__(self, name, typ=None):
        if False:
            while True:
                i = 10
        self.name = name
        self.typ = typ
        self.destinations = []

    def get_value(self):
        if False:
            i = 10
            return i + 15
        return self.value

    def set_value(self, value):
        if False:
            while True:
                i = 10
        self.value = value

    def link(self, destination):
        if False:
            print('Hello World!')
        if not issubclass(type(destination), Input):
            return
        self.destinations.append(destination)

    def is_linked(self):
        if False:
            print('Hello World!')
        return len(self.destinations) > 0

    def unlink(self, destination=None):
        if False:
            for i in range(10):
                print('nop')
        if not destination is None:
            self.destinations.remove(destination)
            return
        self.destinations = []

class ObjectBlock:
    name = None
    FBs = None
    inputs = None
    outputs = None

    def __init__(self, name):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.FBs = {}
        self.inputs = {}
        self.outputs = {}

    def add_io(self, io):
        if False:
            for i in range(10):
                print('nop')
        if issubclass(type(io), Input):
            self.inputs[io.name] = io
        else:
            self.outputs[io.name] = io

class FunctionBlock:
    name = None
    inputs = None
    outputs = None

    def decorate_process(output_list):
        if False:
            while True:
                i = 10
        ' setup a method as a process FunctionBlock '
        '\n            input parameters can be obtained by introspection\n            outputs values (return values) are to be described with decorator\n        '

        def add_annotation(method):
            if False:
                for i in range(10):
                    print('nop')
            setattr(method, '_outputs', output_list)
            return method
        return add_annotation

    def __init__(self, name):
        if False:
            print('Hello World!')
        self.name = name
        self.inputs = {}
        self.outputs = {}

    def add_io(self, io):
        if False:
            print('Hello World!')
        if issubclass(type(io), Input):
            self.inputs[io.name] = io
        else:
            self.outputs[io.name] = io

    @decorate_process([])
    def do(self):
        if False:
            return 10
        return None

class Link:
    source = None
    destination = None

    def __init__(self, source_widget, destination_widget):
        if False:
            print('Hello World!')
        self.source = source_widget
        self.destination = destination_widget

    def unlink(self):
        if False:
            print('Hello World!')
        self.source.unlink(self.destination)
        self.destination.unlink()

class Process:
    function_blocks = None
    object_blocks = None

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.function_blocks = {}
        self.object_blocks = {}

    def add_function_block(self, function_block):
        if False:
            while True:
                i = 10
        self.function_blocks[function_block.name] = function_block

    def add_object_block(self, object_block):
        if False:
            i = 10
            return i + 15
        self.object_blocks[object_block.name] = object_block

    def do(self):
        if False:
            while True:
                i = 10
        sub_function_blocks = []
        for object_block in self.object_blocks.values():
            for function_block in object_block.FBs.values():
                sub_function_blocks.append(function_block)
        for function_block in (*self.function_blocks.values(), *sub_function_blocks):
            parameters = {}
            all_inputs_connected = True
            for IN in function_block.inputs.values():
                if not IN.is_linked() and (not IN.has_default()):
                    all_inputs_connected = False
                    continue
                parameters[IN.name] = IN.get_value()
            if not all_inputs_connected:
                continue
            output_results = function_block.do(**parameters)
            if output_results is None:
                continue
            i = 0
            for OUT in function_block.outputs.values():
                if type(output_results) in (tuple, list):
                    OUT.set_value(output_results[i])
                else:
                    OUT.set_value(output_results)
                i += 1