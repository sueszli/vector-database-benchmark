"""The base classes for node custom widgets for nodes."""
from ryvencore import Data

class NodeMainWidget:
    """Base class for the main widget of a node."""

    def __init__(self, params):
        if False:
            return 10
        (self.node, self.node_item, self.node_gui) = params

    def get_state(self) -> dict:
        if False:
            while True:
                i = 10
        '\n        *VIRTUAL*\n\n        Return the state of the widget, in a (pickle) serializable format.\n        '
        data = {}
        return data

    def set_state(self, data: dict):
        if False:
            for i in range(10):
                print('nop')
        '\n        *VIRTUAL*\n\n        Set the state of the widget, where data corresponds to the dict\n        returned by get_state().\n        '
        pass

    def update_node(self):
        if False:
            for i in range(10):
                print('nop')
        self.node.update()

    def update_node_shape(self):
        if False:
            print('Hello World!')
        self.node_item.update_shape()

class NodeInputWidget:
    """Base class for the input widget of a node."""

    def __init__(self, params):
        if False:
            print('Hello World!')
        (self.input, self.input_item, self.node, self.node_gui, self.position) = params

    def get_state(self) -> dict:
        if False:
            print('Hello World!')
        '\n        *VIRTUAL*\n\n        Return the state of the widget, in a (pickle) serializable format.\n        '
        data = {}
        return data

    def set_state(self, data: dict):
        if False:
            return 10
        '\n        *VIRTUAL*\n\n        Set the state of the widget, where data corresponds to the dict\n        returned by get_state().\n        '
        pass

    def val_update_event(self, val: Data):
        if False:
            for i in range(10):
                print('nop')
        "\n        *VIRTUAL*\n\n        Called when the input's value is updated through a connection.\n        This can be used to represent the value in the widget.\n        The widget is disabled when the port is connected.\n        "
        pass

    def update_node_input(self, val: Data, silent=False):
        if False:
            i = 10
            return i + 15
        "\n        Update the input's value and update the node.\n        "
        self.input.default = val
        if not silent:
            self.input.node.update(self.node.inputs.index(self.input))

    def update_node(self):
        if False:
            print('Hello World!')
        self.node.update(self.node.inputs.index(self.input))

    def update_node_shape(self):
        if False:
            i = 10
            return i + 15
        self.node_gui.update_shape()