from queue import Queue
from typing import List, Dict, Tuple, Optional, Union
from qtpy.QtCore import QObject, Signal

class NodeGUI(QObject):
    """
    Interface class between nodes and their GUI representation.
    """
    description_html: str = None
    main_widget_class: list = None
    main_widget_pos: str = 'below ports'
    input_widget_classes: dict = {}
    init_input_widgets: dict = {}
    style: str = 'normal'
    color: str = '#c69a15'
    display_title: str = None
    icon: str = None
    updating = Signal()
    update_error = Signal(object)
    input_added = Signal(int, object)
    output_added = Signal(int, object)
    input_removed = Signal(int, object)
    output_removed = Signal(int, object)
    update_shape_triggered = Signal()
    hide_unconnected_ports_triggered = Signal()
    show_unconnected_ports_triggered = Signal()

    def __init__(self, params):
        if False:
            while True:
                i = 10
        QObject.__init__(self)
        (node, session_gui) = params
        self.node = node
        self.item = None
        self.session_gui = session_gui
        setattr(node, 'gui', self)
        self.actions = self._init_default_actions()
        if self.display_title is None:
            self.display_title = self.node.title
        self.input_widgets = {}
        for (i, widget_data) in self.init_input_widgets.items():
            self.input_widgets[self.node.inputs[i]] = widget_data
        self._next_input_widgets = Queue()
        self.error_during_update = False
        self.node.updating.sub(self._on_updating)
        self.node.update_error.sub(self._on_update_error)
        self.node.input_added.sub(self._on_new_input_added)
        self.node.output_added.sub(self._on_new_output_added)
        self.node.input_removed.sub(self._on_input_removed)
        self.node.output_removed.sub(self._on_output_removed)

    def initialized(self):
        if False:
            i = 10
            return i + 15
        '\n        *VIRTUAL*\n\n        Called after the node GUI has been fully initialized.\n        The Node has been created already (including all ports) and loaded.\n        No connections have been made to ports of the node yet.\n        '
        pass
    '\n    slots\n    '

    def _on_update_error(self, e):
        if False:
            while True:
                i = 10
        self.update_error.emit(e)

    def _on_updating(self, inp: int):
        if False:
            for i in range(10):
                print('nop')
        if inp != -1 and self.item.inputs[inp].widget is not None:
            o = self.node.flow.connected_output(self.node.inputs[inp])
            if o is not None:
                self.item.inputs[inp].widget.val_update_event(o.val)
        self.updating.emit()

    def _on_new_input_added(self, _, index, inp):
        if False:
            for i in range(10):
                print('nop')
        if not self._next_input_widgets.empty():
            self.input_widgets[inp] = self._next_input_widgets.get()
        self.input_added.emit(index, inp)

    def _on_new_output_added(self, _, index, out):
        if False:
            return 10
        self.output_added.emit(index, out)

    def _on_input_removed(self, _, index, inp):
        if False:
            i = 10
            return i + 15
        self.input_removed.emit(index, inp)

    def _on_output_removed(self, _, index, out):
        if False:
            i = 10
            return i + 15
        self.output_removed.emit(index, out)
    '\n    actions\n    \n    TODO: move actions to ryvencore?\n    '

    def _init_default_actions(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the default actions every node should have\n        '
        return {'update shape': {'method': self.update_shape}, 'hide unconnected ports': {'method': self.hide_unconnected_ports}, 'change title': {'method': self.change_title}}

    def _deserialize_actions(self, actions_data):
        if False:
            print('Hello World!')
        '\n        Recursively reconstructs the actions dict from the serialized version\n        '

        def _transform(actions_data: dict):
            if False:
                print('Hello World!')
            "\n            Mutates the actions_data argument by replacing the method names\n            with the actual methods. Doesn't modify the original dict.\n            "
            new_actions = {}
            for (key, value) in actions_data.items():
                if key == 'method':
                    try:
                        value = getattr(self, value)
                    except AttributeError:
                        print(f'Warning: action method "{value}" not found in node "{self.node.title}", skipping.')
                elif isinstance(value, dict):
                    value = _transform(value)
                new_actions[key] = value
            return new_actions
        return _transform(actions_data)

    def _serialize_actions(self, actions):
        if False:
            i = 10
            return i + 15
        "\n        Recursively transforms the actions dict into a JSON-compatible dict\n        by replacing methods with their name. Doesn't modify the original dict.\n        "

        def _transform(actions: dict):
            if False:
                return 10
            new_actions = {}
            for (key, value) in actions.items():
                if key == 'method':
                    new_actions[key] = value.__name__
                elif isinstance(value, dict):
                    new_actions[key] = _transform(value)
                else:
                    new_actions[key] = value
            return new_actions
        return _transform(actions)
    '\n    serialization\n    '

    def data(self):
        if False:
            while True:
                i = 10
        return {'actions': self._serialize_actions(self.actions), 'display title': self.display_title}

    def load(self, data):
        if False:
            return 10
        if 'actions' in data:
            self.actions = self._deserialize_actions(data['actions'])
        if 'display title' in data:
            self.display_title = data['display title']
        if 'special actions' in data:
            self.actions = self._deserialize_actions(data['special actions'])
    '\n    GUI access methods\n    '

    def set_display_title(self, t: str):
        if False:
            i = 10
            return i + 15
        self.display_title = t
        self.update_shape()

    def flow_view(self):
        if False:
            return 10
        return self.item.flow_view

    def main_widget(self):
        if False:
            for i in range(10):
                print('nop')
        "Returns the main_widget object, or None if the item doesn't exist (yet)"
        return self.item.main_widget

    def attach_input_widgets(self, widget_names: List[str]):
        if False:
            i = 10
            return i + 15
        'Attaches the input widget to the next created input.'
        for w in widget_names:
            self._next_input_widgets.queue(w)

    def input_widget(self, index: int):
        if False:
            return 10
        'Returns a reference to the widget of the corresponding input'
        return self.item.inputs[index].widget

    def session_stylesheet(self):
        if False:
            for i in range(10):
                print('nop')
        return self.session_gui.design.global_stylesheet

    def update_shape(self):
        if False:
            while True:
                i = 10
        'Causes recompilation of the whole shape of the GUI item.'
        self.update_shape_triggered.emit()

    def hide_unconnected_ports(self):
        if False:
            return 10
        'Hides all ports that are not connected to anything.'
        del self.actions['hide unconnected ports']
        self.actions['show unconnected ports'] = {'method': self.show_unconnected_ports}
        self.hide_unconnected_ports_triggered.emit()

    def show_unconnected_ports(self):
        if False:
            return 10
        'Shows all ports that are not connected to anything.'
        del self.actions['show unconnected ports']
        self.actions['hide unconnected ports'] = {'method': self.hide_unconnected_ports}
        self.show_unconnected_ports_triggered.emit()

    def change_title(self):
        if False:
            i = 10
            return i + 15
        from qtpy.QtWidgets import QDialog, QVBoxLayout, QLineEdit

        class ChangeTitleDialog(QDialog):

            def __init__(self, title):
                if False:
                    return 10
                super().__init__()
                self.new_title = None
                self.setLayout(QVBoxLayout())
                self.line_edit = QLineEdit(title)
                self.layout().addWidget(self.line_edit)
                self.line_edit.returnPressed.connect(self.return_pressed)

            def return_pressed(self):
                if False:
                    while True:
                        i = 10
                self.new_title = self.line_edit.text()
                self.accept()
        d = ChangeTitleDialog(self.display_title)
        d.exec_()
        if d.new_title:
            self.set_display_title(d.new_title)