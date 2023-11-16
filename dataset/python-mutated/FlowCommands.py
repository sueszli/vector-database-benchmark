"""
This file contains the implementations of undoable actions for FlowView.
"""
from qtpy.QtCore import QObject, QPointF
from qtpy.QtWidgets import QUndoCommand
from .drawings.DrawingObject import DrawingObject
from .nodes.NodeItem import NodeItem
from typing import Tuple
from ryvencore.NodePort import NodePort, NodeInput, NodeOutput

class FlowUndoCommand(QObject, QUndoCommand):
    """
    The main difference to normal QUndoCommands is the activate feature. This allows the flow widget to add the
    undo command to the undo stack before redo() is called. This is important since some of these commands can cause
    other commands to be added while they are performing redo(), so to prevent those commands to be added to the
    undo stack before the parent command, it is here blocked at first.
    """

    def __init__(self, flow_view):
        if False:
            i = 10
            return i + 15
        self.flow_view = flow_view
        self.flow = flow_view.flow
        self._activated = False
        QObject.__init__(self)
        QUndoCommand.__init__(self)

    def activate(self):
        if False:
            while True:
                i = 10
        self._activated = True
        self.redo()

    def redo(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not self._activated:
            return
        else:
            self.redo_()

    def undo(self) -> None:
        if False:
            return 10
        self.undo_()

    def redo_(self):
        if False:
            return 10
        'subclassed'
        pass

    def undo_(self):
        if False:
            return 10
        'subclassed'
        pass

class MoveComponents_Command(FlowUndoCommand):

    def __init__(self, flow_view, items_list, p_from, p_to):
        if False:
            return 10
        super(MoveComponents_Command, self).__init__(flow_view)
        self.items_list = items_list
        self.p_from = p_from
        self.p_to = p_to
        self.last_item_group_pos = p_to

    def undo_(self):
        if False:
            print('Hello World!')
        items_group = self.items_group()
        items_group.setPos(self.p_from)
        self.last_item_group_pos = items_group.pos()
        self.destroy_items_group(items_group)

    def redo_(self):
        if False:
            while True:
                i = 10
        items_group = self.items_group()
        items_group.setPos(self.p_to - self.last_item_group_pos)
        self.destroy_items_group(items_group)

    def items_group(self):
        if False:
            print('Hello World!')
        return self.flow_view.scene().createItemGroup(self.items_list)

    def destroy_items_group(self, items_group):
        if False:
            return 10
        self.flow_view.scene().destroyItemGroup(items_group)

class PlaceNode_Command(FlowUndoCommand):

    def __init__(self, flow_view, node_class, pos):
        if False:
            return 10
        super().__init__(flow_view)
        self.node_class = node_class
        self.node = None
        self.item_pos = pos

    def undo_(self):
        if False:
            return 10
        self.flow.remove_node(self.node)

    def redo_(self):
        if False:
            return 10
        if self.node:
            self.flow.add_node(self.node)
        else:
            self.node = self.flow.create_node(self.node_class)

class PlaceDrawing_Command(FlowUndoCommand):

    def __init__(self, flow_view, posF, drawing):
        if False:
            i = 10
            return i + 15
        super().__init__(flow_view)
        self.drawing = drawing
        self.drawing_obj_place_pos = posF
        self.drawing_obj_pos = self.drawing_obj_place_pos

    def undo_(self):
        if False:
            while True:
                i = 10
        self.drawing_obj_pos = self.drawing.pos()
        self.flow_view.remove_component(self.drawing)

    def redo_(self):
        if False:
            for i in range(10):
                print('nop')
        self.flow_view.add_drawing(self.drawing, self.drawing_obj_pos)

class RemoveComponents_Command(FlowUndoCommand):

    def __init__(self, flow_view, items):
        if False:
            while True:
                i = 10
        super().__init__(flow_view)
        self.items = items
        self.broken_connections = []
        self.internal_connections = set()
        self.node_items = []
        self.nodes = []
        self.drawings = []
        for i in self.items:
            if isinstance(i, NodeItem):
                self.node_items.append(i)
                self.nodes.append(i.node)
            elif isinstance(i, DrawingObject):
                self.drawings.append(i)
        for n in self.nodes:
            for i in n.inputs:
                cp = n.flow.connected_output(i)
                if cp is not None:
                    cn = cp.node
                    if cn not in self.nodes:
                        self.broken_connections.append((cp, i))
                    else:
                        self.internal_connections.add((cp, i))
            for o in n.outputs:
                for cp in n.flow.connected_inputs(o):
                    cn = cp.node
                    if cn not in self.nodes:
                        self.broken_connections.append((o, cp))
                    else:
                        self.internal_connections.add((o, cp))

    def undo_(self):
        if False:
            return 10
        for n in self.nodes:
            self.flow.add_node(n)
        for d in self.drawings:
            self.flow_view.add_drawing(d)
        self.restore_broken_connections()
        self.restore_internal_connections()

    def redo_(self):
        if False:
            print('Hello World!')
        self.remove_broken_connections()
        self.remove_internal_connections()
        for n in self.nodes:
            self.flow.remove_node(n)
        for d in self.drawings:
            self.flow_view.remove_drawing(d)

    def restore_internal_connections(self):
        if False:
            i = 10
            return i + 15
        for c in self.internal_connections:
            self.flow.add_connection(c)

    def remove_internal_connections(self):
        if False:
            while True:
                i = 10
        for c in self.internal_connections:
            self.flow.remove_connection(c)

    def restore_broken_connections(self):
        if False:
            for i in range(10):
                print('nop')
        for c in self.broken_connections:
            self.flow.add_connection(c)

    def remove_broken_connections(self):
        if False:
            for i in range(10):
                print('nop')
        for c in self.broken_connections:
            self.flow.remove_connection(c)

class ConnectPorts_Command(FlowUndoCommand):

    def __init__(self, flow_view, out, inp):
        if False:
            print('Hello World!')
        super().__init__(flow_view)
        self.out = out
        self.inp = inp
        self.connection = None
        self.connecting = True
        for i in flow_view.flow.connected_inputs(out):
            if i == self.inp:
                self.connection = (out, i)
                self.connecting = False

    def undo_(self):
        if False:
            while True:
                i = 10
        if self.connecting:
            self.flow.remove_connection(self.connection)
        else:
            self.flow.add_connection(self.connection)

    def redo_(self):
        if False:
            print('Hello World!')
        if self.connecting:
            if self.connection:
                self.flow.add_connection(self.connection)
            else:
                self.connection = self.flow.connect_nodes(self.out, self.inp)
        else:
            self.flow.remove_connection(self.connection)

class Paste_Command(FlowUndoCommand):

    def __init__(self, flow_view, data, offset_for_middle_pos):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(flow_view)
        self.data = data
        self.modify_data_positions(offset_for_middle_pos)
        self.pasted_components = None

    def modify_data_positions(self, offset):
        if False:
            print('Hello World!')
        "adds the offset to the components' positions in data"
        for node in self.data['nodes']:
            node['pos x'] = node['pos x'] + offset.x()
            node['pos y'] = node['pos y'] + offset.y()
        for drawing in self.data['drawings']:
            drawing['pos x'] = drawing['pos x'] + offset.x()
            drawing['pos y'] = drawing['pos y'] + offset.y()

    def redo_(self):
        if False:
            return 10
        if self.pasted_components is None:
            self.pasted_components = {}
            self.create_drawings()
            (self.pasted_components['nodes'], self.pasted_components['connections']) = self.flow.load_components(nodes_data=self.data['nodes'], conns_data=self.data['connections'], output_data=self.data['output data'])
            self.select_new_components_in_view()
        else:
            self.add_existing_components()

    def undo_(self):
        if False:
            for i in range(10):
                print('nop')
        for c in self.pasted_components['connections']:
            self.flow.remove_connection(c)
        for n in self.pasted_components['nodes']:
            self.flow.remove_node(n)
        for d in self.pasted_components['drawings']:
            self.flow_view.remove_drawing(d)

    def add_existing_components(self):
        if False:
            for i in range(10):
                print('nop')
        for n in self.pasted_components['nodes']:
            self.flow.add_node(n)
        for c in self.pasted_components['connections']:
            self.flow.add_connection(c)
        for d in self.pasted_components['drawings']:
            self.flow_view.add_drawing(d)
        self.select_new_components_in_view()

    def select_new_components_in_view(self):
        if False:
            print('Hello World!')
        self.flow_view.clear_selection()
        for d in self.pasted_components['drawings']:
            d: DrawingObject
            d.setSelected(True)
        for n in self.pasted_components['nodes']:
            n: NodeItem
            ni: NodeItem = self.flow_view.node_items[n]
            ni.setSelected(True)

    def create_drawings(self):
        if False:
            while True:
                i = 10
        drawings = []
        for d in self.data['drawings']:
            new_drawing = self.flow_view.create_drawing(d)
            self.flow_view.add_drawing(new_drawing, posF=QPointF(d['pos x'], d['pos y']))
            drawings.append(new_drawing)
        self.pasted_components['drawings'] = drawings