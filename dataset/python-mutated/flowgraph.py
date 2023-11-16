"""
Copyright 2007-2011, 2016q Free Software Foundation, Inc.
This file is part of GNU Radio

SPDX-License-Identifier: GPL-2.0-or-later

"""
import ast
import functools
import random
from shutil import which as find_executable
from itertools import count
from gi.repository import GLib, Gtk
from . import colors
from .drawable import Drawable
from .connection import DummyConnection
from .. import Actions, Constants, Utils, Bars, Dialogs, MainWindow
from ..external_editor import ExternalEditor
from ...core import Messages
from ...core.FlowGraph import FlowGraph as CoreFlowgraph

class _ContextMenu(object):
    """
    Help with drawing the right click context menu
    """

    def __init__(self, main_window):
        if False:
            while True:
                i = 10
        self._menu = Gtk.Menu.new_from_model(Bars.ContextMenu())
        self._menu.attach_to_widget(main_window)
        if Gtk.check_version(3, 22, 0) is None:
            self.popup = self._menu.popup_at_pointer

    def popup(self, event):
        if False:
            i = 10
            return i + 15
        self._menu.popup(None, None, None, None, event.button, event.time)

class FlowGraph(CoreFlowgraph, Drawable):
    """
    FlowGraph is the data structure to store graphical signal blocks,
    graphical inputs and outputs,
    and the connections between inputs and outputs.
    """

    def __init__(self, parent, **kwargs):
        if False:
            while True:
                i = 10
        '\n        FlowGraph constructor.\n        Create a list for signal blocks and connections. Connect mouse handlers.\n        '
        super(self.__class__, self).__init__(parent, **kwargs)
        Drawable.__init__(self)
        app = Gtk.Application.get_default()
        main_window = None
        for window in app.get_windows():
            if isinstance(window, MainWindow.MainWindow):
                main_window = window
                break
        self.drawing_area = None
        self.element_moved = False
        self.mouse_pressed = False
        self.press_coor = (0, 0)
        self.selected_elements = set()
        self._old_selected_port = None
        self._new_selected_port = None
        self.element_under_mouse = None
        self._context_menu = _ContextMenu(main_window)
        self.get_context_menu = lambda : self._context_menu
        self._new_connection = None
        self._elements_to_draw = []
        self._external_updaters = {}

    def _get_unique_id(self, base_id=''):
        if False:
            print('Hello World!')
        '\n        Get a unique id starting with the base id.\n\n        Args:\n            base_id: the id starts with this and appends a count\n\n        Returns:\n            a unique id\n        '
        block_ids = set((b.name for b in self.blocks))
        for index in count():
            block_id = '{}_{}'.format(base_id, index)
            if block_id not in block_ids:
                break
        return block_id

    def install_external_editor(self, param, parent=None):
        if False:
            i = 10
            return i + 15
        target = (param.parent_block.name, param.key)
        if target in self._external_updaters:
            editor = self._external_updaters[target]
        else:
            config = self.parent_platform.config
            editor = find_executable(config.editor) or Dialogs.choose_editor(parent, config)
            if not editor:
                return
            updater = functools.partial(self.handle_external_editor_change, target=target)
            editor = self._external_updaters[target] = ExternalEditor(editor=editor, name=target[0], value=param.get_value(), callback=functools.partial(GLib.idle_add, updater))
            editor.start()
        try:
            editor.open_editor()
        except Exception as e:
            Messages.send('>>> Error opening an external editor. Please select a different editor.\n')
            self.parent_platform.config.editor = ''
            self.remove_external_editor(target=target)

    def remove_external_editor(self, target=None, param=None):
        if False:
            print('Hello World!')
        if target is None:
            target = (param.parent_block.name, param.key)
        if target in self._external_updaters:
            self._external_updaters[target].stop()
            del self._external_updaters[target]

    def handle_external_editor_change(self, new_value, target):
        if False:
            i = 10
            return i + 15
        try:
            (block_id, param_key) = target
            self.get_block(block_id).params[param_key].set_value(new_value)
        except (IndexError, ValueError):
            self.remove_external_editor(target=target)
            return
        Actions.EXTERNAL_UPDATE()

    def add_new_block(self, key, coor=None):
        if False:
            while True:
                i = 10
        '\n        Add a block of the given key to this flow graph.\n\n        Args:\n            key: the block key\n            coor: an optional coordinate or None for random\n        '
        id = self._get_unique_id(key)
        scroll_pane = self.drawing_area.get_parent().get_parent()
        h_adj = scroll_pane.get_hadjustment()
        v_adj = scroll_pane.get_vadjustment()
        if coor is None:
            coor = (int(random.uniform(0.25, 0.75) * h_adj.get_page_size() + h_adj.get_value()), int(random.uniform(0.25, 0.75) * v_adj.get_page_size() + v_adj.get_value()))
        block = self.new_block(key)
        block.coordinate = coor
        block.params['id'].set_value(id)
        Actions.ELEMENT_CREATE()
        return id

    def make_connection(self):
        if False:
            i = 10
            return i + 15
        'this selection and the last were ports, try to connect them'
        if self._new_connection and self._new_connection.has_real_sink:
            self._old_selected_port = self._new_connection.source_port
            self._new_selected_port = self._new_connection.sink_port
        if self._old_selected_port and self._new_selected_port:
            try:
                self.connect(self._old_selected_port, self._new_selected_port)
                Actions.ELEMENT_CREATE()
            except Exception as e:
                Messages.send_fail_connection(e)
            self._old_selected_port = None
            self._new_selected_port = None
            return True
        return False

    def update(self):
        if False:
            i = 10
            return i + 15
        '\n        Call the top level rewrite and validate.\n        Call the top level create labels and shapes.\n        '
        self.rewrite()
        self.validate()
        self.update_elements_to_draw()
        self.create_labels()
        self.create_shapes()

    def reload(self):
        if False:
            return 10
        '\n        Reload flow-graph (with updated blocks)\n\n        Args:\n            page: the page to reload (None means current)\n        Returns:\n            False if some error occurred during import\n        '
        success = False
        data = self.export_data()
        if data:
            self.unselect()
            success = self.import_data(data)
            self.update()
        return success

    def copy_to_clipboard(self):
        if False:
            print('Hello World!')
        '\n        Copy the selected blocks and connections into the clipboard.\n\n        Returns:\n            the clipboard\n        '
        blocks = list(self.selected_blocks())
        if not blocks:
            return None
        (x_min, y_min) = blocks[0].coordinate
        for block in blocks:
            (x, y) = block.coordinate
            x_min = min(x, x_min)
            y_min = min(y, y_min)
        connections = list(filter(lambda c: c.source_block in blocks and c.sink_block in blocks, self.connections))
        clipboard = ((x_min, y_min), [block.export_data() for block in blocks], [connection.export_data() for connection in connections])
        return clipboard

    def paste_from_clipboard(self, clipboard):
        if False:
            return 10
        '\n        Paste the blocks and connections from the clipboard.\n\n        Args:\n            clipboard: the nested data of blocks, connections\n        '
        ((x_min, y_min), blocks_n, connections_n) = clipboard
        scroll_pane = self.drawing_area.get_parent().get_parent()
        h_adj = scroll_pane.get_hadjustment()
        v_adj = scroll_pane.get_vadjustment()
        x_off = h_adj.get_value() - x_min + h_adj.get_page_size() / 4
        y_off = v_adj.get_value() - y_min + v_adj.get_page_size() / 4
        if len(self.get_elements()) <= 1:
            (x_off, y_off) = (0, 0)
        pasted_blocks = {}
        for block_n in blocks_n:
            block_key = block_n.get('id')
            if block_key == 'options':
                continue
            block_name = block_n.get('name')
            if block_name in (blk.name for blk in self.blocks):
                block_n = block_n.copy()
                block_n['name'] = self._get_unique_id(block_name)
            block = self.new_block(block_key)
            if not block:
                continue
            block.import_data(**block_n)
            pasted_blocks[block_name] = block
            block.move((x_off, y_off))
            while any((Utils.align_to_grid(block.coordinate) == Utils.align_to_grid(other.coordinate) for other in self.blocks if other is not block)):
                block.move((Constants.CANVAS_GRID_SIZE, Constants.CANVAS_GRID_SIZE))
                x_off += Constants.CANVAS_GRID_SIZE
                y_off += Constants.CANVAS_GRID_SIZE
        self.selected_elements = set(pasted_blocks.values())
        self.update()
        for (src_block, src_port, dst_block, dst_port) in connections_n:
            source = pasted_blocks[src_block].get_source(src_port)
            sink = pasted_blocks[dst_block].get_sink(dst_port)
            connection = self.connect(source, sink)
            self.selected_elements.add(connection)

    def type_controller_modify_selected(self, direction):
        if False:
            while True:
                i = 10
        '\n        Change the registered type controller for the selected signal blocks.\n\n        Args:\n            direction: +1 or -1\n\n        Returns:\n            true for change\n        '
        return any([sb.type_controller_modify(direction) for sb in self.selected_blocks()])

    def port_controller_modify_selected(self, direction):
        if False:
            return 10
        '\n        Change port controller for the selected signal blocks.\n\n        Args:\n            direction: +1 or -1\n\n        Returns:\n            true for changed\n        '
        return any([sb.port_controller_modify(direction) for sb in self.selected_blocks()])

    def change_state_selected(self, new_state):
        if False:
            i = 10
            return i + 15
        '\n        Enable/disable the selected blocks.\n\n        Args:\n            new_state: a block state\n\n        Returns:\n            true if changed\n        '
        changed = False
        for block in self.selected_blocks():
            changed |= block.state != new_state
            block.state = new_state
        return changed

    def move_selected(self, delta_coordinate):
        if False:
            for i in range(10):
                print('nop')
        '\n        Move the element and by the change in coordinates.\n\n        Args:\n            delta_coordinate: the change in coordinates\n        '
        blocks = list(self.selected_blocks())
        if not blocks:
            return
        (min_x, min_y) = self.selected_block.coordinate
        for selected_block in blocks:
            (x, y) = selected_block.coordinate
            (min_x, min_y) = (min(min_x, x), min(min_y, y))
        delta_coordinate = (max(delta_coordinate[0], -min_x), max(delta_coordinate[1], -min_y))
        for selected_block in blocks:
            selected_block.move(delta_coordinate)
            self.element_moved = True

    def align_selected(self, calling_action=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Align the selected blocks.\n\n        Args:\n            calling_action: the action initiating the alignment\n\n        Returns:\n            True if changed, otherwise False\n        '
        blocks = list(self.selected_blocks())
        if calling_action is None or not blocks:
            return False
        (min_x, min_y) = (max_x, max_y) = blocks[0].coordinate
        for selected_block in blocks:
            (x, y) = selected_block.coordinate
            (min_x, min_y) = (min(min_x, x), min(min_y, y))
            x += selected_block.width
            y += selected_block.height
            (max_x, max_y) = (max(max_x, x), max(max_y, y))
        (ctr_x, ctr_y) = ((max_x + min_x) / 2, (max_y + min_y) / 2)
        transform = {Actions.BLOCK_VALIGN_TOP: lambda x, y, w, h: (x, min_y), Actions.BLOCK_VALIGN_MIDDLE: lambda x, y, w, h: (x, ctr_y - h / 2), Actions.BLOCK_VALIGN_BOTTOM: lambda x, y, w, h: (x, max_y - h), Actions.BLOCK_HALIGN_LEFT: lambda x, y, w, h: (min_x, y), Actions.BLOCK_HALIGN_CENTER: lambda x, y, w, h: (ctr_x - w / 2, y), Actions.BLOCK_HALIGN_RIGHT: lambda x, y, w, h: (max_x - w, y)}.get(calling_action, lambda *args: args)
        for selected_block in blocks:
            (x, y) = selected_block.coordinate
            (w, h) = (selected_block.width, selected_block.height)
            selected_block.coordinate = transform(x, y, w, h)
        return True

    def rotate_selected(self, rotation):
        if False:
            print('Hello World!')
        '\n        Rotate the selected blocks by multiples of 90 degrees.\n\n        Args:\n            rotation: the rotation in degrees\n\n        Returns:\n            true if changed, otherwise false.\n        '
        if not any(self.selected_blocks()):
            return False
        (min_x, min_y) = (max_x, max_y) = self.selected_block.coordinate
        for selected_block in self.selected_blocks():
            selected_block.rotate(rotation)
            (x, y) = selected_block.coordinate
            (min_x, min_y) = (min(min_x, x), min(min_y, y))
            (max_x, max_y) = (max(max_x, x), max(max_y, y))
        (ctr_x, ctr_y) = ((max_x + min_x) / 2, (max_y + min_y) / 2)
        for selected_block in self.selected_blocks():
            (x, y) = selected_block.coordinate
            (x, y) = Utils.get_rotated_coordinate((x - ctr_x, y - ctr_y), rotation)
            selected_block.coordinate = (x + ctr_x, y + ctr_y)
        return True

    def remove_selected(self):
        if False:
            return 10
        '\n        Remove selected elements\n\n        Returns:\n            true if changed.\n        '
        changed = False
        for selected_element in self.selected_elements:
            self.remove_element(selected_element)
            changed = True
        return changed

    def update_selected(self):
        if False:
            while True:
                i = 10
        '\n        Remove deleted elements from the selected elements list.\n        Update highlighting so only the selected are highlighted.\n        '
        selected_elements = self.selected_elements
        elements = self.get_elements()
        for selected in list(selected_elements):
            if selected in elements:
                continue
            selected_elements.remove(selected)
        if self._old_selected_port and self._old_selected_port.parent not in elements:
            self._old_selected_port = None
        if self._new_selected_port and self._new_selected_port.parent not in elements:
            self._new_selected_port = None
        for element in elements:
            element.highlighted = element in selected_elements

    def update_elements_to_draw(self):
        if False:
            while True:
                i = 10
        hide_disabled_blocks = Actions.TOGGLE_HIDE_DISABLED_BLOCKS.get_active()
        hide_variables = Actions.TOGGLE_HIDE_VARIABLES.get_active()

        def draw_order(elem):
            if False:
                i = 10
                return i + 15
            return (elem.highlighted, elem.is_block, elem.enabled)
        elements = sorted(self.get_elements(), key=draw_order)
        del self._elements_to_draw[:]
        for element in elements:
            if hide_disabled_blocks and (not element.enabled):
                continue
            if hide_variables and (element.is_variable or element.is_import):
                continue
            self._elements_to_draw.append(element)

    def create_labels(self, cr=None):
        if False:
            while True:
                i = 10
        for element in self._elements_to_draw:
            element.create_labels(cr)

    def create_shapes(self):
        if False:
            for i in range(10):
                print('nop')
        for element in filter(lambda x: x.is_block, self._elements_to_draw):
            element.create_shapes()
        for element in filter(lambda x: not x.is_block, self._elements_to_draw):
            element.create_shapes()

    def _drawables(self):
        if False:
            print('Hello World!')
        show_comments = Actions.TOGGLE_SHOW_BLOCK_COMMENTS.get_active()
        hide_disabled_blocks = Actions.TOGGLE_HIDE_DISABLED_BLOCKS.get_active()
        for element in self._elements_to_draw:
            if element.is_block and show_comments and element.enabled:
                yield element.draw_comment
        if self._new_connection is not None:
            yield self._new_connection.draw
        for element in self._elements_to_draw:
            if element not in self.selected_elements:
                yield element.draw
        for element in self.selected_elements:
            if element.enabled or not hide_disabled_blocks:
                yield element.draw

    def draw(self, cr):
        if False:
            for i in range(10):
                print('nop')
        'Draw blocks connections comment and select rectangle'
        for draw_element in self._drawables():
            cr.save()
            draw_element(cr)
            cr.restore()
        draw_multi_select_rectangle = self.mouse_pressed and (not self.selected_elements or self.drawing_area.ctrl_mask) and (not self._new_connection)
        if draw_multi_select_rectangle:
            (x1, y1) = self.press_coor
            (x2, y2) = self.coordinate
            (x, y) = (int(min(x1, x2)), int(min(y1, y2)))
            (w, h) = (int(abs(x1 - x2)), int(abs(y1 - y2)))
            cr.set_source_rgba(colors.HIGHLIGHT_COLOR[0], colors.HIGHLIGHT_COLOR[1], colors.HIGHLIGHT_COLOR[2], 0.5)
            cr.rectangle(x, y, w, h)
            cr.fill()
            cr.rectangle(x, y, w, h)
            cr.stroke()

    def update_selected_elements(self):
        if False:
            return 10
        '\n        Update the selected elements.\n        The update behavior depends on the state of the mouse button.\n        When the mouse button pressed the selection will change when\n        the control mask is set or the new selection is not in the current group.\n        When the mouse button is released the selection will change when\n        the mouse has moved and the control mask is set or the current group is empty.\n        Attempt to make a new connection if the old and ports are filled.\n        If the control mask is set, merge with the current elements.\n        '
        selected_elements = None
        if self.mouse_pressed:
            new_selections = self.what_is_selected(self.coordinate)
            if not new_selections:
                selected_elements = set()
            elif self.drawing_area.ctrl_mask or self.selected_elements.isdisjoint(new_selections):
                selected_elements = new_selections
            if self._old_selected_port:
                self._old_selected_port.force_show_label = False
                self.create_shapes()
                self.drawing_area.queue_draw()
            elif self._new_selected_port:
                self._new_selected_port.force_show_label = True
        elif not self.element_moved and (not self.selected_elements or self.drawing_area.ctrl_mask) and (not self._new_connection):
            selected_elements = self.what_is_selected(self.coordinate, self.press_coor)
        if self.make_connection():
            return
        if selected_elements is None:
            return
        if self.drawing_area.ctrl_mask:
            self.selected_elements ^= selected_elements
        else:
            self.selected_elements.clear()
            self.selected_elements.update(selected_elements)
        Actions.ELEMENT_SELECT()

    def what_is_selected(self, coor, coor_m=None):
        if False:
            return 10
        '\n        What is selected?\n        At the given coordinate, return the elements found to be selected.\n        If coor_m is unspecified, return a list of only the first element found to be selected:\n        Iterate though the elements backwards since top elements are at the end of the list.\n        If an element is selected, place it at the end of the list so that is is drawn last,\n        and hence on top. Update the selected port information.\n\n        Args:\n            coor: the coordinate of the mouse click\n            coor_m: the coordinate for multi select\n\n        Returns:\n            the selected blocks and connections or an empty list\n        '
        selected_port = None
        selected = set()
        for element in reversed(self._elements_to_draw):
            selected_element = element.what_is_selected(coor, coor_m)
            if not selected_element:
                continue
            if selected_element.is_port:
                if not coor_m:
                    selected_port = selected_element
                selected_element = selected_element.parent_block
            selected.add(selected_element)
            if not coor_m:
                break
        if selected_port and selected_port.is_source:
            selected.remove(selected_port.parent_block)
            self._new_connection = DummyConnection(selected_port, coordinate=coor)
            self.drawing_area.queue_draw()
        if selected_port is not self._new_selected_port:
            self._old_selected_port = self._new_selected_port
            self._new_selected_port = selected_port
        return selected

    def unselect(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set selected elements to an empty set.\n        '
        self.selected_elements.clear()

    def select_all(self):
        if False:
            while True:
                i = 10
        'Select all blocks in the flow graph'
        self.selected_elements.clear()
        self.selected_elements.update(self._elements_to_draw)

    def selected_blocks(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get a group of selected blocks.\n\n        Returns:\n            sub set of blocks in this flow graph\n        '
        return (e for e in self.selected_elements.copy() if e.is_block)

    def selected_connections(self):
        if False:
            print('Hello World!')
        '\n        Get a group of selected connections.\n\n        Returns:\n            sub set of connections in this flow graph\n        '
        return (e for e in self.selected_elements.copy() if e.is_connection)

    @property
    def selected_block(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the selected block when a block or port is selected.\n\n        Returns:\n            a block or None\n        '
        return next(self.selected_blocks(), None)

    @property
    def selected_connection(self):
        if False:
            print('Hello World!')
        '\n        Get the selected connection\n\n        Returns:\n            a connection or None\n        '
        return next(self.selected_connections(), None)

    def get_selected_elements(self):
        if False:
            i = 10
            return i + 15
        '\n        Get the group of selected elements.\n\n        Returns:\n            sub set of elements in this flow graph\n        '
        return self.selected_elements

    def get_selected_element(self):
        if False:
            return 10
        '\n        Get the selected element.\n\n        Returns:\n            a block, port, or connection or None\n        '
        return next(iter(self.selected_elements), None)

    def handle_mouse_context_press(self, coordinate, event):
        if False:
            return 10
        '\n        The context mouse button was pressed:\n        If no elements were selected, perform re-selection at this coordinate.\n        Then, show the context menu at the mouse click location.\n        '
        selections = self.what_is_selected(coordinate)
        if not selections.intersection(self.selected_elements):
            self.coordinate = coordinate
            self.mouse_pressed = True
            self.update_selected_elements()
            self.mouse_pressed = False
        if self._new_connection:
            self._new_connection = None
            self.drawing_area.queue_draw()
        self._context_menu.popup(event)

    def handle_mouse_selector_press(self, double_click, coordinate):
        if False:
            print('Hello World!')
        '\n        The selector mouse button was pressed:\n        Find the selected element. Attempt a new connection if possible.\n        Open the block params window on a double click.\n        Update the selection state of the flow graph.\n        '
        self.press_coor = coordinate
        self.coordinate = coordinate
        self.mouse_pressed = True
        if double_click:
            self.unselect()
        self.update_selected_elements()
        if double_click and self.selected_block:
            self.mouse_pressed = False
            Actions.BLOCK_PARAM_MODIFY()

    def handle_mouse_selector_release(self, coordinate):
        if False:
            while True:
                i = 10
        '\n        The selector mouse button was released:\n        Update the state, handle motion (dragging).\n        And update the selected flowgraph elements.\n        '
        self.coordinate = coordinate
        self.mouse_pressed = False
        if self.element_moved:
            Actions.BLOCK_MOVE()
            self.element_moved = False
        self.update_selected_elements()
        if self._new_connection:
            self._new_connection = None
            self.drawing_area.queue_draw()

    def handle_mouse_motion(self, coordinate):
        if False:
            print('Hello World!')
        '\n        The mouse has moved, respond to mouse dragging or notify elements\n        Move a selected element to the new coordinate.\n        Auto-scroll the scroll bars at the boundaries.\n        '
        redraw = False
        if not self.mouse_pressed or self._new_connection:
            redraw = self._handle_mouse_motion_move(coordinate)
        if self.mouse_pressed:
            redraw = redraw or self._handle_mouse_motion_drag(coordinate)
        if redraw:
            self.drawing_area.queue_draw()

    def _handle_mouse_motion_move(self, coordinate):
        if False:
            while True:
                i = 10
        redraw = False
        for element in self._elements_to_draw:
            over_element = element.what_is_selected(coordinate)
            if not over_element:
                continue
            if over_element != self.element_under_mouse:
                if self.element_under_mouse:
                    redraw |= self.element_under_mouse.mouse_out() or False
                self.element_under_mouse = over_element
                redraw |= over_element.mouse_over() or False
            break
        else:
            if self.element_under_mouse:
                redraw |= self.element_under_mouse.mouse_out() or False
                self.element_under_mouse = None
        if not Actions.TOGGLE_AUTO_HIDE_PORT_LABELS.get_active():
            return
        if redraw:
            self.create_shapes()
        return redraw

    def _handle_mouse_motion_drag(self, coordinate):
        if False:
            print('Hello World!')
        redraw = False
        if len(self.selected_elements) == 1 and self.get_selected_element().is_connection:
            Actions.ELEMENT_DELETE()
            redraw = True
        if self._new_connection:
            e = self.element_under_mouse
            if e and e.is_port and e.is_sink:
                self._new_connection.update(sink_port=self.element_under_mouse)
            else:
                self._new_connection.update(coordinate=coordinate, rotation=0)
            return True
        (x, y) = coordinate
        if not self.drawing_area.ctrl_mask:
            (X, Y) = self.coordinate
            (dX, dY) = (x - X, y - Y)
            if Actions.TOGGLE_SNAP_TO_GRID.get_active() or self.drawing_area.mod1_mask:
                (dX, dY) = (int(round(dX / Constants.CANVAS_GRID_SIZE)), int(round(dY / Constants.CANVAS_GRID_SIZE)))
                (dX, dY) = (dX * Constants.CANVAS_GRID_SIZE, dY * Constants.CANVAS_GRID_SIZE)
            else:
                (dX, dY) = (int(round(dX)), int(round(dY)))
            if dX != 0 or dY != 0:
                self.move_selected((dX, dY))
                self.coordinate = (X + dX, Y + dY)
                redraw = True
        return redraw

    def get_extents(self):
        if False:
            for i in range(10):
                print('nop')
        show_comments = Actions.TOGGLE_SHOW_BLOCK_COMMENTS.get_active()

        def sub_extents():
            if False:
                while True:
                    i = 10
            for element in self._elements_to_draw:
                yield element.get_extents()
                if element.is_block and show_comments and element.enabled:
                    yield element.get_extents_comment()
        extent = (10000000, 10000000, 0, 0)
        cmps = (min, min, max, max)
        for sub_extent in sub_extents():
            extent = [cmp(xy, e_xy) for (cmp, xy, e_xy) in zip(cmps, extent, sub_extent)]
        return tuple(extent)