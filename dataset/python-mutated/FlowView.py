import json
from typing import Tuple
from qtpy.QtCore import Qt, QPointF, QPoint, QRectF, QSizeF, Signal, QTimer, QTimeLine, QEvent
from qtpy.QtGui import QPainter, QPen, QColor, QKeySequence, QTabletEvent, QImage, QGuiApplication, QFont, QTouchEvent
from qtpy.QtWidgets import QGraphicsView, QGraphicsScene, QShortcut, QMenu, QGraphicsItem, QUndoStack
from ryvencore.Flow import Flow
from ryvencore.Node import Node
from ryvencore.NodePort import NodePort, NodeInput, NodeOutput
from ryvencore.InfoMsgs import InfoMsgs
from ryvencore.RC import PortObjPos
from ryvencore.utils import node_from_identifier
from ..GUIBase import GUIBase
from ..utils import *
from .FlowCommands import MoveComponents_Command, PlaceNode_Command, PlaceDrawing_Command, RemoveComponents_Command, ConnectPorts_Command, Paste_Command, FlowUndoCommand
from .FlowViewProxyWidget import FlowViewProxyWidget
from .FlowViewStylusModesWidget import FlowViewStylusModesWidget
from .node_list_widget.NodeListWidget import NodeListWidget
from .nodes.NodeGUI import NodeGUI
from .nodes.NodeItem import NodeItem
from .nodes.PortItem import PortItemPin, PortItem
from .connections.ConnectionItem import default_cubic_connection_path, ConnectionItem, DataConnectionItem, ExecConnectionItem
from .drawings.DrawingObject import DrawingObject

class FlowView(GUIBase, QGraphicsView):
    """Manages the GUI of flows"""
    nodes_selection_changed = Signal(list)
    node_placed = Signal(Node)
    create_node_request = Signal(object, dict)
    remove_node_request = Signal(Node)
    check_connection_validity_request = Signal((NodeOutput, NodeInput), bool)
    connect_request = Signal(NodePort, NodePort)
    get_flow_data_request = Signal()
    viewport_update_mode_changed = Signal(str)

    def __init__(self, session_gui, flow, parent=None):
        if False:
            i = 10
            return i + 15
        GUIBase.__init__(self, representing_component=flow)
        QGraphicsView.__init__(self, parent=parent)
        self._undo_stack = QUndoStack(self)
        self._undo_action = self._undo_stack.createUndoAction(self, 'undo')
        self._undo_action.setShortcuts(QKeySequence.Undo)
        self._redo_action = self._undo_stack.createRedoAction(self, 'redo')
        self._redo_action.setShortcuts(QKeySequence.Redo)
        self._init_shortcuts()
        self.session_gui = session_gui
        self.flow: Flow = flow
        self.node_items: dict = {}
        self.node_items__cache: dict = {}
        self.connection_items: dict = {}
        self.connection_items__cache: dict = {}
        self._tmp_data = None
        self._selected_pin: PortItemPin = None
        self._dragging_connection = False
        self._temp_connection_ports = None
        self._waiting_for_connection_request: bool = False
        self.mouse_event_taken = False
        self._last_mouse_move_pos: QPointF = None
        self._node_place_pos = QPointF()
        self._left_mouse_pressed_in_flow = False
        self._right_mouse_pressed_in_flow = False
        self._mouse_press_pos: QPointF = None
        self._auto_connection_pin = None
        self._panning = False
        self._pan_last_x = None
        self._pan_last_y = None
        self._current_scale = 1
        self._total_scale_div = 1
        self._zoom_data = {'viewport pos': None, 'scene pos': None, 'delta': 0}
        self.create_node_request.connect(self.flow.create_node)
        self.remove_node_request.connect(self.flow.remove_node)
        self.check_connection_validity_request.connect(self.flow.check_connection_validity)
        self.get_flow_data_request.connect(self.flow.data)
        self.flow.node_added.sub(self.add_node)
        self.flow.node_removed.sub(self.remove_node)
        self.flow.connection_added.sub(self.add_connection)
        self.flow.connection_removed.sub(self.remove_connection)
        self.flow.connection_request_valid.sub(self.connection_request_valid)
        scene = QGraphicsScene(self)
        scene.setItemIndexMethod(QGraphicsScene.NoIndex)
        scene.setSceneRect(0, 0, 10000, 7000)
        self.setScene(scene)
        self.setCacheMode(QGraphicsView.CacheBackground)
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        scene.selectionChanged.connect(self._scene_selection_changed)
        self.setAcceptDrops(True)
        self.centerOn(QPointF(self.viewport().width() / 2, self.viewport().height() / 2))
        self.scene_rect_width = self.mapFromScene(self.sceneRect()).boundingRect().width()
        self.scene_rect_height = self.mapFromScene(self.sceneRect()).boundingRect().height()
        self._node_list_widget = NodeListWidget(self.session_gui)
        self._node_list_widget.setMinimumWidth(260)
        self._node_list_widget.setFixedHeight(300)
        self._node_list_widget.escaped.connect(self.hide_node_list_widget)
        self._node_list_widget.node_chosen.connect(self.create_node__cmd)
        self._node_list_widget_proxy = FlowViewProxyWidget(self)
        self._node_list_widget_proxy.setZValue(1000)
        self._node_list_widget_proxy.setWidget(self._node_list_widget)
        self.scene().addItem(self._node_list_widget_proxy)
        self.hide_node_list_widget()
        self.stylus_mode = ''
        self._current_drawing = None
        self._drawing = False
        self.drawings = []
        self._stylus_modes_proxy = FlowViewProxyWidget(self)
        self._stylus_modes_proxy.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self._stylus_modes_proxy.setZValue(1001)
        self._stylus_modes_widget = FlowViewStylusModesWidget(self)
        self._stylus_modes_proxy.setWidget(self._stylus_modes_widget)
        self.scene().addItem(self._stylus_modes_proxy)
        self.set_stylus_proxy_pos()
        self.setAttribute(Qt.WA_TabletTracking)
        self.viewport().setAttribute(Qt.WA_AcceptTouchEvents)
        self.last_pinch_points_dist = 0
        self.session_gui.design.flow_theme_changed.connect(self._theme_changed)
        self.session_gui.design.performance_mode_changed.connect(self._perf_mode_changed)
        data = self.flow.load_data
        if data is not None:
            view_data = data['flow view']
            if 'drawings' in view_data:
                self.place_drawings_from_data(view_data['drawings'])
            if 'view size' in view_data:
                self.setSceneRect(0, 0, view_data['view size'][0], view_data['view size'][1])
            self._undo_stack.clear()
        for node in self.flow.nodes:
            self.add_node(node)
        for c in [(o, i) for (o, conns) in self.flow.graph_adj.items() for i in conns]:
            self.add_connection(c)

    def _init_shortcuts(self):
        if False:
            print('Hello World!')
        place_new_node_shortcut = QShortcut(QKeySequence('Shift+P'), self)
        place_new_node_shortcut.activated.connect(self._place_new_node_by_shortcut)
        move_selected_components_left_shortcut = QShortcut(QKeySequence('Shift+Left'), self)
        move_selected_components_left_shortcut.activated.connect(self._move_selected_comps_left)
        move_selected_components_up_shortcut = QShortcut(QKeySequence('Shift+Up'), self)
        move_selected_components_up_shortcut.activated.connect(self._move_selected_comps_up)
        move_selected_components_right_shortcut = QShortcut(QKeySequence('Shift+Right'), self)
        move_selected_components_right_shortcut.activated.connect(self._move_selected_comps_right)
        move_selected_components_down_shortcut = QShortcut(QKeySequence('Shift+Down'), self)
        move_selected_components_down_shortcut.activated.connect(self._move_selected_comps_down)
        select_all_shortcut = QShortcut(QKeySequence('Ctrl+A'), self)
        select_all_shortcut.activated.connect(self.select_all)
        copy_shortcut = QShortcut(QKeySequence.Copy, self)
        copy_shortcut.activated.connect(self._copy)
        cut_shortcut = QShortcut(QKeySequence.Cut, self)
        cut_shortcut.activated.connect(self._cut)
        paste_shortcut = QShortcut(QKeySequence.Paste, self)
        paste_shortcut.activated.connect(self._paste)
        undo_shortcut = QShortcut(QKeySequence.Undo, self)
        undo_shortcut.activated.connect(self._undo_activated)
        redo_shortcut = QShortcut(QKeySequence.Redo, self)
        redo_shortcut.activated.connect(self._redo_activated)

    def _theme_changed(self, t):
        if False:
            while True:
                i = 10
        self._node_list_widget.setStyleSheet(self.session_gui.design.node_selection_stylesheet)
        for (n, ni) in self.node_items.items():
            ni.widget.rebuild_ui()
        self.viewport().update()
        self.scene().update(self.sceneRect())

    def _perf_mode_changed(self, mode):
        if False:
            i = 10
            return i + 15
        update_widget_value = mode == 'pretty'
        for (n, ni) in self.node_items.items():
            for inp in ni.inputs:
                inp.update_widget_value = update_widget_value and inp.widget
        self.viewport().update()
        self.scene().update(self.sceneRect())

    def _scene_selection_changed(self):
        if False:
            for i in range(10):
                print('nop')
        self.nodes_selection_changed.emit(self.selected_nodes())

    def contextMenuEvent(self, event):
        if False:
            return 10
        QGraphicsView.contextMenuEvent(self, event)
        if event.isAccepted():
            return
        for i in self.items(event.pos()):
            if isinstance(i, NodeItem):
                ni: NodeItem = i
                menu: QMenu = ni.get_context_menu()
                menu.exec_(event.globalPos())
                event.accept()

    def _push_undo(self, cmd: FlowUndoCommand):
        if False:
            i = 10
            return i + 15
        self._undo_stack.push(cmd)
        cmd.activate()

    def _undo_activated(self):
        if False:
            i = 10
            return i + 15
        'Triggered by ctrl+z'
        self._undo_stack.undo()
        self.viewport().update()

    def _redo_activated(self):
        if False:
            for i in range(10):
                print('nop')
        'Triggered by ctrl+y'
        self._undo_stack.redo()
        self.viewport().update()

    def mousePressEvent(self, event):
        if False:
            print('Hello World!')
        if self.mouse_event_taken:
            self.mouse_event_taken = False
            return
        QGraphicsView.mousePressEvent(self, event)
        if self.mouse_event_taken:
            self.mouse_event_taken = False
            return
        if event.button() == Qt.LeftButton:
            if self._node_list_widget_proxy.isVisible():
                self.hide_node_list_widget()
        elif event.button() == Qt.RightButton:
            self._right_mouse_pressed_in_flow = True
            event.accept()
        self._mouse_press_pos = self.mapToScene(event.pos())

    def mouseMoveEvent(self, event):
        if False:
            while True:
                i = 10
        QGraphicsView.mouseMoveEvent(self, event)
        if self._right_mouse_pressed_in_flow:
            if not self._panning:
                self._panning = True
                self._pan_last_x = event.x()
                self._pan_last_y = event.y()
            self.pan(event.pos())
            event.accept()
        self._last_mouse_move_pos = self.mapToScene(event.pos())
        if self._dragging_connection:
            self.viewport().repaint()

    def mouseReleaseEvent(self, event):
        if False:
            while True:
                i = 10
        QGraphicsView.mouseReleaseEvent(self, event)
        node_item_at_event_pos = None
        for item in self.items(event.pos()):
            if isinstance(item, NodeItem):
                node_item_at_event_pos = item
        if self.mouse_event_taken:
            self.mouse_event_taken = False
            self.viewport().repaint()
            return
        elif self._panning:
            self._panning = False
        elif event.button() == Qt.RightButton:
            self._right_mouse_pressed_in_flow = False
            if self._mouse_press_pos == self._last_mouse_move_pos:
                self.show_place_node_widget(event.pos())
                return
        if self._dragging_connection:
            port_items = {i: isinstance(i, PortItem) for i in self.items(event.pos())}
            if any(port_items.values()):
                p_i = list(port_items.keys())[list(port_items.values()).index(True)]
                self.connect_node_ports__cmd(self._selected_pin.port, p_i.port)
            elif node_item_at_event_pos:
                ni_under_drop = None
                for item in self.items(event.pos()):
                    if isinstance(item, NodeItem):
                        ni_under_drop = item
                        self.auto_connect(self._selected_pin.port, ni_under_drop.node)
                        break
            else:
                self._auto_connection_pin = self._selected_pin
                self.show_place_node_widget(event.pos())
            self._dragging_connection = False
            self._selected_pin = None
        elif event.button() == Qt.RightButton:
            self._right_mouse_pressed_in_flow = False
        self.viewport().repaint()

    def keyPressEvent(self, event):
        if False:
            return 10
        QGraphicsView.keyPressEvent(self, event)
        if event.isAccepted():
            return
        if event.key() == Qt.Key_Escape:
            self.clearFocus()
            self.setFocus()
            return True
        elif event.key() == Qt.Key_Delete:
            self.remove_selected_components__cmd()

    def wheelEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        if event.modifiers() & Qt.ControlModifier:
            event.accept()
            self._zoom_data['viewport pos'] = event.posF()
            self._zoom_data['scene pos'] = pointF_mapped(self.mapToScene(event.pos()), event.posF())
            self._zoom_data['delta'] += event.delta()
            if self._zoom_data['delta'] * event.delta() < 0:
                self._zoom_data['delta'] = event.delta()
            anim = QTimeLine(100, self)
            anim.setUpdateInterval(10)
            anim.valueChanged.connect(self._scaling_time)
            anim.start()
        else:
            super().wheelEvent(event)

    def _scaling_time(self, x):
        if False:
            i = 10
            return i + 15
        delta = self._zoom_data['delta'] / 8
        if abs(delta) <= 5:
            delta = self._zoom_data['delta']
        self._zoom_data['delta'] -= delta
        self.zoom(self._zoom_data['viewport pos'], self._zoom_data['scene pos'], delta)

    def viewportEvent(self, event: QEvent) -> bool:
        if False:
            i = 10
            return i + 15
        'handling some touch features here'
        if event.type() == QEvent.TouchBegin:
            self.setDragMode(QGraphicsView.NoDrag)
            return True
        elif event.type() == QEvent.TouchUpdate:
            event: QTouchEvent
            if len(event.touchPoints()) == 2:
                (tp0, tp1) = (event.touchPoints()[0], event.touchPoints()[1])
                (p0, p1) = (tp0.pos(), tp1.pos())
                pinch_points_dist = points_dist(p0, p1)
                if self.last_pinch_points_dist == 0:
                    self.last_pinch_points_dist = pinch_points_dist
                center = middle_point(p0, p1)
                self.zoom(p_abs=center, p_mapped=self.mapToScene(center.toPoint()), angle=((pinch_points_dist / self.last_pinch_points_dist) ** 10 - 1) * 100)
                self.last_pinch_points_dist = pinch_points_dist
            return True
        elif event.type() == QEvent.TouchEnd:
            self.last_pinch_points_dist = 0
            self.setDragMode(QGraphicsView.RubberBandDrag)
            return True
        else:
            return super().viewportEvent(event)

    def tabletEvent(self, event):
        if False:
            return 10
        'tabletEvent gets called by stylus operations.\n        LeftButton: std, no button pressed\n        RightButton: upper button pressed'
        if self.stylus_mode == 'edit' and (not self._panning) and (not (event.type() == QTabletEvent.TabletPress and event.button() == Qt.RightButton)):
            return
        scaled_event_pos: QPointF = event.posF() / self._current_scale
        if event.type() == QTabletEvent.TabletPress:
            self.mouse_event_taken = True
            if event.button() == Qt.LeftButton:
                if self.stylus_mode == 'comment':
                    view_pos = self.mapToScene(self.viewport().pos())
                    new_drawing = self._create_and_place_drawing__cmd(view_pos + scaled_event_pos, data={**self._stylus_modes_widget.get_pen_settings(), 'viewport pos': view_pos})
                    self._current_drawing = new_drawing
                    self._drawing = True
            elif event.button() == Qt.RightButton:
                self._panning = True
                self._pan_last_x = event.x()
                self._pan_last_y = event.y()
        elif event.type() == QTabletEvent.TabletMove:
            self.mouse_event_taken = True
            if self._panning:
                self.pan(event.pos())
            elif event.pointerType() == QTabletEvent.Eraser:
                if self.stylus_mode == 'comment':
                    for i in self.items(event.pos()):
                        if isinstance(i, DrawingObject):
                            self.remove_drawing(i)
                            break
            elif self.stylus_mode == 'comment' and self._drawing:
                if self._current_drawing.append_point(scaled_event_pos):
                    self._current_drawing.stroke_weights.append(event.pressure() * self._stylus_modes_widget.pen_width())
                self._current_drawing.update()
                self.viewport().update()
        elif event.type() == QTabletEvent.TabletRelease:
            if self._panning:
                self._panning = False
            if self.stylus_mode == 'comment' and self._drawing:
                self._current_drawing.finish()
                InfoMsgs.write('drawing finished')
                self._current_drawing = None
                self._drawing = False
    '\n    --> https://forum.qt.io/topic/121473/qgesturerecognizer-registerrecognizer-crashes-using-pyside2\n\n    def event(self, event) -> bool:\n        # if event.type() == QEvent.Gesture:\n        #     if event.gesture(PanGesture) is not None:\n        #         return self.pan_gesture(event)\n\n        return QGraphicsView.event(self, event)\n\n    def pan_gesture(self, event: QGestureEvent) -> bool:\n        pan: PanGesture = event.gesture(PanGesture)\n        print(pan)\n        return True\n    '

    def dragEnterEvent(self, event):
        if False:
            return 10
        if event.mimeData().hasFormat('application/json'):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if False:
            print('Hello World!')
        if event.mimeData().hasFormat('application/json'):
            event.acceptProposedAction()

    def dropEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        try:
            text = str(event.mimeData().data('application/json'), 'utf-8')
            data: dict = json.loads(text)
            if data['type'] == 'node':
                self._node_place_pos = self.mapToScene(event.pos())
                self.create_node__cmd(node_from_identifier(data['node identifier'], self.session_gui.core_session.nodes))
        except Exception:
            pass

    def drawBackground(self, painter, rect):
        if False:
            while True:
                i = 10
        painter.setBrush(self.session_gui.design.flow_theme.flow_background_brush)
        painter.drawRect(rect.intersected(self.sceneRect()))
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.sceneRect())
        if self.session_gui.design.performance_mode == 'pretty':
            theme = self.session_gui.design.flow_theme
            if theme.flow_background_grid and self._current_scale >= 0.7:
                if theme.flow_background_grid[0] == 'points':
                    color = theme.flow_background_grid[1]
                    pen_width = theme.flow_background_grid[2]
                    diff_x = theme.flow_background_grid[3]
                    diff_y = theme.flow_background_grid[4]
                    pen = QPen(color)
                    pen.setWidthF(pen_width)
                    painter.setPen(pen)
                    for x in range(diff_x, self.sceneRect().toRect().width(), diff_x):
                        for y in range(diff_y, self.sceneRect().toRect().height(), diff_y):
                            painter.drawPoint(x, y)
        self.set_stylus_proxy_pos()

    def drawForeground(self, painter, rect):
        if False:
            print('Hello World!')
        if self._dragging_connection:
            pen = QPen(QColor('#101520'))
            pen.setWidth(3)
            pen.setStyle(Qt.DotLine)
            painter.setPen(pen)
            pin_pos = self._selected_pin.get_scene_center_pos()
            spp = self._selected_pin.port
            cursor_pos = self._last_mouse_move_pos
            pos1 = pin_pos if spp.io_pos == PortObjPos.OUTPUT else cursor_pos
            pos2 = pin_pos if spp.io_pos == PortObjPos.INPUT else cursor_pos
            painter.drawPath(default_cubic_connection_path(pos1, pos2))
        for p_o in self.selected_drawings():
            pen = QPen(QColor('#a3cc3b'))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            size_factor = 1.05
            x = p_o.pos().x() - p_o.width / 2 * size_factor
            y = p_o.pos().y() - p_o.height / 2 * size_factor
            w = p_o.width * size_factor
            h = p_o.height * size_factor
            painter.drawRoundedRect(x, y, w, h, 6, 6)
            painter.drawEllipse(p_o.pos().x(), p_o.pos().y(), 2, 2)

    def get_viewport_img(self) -> QImage:
        if False:
            for i in range(10):
                print('nop')
        'Returns a clear image of the viewport'
        self.hide_proxies()
        img = QImage(self.viewport().rect().width(), self.viewport().height(), QImage.Format_ARGB32)
        img.fill(Qt.transparent)
        painter = QPainter(img)
        painter.setRenderHint(QPainter.Antialiasing)
        self.render(painter, self.viewport().rect(), self.viewport().rect())
        self.show_proxies()
        return img

    def get_whole_scene_img(self) -> QImage:
        if False:
            print('Hello World!')
        'Returns an image of the whole scene, scaled accordingly to current scale factor.\n        Due to a bug this only works from the viewport position down and right, so the user has to scroll to\n        the top left corner in order to get the full scene'
        self.hide_proxies()
        img = QImage(self.sceneRect().width() / self._total_scale_div, self.sceneRect().height() / self._total_scale_div, QImage.Format_RGB32)
        img.fill(Qt.transparent)
        painter = QPainter(img)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF()
        rect.setLeft(-self.viewport().pos().x())
        rect.setTop(-self.viewport().pos().y())
        rect.setWidth(img.rect().width())
        rect.setHeight(img.rect().height())
        self.render(painter, rect, rect.toRect())
        self.show_proxies()
        return img

    def set_stylus_proxy_pos(self):
        if False:
            while True:
                i = 10
        self._stylus_modes_proxy.setPos(self.mapToScene(self.viewport().width() - self._stylus_modes_widget.width(), 0))

    def hide_proxies(self):
        if False:
            for i in range(10):
                print('nop')
        self._stylus_modes_proxy.hide()

    def show_proxies(self):
        if False:
            return 10
        self._stylus_modes_proxy.show()

    def show_place_node_widget(self, pos, nodes=None):
        if False:
            print('Hello World!')
        'Opens the place node dialog in the scene.'
        self._node_place_pos = self.mapToScene(pos)
        dialog_pos = QPoint(pos.x() + 1, pos.y() + 1)
        if dialog_pos.x() + self._node_list_widget.width() / self._total_scale_div > self.viewport().width():
            dialog_pos.setX(dialog_pos.x() - (dialog_pos.x() + self._node_list_widget.width() / self._total_scale_div - self.viewport().width()))
        if dialog_pos.y() + self._node_list_widget.height() / self._total_scale_div > self.viewport().height():
            dialog_pos.setY(dialog_pos.y() - (dialog_pos.y() + self._node_list_widget.height() / self._total_scale_div - self.viewport().height()))
        dialog_pos = self.mapToScene(dialog_pos)
        self._node_list_widget.update_list(nodes if nodes is not None else self.session_gui.core_session.nodes)
        self._node_list_widget_proxy.setPos(dialog_pos)
        self._node_list_widget_proxy.show()
        self._node_list_widget.refocus()

    def hide_node_list_widget(self):
        if False:
            return 10
        self._node_list_widget_proxy.hide()
        self._node_list_widget.clearFocus()
        self._auto_connection_pin = None

    def _place_new_node_by_shortcut(self):
        if False:
            return 10
        point_in_viewport = None
        selected_NIs = self.selected_node_items()
        if len(selected_NIs) > 0:
            x = selected_NIs[-1].pos().x() + 150
            y = selected_NIs[-1].pos().y()
            self._node_place_pos = QPointF(x, y)
            point_in_viewport = self.mapFromScene(QPoint(x, y))
        else:
            viewport_x = self.viewport().width() / 2
            viewport_y = self.viewport().height() / 2
            point_in_viewport = QPointF(viewport_x, viewport_y).toPoint()
            self._node_place_pos = self.mapToScene(point_in_viewport)
        self.show_place_node_widget(point_in_viewport)

    def pan(self, new_pos):
        if False:
            for i in range(10):
                print('nop')
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - (new_pos.x() - self._pan_last_x))
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() - (new_pos.y() - self._pan_last_y))
        self._pan_last_x = new_pos.x()
        self._pan_last_y = new_pos.y()

    def zoom_in(self, amount):
        if False:
            i = 10
            return i + 15
        local_viewport_center = QPoint(self.viewport().width() / 2, self.viewport().height() / 2)
        self.zoom(local_viewport_center, self.mapToScene(local_viewport_center), amount)

    def zoom_out(self, amount):
        if False:
            while True:
                i = 10
        local_viewport_center = QPoint(self.viewport().width() / 2, self.viewport().height() / 2)
        self.zoom(local_viewport_center, self.mapToScene(local_viewport_center), -amount)

    def zoom(self, p_abs, p_mapped, angle):
        if False:
            for i in range(10):
                print('nop')
        by = 0
        velocity = 2 * (1 / self._current_scale) + 0.5
        if velocity > 3:
            velocity = 3
        if self._current_scale < 1:
            velocity *= self._current_scale
        zoom_dir_IN = angle > 0
        if zoom_dir_IN:
            by = 1 + angle / 4000 * velocity
        else:
            by = 1 - -angle / 4000 * velocity
        if zoom_dir_IN:
            if self._current_scale * by < 3:
                self.scale(by, by)
                self._current_scale *= by
        elif self.scene_rect_width * by >= self.viewport().size().width() and self.scene_rect_height * by >= self.viewport().size().height():
            self.scale(by, by)
            self._current_scale *= by
        w = self.viewport().width()
        h = self.viewport().height()
        wf = self.mapToScene(QPoint(w - 1, 0)).x() - self.mapToScene(QPoint(0, 0)).x()
        hf = self.mapToScene(QPoint(0, h - 1)).y() - self.mapToScene(QPoint(0, 0)).y()
        lf = p_mapped.x() - p_abs.x() * wf / w
        tf = p_mapped.y() - p_abs.y() * hf / h
        self.ensureVisible(lf, tf, wf, hf, 0, 0)
        target_rect = QRectF(QPointF(lf, tf), QSizeF(wf, hf))
        self._total_scale_div = target_rect.width() / self.viewport().width()
        self.ensureVisible(target_rect, 0, 0)

    def create_node__cmd(self, node_class):
        if False:
            print('Hello World!')
        self._push_undo(PlaceNode_Command(self, node_class, self._node_place_pos))

    def add_node(self, node):
        if False:
            return 10
        item: NodeItem = None
        if node in self.node_items__cache.keys():
            item = self.node_items__cache[node]
            self._add_node_item(item)
        else:
            item = NodeItem(node=node, node_gui=(node.GUI if hasattr(node, 'GUI') else NodeGUI)((node, self.session_gui)), flow_view=self, design=self.session_gui.design)
            item.initialize()
            self.node_placed.emit(node)
            item_data = node.load_data
            if item_data is not None and 'pos x' in item_data:
                pos = QPointF(item_data['pos x'], item_data['pos y'])
            else:
                pos = self._node_place_pos
            self._add_node_item(item, pos)
        if self._auto_connection_pin:
            self.auto_connect(self._auto_connection_pin.port, node)

    def _add_node_item(self, item: NodeItem, pos=None):
        if False:
            return 10
        self.node_items[item.node] = item
        self.scene().addItem(item)
        if pos:
            item.setPos(pos)
        self.clear_selection()
        item.setSelected(True)

    def remove_node(self, node):
        if False:
            i = 10
            return i + 15
        item = self.node_items[node]
        self._remove_node_item(item)
        del self.node_items[node]

    def _remove_node_item(self, item: NodeItem):
        if False:
            return 10
        self.node_items__cache[item.node] = item
        self.scene().removeItem(item)

    def connect_node_ports__cmd(self, p1: NodePort, p2: NodePort):
        if False:
            i = 10
            return i + 15
        if isinstance(p1, NodeOutput) and isinstance(p2, NodeInput):
            self._temp_connection_ports = (p1, p2)
            self._waiting_for_connection_request = True
            self.check_connection_validity_request.emit((p1, p2), True)
        elif isinstance(p1, NodeInput) and isinstance(p2, NodeOutput):
            self._temp_connection_ports = (p2, p1)
            self._waiting_for_connection_request = True
            self.check_connection_validity_request.emit((p2, p1), True)
        else:
            self.connection_request_valid(False)

    def connection_request_valid(self, valid: bool):
        if False:
            print('Hello World!')
        '\n        Triggered after the abstract flow evaluated validity of pending connect request.\n        This can also lead to a disconnect!\n        '
        if self._waiting_for_connection_request:
            self._waiting_for_connection_request = False
        else:
            return
        if valid:
            (out, inp) = self._temp_connection_ports
            if out.io_pos == PortObjPos.INPUT:
                (out, inp) = (inp, out)
            if self.flow.graph_adj_rev[inp] not in (None, out):
                self._push_undo(ConnectPorts_Command(self, out=self.flow.graph_adj_rev[inp], inp=inp))
            if self.flow.connected_output(inp) == out:
                self._push_undo(ConnectPorts_Command(self, out=self.flow.connected_output(inp), inp=inp))
            else:
                self._push_undo(ConnectPorts_Command(self, out=out, inp=inp))

    def add_connection(self, c: Tuple[NodeOutput, NodeInput]):
        if False:
            return 10
        (out, inp) = c
        item: ConnectionItem = None
        if c in self.connection_items__cache.keys():
            item = self.connection_items__cache[c]
        elif inp.type_ == 'data':
            item = DataConnectionItem(c, self.session_gui.design)
        else:
            item = ExecConnectionItem(c, self.session_gui.design)
        self._add_connection_item(item)
        item.out_item.port_connected()
        item.inp_item.port_connected()

    def _add_connection_item(self, item: ConnectionItem):
        if False:
            return 10
        self.connection_items[item.connection] = item
        self.scene().addItem(item)
        item.setZValue(10)

    def remove_connection(self, c: Tuple[NodeOutput, NodeInput]):
        if False:
            return 10
        item = self.connection_items[c]
        self._remove_connection_item(item)
        item.out_item.port_disconnected()
        item.inp_item.port_disconnected()
        del self.connection_items[c]

    def _remove_connection_item(self, item: ConnectionItem):
        if False:
            return 10
        self.connection_items__cache[item.connection] = item
        self.scene().removeItem(item)

    def auto_connect(self, p: NodePort, n: Node):
        if False:
            print('Hello World!')
        if p.io_pos == PortObjPos.OUTPUT:
            for inp in n.inputs:
                if p.type_ == inp.type_:
                    self.connect_node_ports__cmd(p, inp)
                    return
        elif p.io_pos == PortObjPos.INPUT:
            for out in n.outputs:
                if p.type_ == out.type_:
                    self.connect_node_ports__cmd(p, out)
                    return

    def update_conn_item(self, c: Tuple[NodeOutput, NodeInput]):
        if False:
            print('Hello World!')
        if c in self.connection_items:
            self.connection_items[c].changed = True
            self.connection_items[c].update()

    def create_drawing(self, data=None) -> DrawingObject:
        if False:
            while True:
                i = 10
        'Creates and returns a new DrawingObject.'
        new_drawing = DrawingObject(self, data)
        return new_drawing

    def add_drawing(self, drawing_obj, posF=None):
        if False:
            while True:
                i = 10
        'Adds a DrawingObject to the scene.'
        self.scene().addItem(drawing_obj)
        if posF:
            drawing_obj.setPos(posF)
        self.drawings.append(drawing_obj)

    def add_drawings(self, drawings):
        if False:
            print('Hello World!')
        'Adds a list of DrawingObjects to the scene.'
        for d in drawings:
            self.add_drawing(d)

    def remove_drawing(self, drawing: DrawingObject):
        if False:
            for i in range(10):
                print('nop')
        'Removes a drawing from the scene.'
        self.scene().removeItem(drawing)
        self.drawings.remove(drawing)

    def place_drawings_from_data(self, drawings_data: list, offset_pos=QPoint(0, 0)):
        if False:
            while True:
                i = 10
        "Creates and places drawings from drawings. The same list is returned by the data_() method\n        at 'drawings'."
        new_drawings = []
        for d_data in drawings_data:
            x = d_data['pos x'] + offset_pos.x()
            y = d_data['pos y'] + offset_pos.y()
            new_drawing = self.create_drawing(data=d_data)
            self.add_drawing(new_drawing, QPointF(x, y))
            new_drawings.append(new_drawing)
        return new_drawings

    def _create_and_place_drawing__cmd(self, posF, data=None):
        if False:
            while True:
                i = 10
        new_drawing_obj = self.create_drawing(data)
        place_command = PlaceDrawing_Command(self, posF, new_drawing_obj)
        self._push_undo(place_command)
        return new_drawing_obj

    def add_component(self, e: QGraphicsItem):
        if False:
            return 10
        if isinstance(e, NodeItem):
            self.add_node(e.node)
            self.add_node_item(e)
        elif isinstance(e, DrawingObject):
            self.add_drawing(e)

    def remove_components(self, comps: [QGraphicsItem]):
        if False:
            i = 10
            return i + 15
        for c in comps:
            self.remove_component(c)

    def remove_component(self, e: QGraphicsItem):
        if False:
            return 10
        if isinstance(e, NodeItem):
            self.remove_node(e.node)
            self.remove_node_item(e)
        elif isinstance(e, DrawingObject):
            self.remove_drawing(e)

    def remove_selected_components__cmd(self):
        if False:
            print('Hello World!')
        self._push_undo(RemoveComponents_Command(self, self.scene().selectedItems()))
        self.viewport().update()

    def _move_selected_copmonents__cmd(self, x, y):
        if False:
            print('Hello World!')
        new_rel_pos = QPointF(x, y)
        left = False
        for i in self.scene().selectedItems():
            new_pos = i.pos() + new_rel_pos
            w = i.boundingRect().width()
            h = i.boundingRect().height()
            if new_pos.x() - w / 2 < 0 or new_pos.x() + w / 2 > self.scene().width() or new_pos.y() - h / 2 < 0 or (new_pos.y() + h / 2 > self.scene().height()):
                left = True
                break
        if not left:
            items_group = self.scene().createItemGroup(self.scene().selectedItems())
            items_group.moveBy(new_rel_pos.x(), new_rel_pos.y())
            self.scene().destroyItemGroup(items_group)
            self._push_undo(MoveComponents_Command(self, self.scene().selectedItems(), p_from=-new_rel_pos, p_to=QPointF(0, 0)))
        self.viewport().repaint()

    def _move_selected_comps_left(self):
        if False:
            return 10
        self._move_selected_copmonents__cmd(-40, 0)

    def _move_selected_comps_up(self):
        if False:
            i = 10
            return i + 15
        self._move_selected_copmonents__cmd(0, -40)

    def _move_selected_comps_right(self):
        if False:
            print('Hello World!')
        self._move_selected_copmonents__cmd(+40, 0)

    def _move_selected_comps_down(self):
        if False:
            for i in range(10):
                print('nop')
        self._move_selected_copmonents__cmd(0, +40)

    def selected_components_moved(self, pos_diff):
        if False:
            for i in range(10):
                print('nop')
        items_list = self.scene().selectedItems()
        self._push_undo(MoveComponents_Command(self, items_list, p_from=-pos_diff, p_to=QPointF(0, 0)))

    def selected_node_items(self) -> [NodeItem]:
        if False:
            print('Hello World!')
        'Returns a list of the currently selected NodeItems.'
        selected_NIs = []
        for i in self.scene().selectedItems():
            if isinstance(i, NodeItem):
                selected_NIs.append(i)
        return selected_NIs

    def selected_nodes(self) -> [Node]:
        if False:
            i = 10
            return i + 15
        return [item.node for item in self.selected_node_items()]

    def selected_drawings(self) -> [DrawingObject]:
        if False:
            for i in range(10):
                print('nop')
        'Returns a list of the currently selected drawings.'
        selected_drawings = []
        for i in self.scene().selectedItems():
            if isinstance(i, DrawingObject):
                selected_drawings.append(i)
        return selected_drawings

    def select_all(self):
        if False:
            i = 10
            return i + 15
        for i in self.scene().items():
            if i.ItemIsSelectable:
                i.setSelected(True)
        self.viewport().repaint()

    def clear_selection(self):
        if False:
            for i in range(10):
                print('nop')
        self.scene().clearSelection()

    def select_components(self, comps):
        if False:
            print('Hello World!')
        self.scene().clearSelection()
        for c in comps:
            c.setSelected(True)

    def _copy(self):
        if False:
            while True:
                i = 10
        data = {'nodes': self._get_nodes_data(self.selected_nodes()), 'connections': self._get_connections_data(self.selected_nodes()), 'output data': self._get_output_data(self.selected_nodes()), 'drawings': self._get_drawings_data(self.selected_drawings())}
        QGuiApplication.clipboard().setText(json.dumps(data))

    def _cut(self):
        if False:
            i = 10
            return i + 15
        data = {'nodes': self._get_nodes_data(self.selected_nodes()), 'connections': self._get_connections_data(self.selected_nodes()), 'drawings': self._get_drawings_data(self.selected_drawings())}
        QGuiApplication.clipboard().setText(json.dumps(data))
        self.remove_selected_components__cmd()

    def _paste(self):
        if False:
            while True:
                i = 10
        data = {}
        try:
            data = json.loads(QGuiApplication.clipboard().text())
        except Exception as e:
            return
        self.clear_selection()
        positions = []
        for d in data['drawings']:
            positions.append({'x': d['pos x'], 'y': d['pos y']})
        for n in data['nodes']:
            positions.append({'x': n['pos x'], 'y': n['pos y']})
        offset_for_middle_pos = QPointF(0, 0)
        if len(positions) > 0:
            rect = QRectF(positions[0]['x'], positions[0]['y'], 0, 0)
            for p in positions:
                x = p['x']
                y = p['y']
                if x < rect.left():
                    rect.setLeft(x)
                if x > rect.right():
                    rect.setRight(x)
                if y < rect.top():
                    rect.setTop(y)
                if y > rect.bottom():
                    rect.setBottom(y)
            offset_for_middle_pos = self._last_mouse_move_pos - rect.center()
        self._push_undo(Paste_Command(self, data, offset_for_middle_pos))

    def complete_data(self, data: dict):
        if False:
            for i in range(10):
                print('nop')
        data['flow view'] = {'drawings': self._get_drawings_data(self.drawings), 'view size': [self.sceneRect().size().width(), self.sceneRect().size().height()]}
        return data

    def _get_nodes_data(self, nodes):
        if False:
            return 10
        'generates the data for the specified list of nodes'
        f_complete_data = self.session_gui.core_session.complete_data
        return f_complete_data(self.flow._gen_nodes_data(nodes))

    def _get_connections_data(self, nodes):
        if False:
            return 10
        'generates the connections data for connections between a specified list of nodes'
        f_complete_data = self.session_gui.core_session.complete_data
        return f_complete_data(self.flow._gen_conns_data(nodes))

    def _get_output_data(self, nodes):
        if False:
            print('Hello World!')
        'generates the serialized data of output ports of the specified nodes'
        f_complete_data = self.session_gui.core_session.complete_data
        return f_complete_data(self.flow._gen_output_data(nodes))

    def _get_drawings_data(self, drawings):
        if False:
            return 10
        'generates the data for a list of drawings'
        return [d.data() for d in drawings]