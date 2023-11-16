import sys
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QOpenGLContext
from PyQt6.QtWidgets import QApplication
from UM.Application import Application
from UM.Event import Event, KeyEvent
from UM.Job import Job
from UM.Logger import Logger
from UM.Math.Color import Color
from UM.Math.Matrix import Matrix
from UM.Mesh.MeshBuilder import MeshBuilder
from UM.Message import Message
from UM.Platform import Platform
from UM.PluginRegistry import PluginRegistry
from UM.Qt.QtApplication import QtApplication
from UM.Resources import Resources
from UM.Scene.Iterator.DepthFirstIterator import DepthFirstIterator
from UM.Scene.Selection import Selection
from UM.Signal import Signal
from UM.View.CompositePass import CompositePass
from UM.View.GL.OpenGL import OpenGL
from UM.View.GL.OpenGLContext import OpenGLContext
from UM.View.GL.ShaderProgram import ShaderProgram
from UM.i18n import i18nCatalog
from cura.CuraView import CuraView
from cura.LayerPolygon import LayerPolygon
from cura.Scene.ConvexHullNode import ConvexHullNode
from cura.CuraApplication import CuraApplication
from .NozzleNode import NozzleNode
from .SimulationPass import SimulationPass
from .SimulationViewProxy import SimulationViewProxy
import numpy
import os.path
from typing import Optional, TYPE_CHECKING, List, cast
if TYPE_CHECKING:
    from UM.Scene.SceneNode import SceneNode
    from UM.Scene.Scene import Scene
    from UM.Settings.ContainerStack import ContainerStack
catalog = i18nCatalog('cura')

class SimulationView(CuraView):
    """The preview layer view. It is used to display g-code paths."""
    LAYER_VIEW_TYPE_MATERIAL_TYPE = 0
    LAYER_VIEW_TYPE_LINE_TYPE = 1
    LAYER_VIEW_TYPE_FEEDRATE = 2
    LAYER_VIEW_TYPE_THICKNESS = 3
    _no_layers_warning_preference = 'view/no_layers_warning'

    def __init__(self, parent=None) -> None:
        if False:
            return 10
        super().__init__(parent)
        self._max_layers = 0
        self._current_layer_num = 0
        self._minimum_layer_num = 0
        self._current_layer_mesh = None
        self._current_layer_jumps = None
        self._top_layers_job = None
        self._activity = False
        self._old_max_layers = 0
        self._max_paths = 0
        self._current_path_num = 0
        self._minimum_path_num = 0
        self.currentLayerNumChanged.connect(self._onCurrentLayerNumChanged)
        self._busy = False
        self._simulation_running = False
        self._ghost_shader = None
        self._layer_pass = None
        self._composite_pass = None
        self._old_layer_bindings = None
        self._simulationview_composite_shader = None
        self._old_composite_shader = None
        self._max_feedrate = sys.float_info.min
        self._min_feedrate = sys.float_info.max
        self._max_thickness = sys.float_info.min
        self._min_thickness = sys.float_info.max
        self._max_line_width = sys.float_info.min
        self._min_line_width = sys.float_info.max
        self._min_flow_rate = sys.float_info.max
        self._max_flow_rate = sys.float_info.min
        self._global_container_stack = None
        self._proxy = None
        self._resetSettings()
        self._legend_items = None
        self._show_travel_moves = False
        self._nozzle_node = None
        Application.getInstance().getPreferences().addPreference('view/top_layer_count', 5)
        Application.getInstance().getPreferences().addPreference('view/only_show_top_layers', False)
        Application.getInstance().getPreferences().addPreference('view/force_layer_view_compatibility_mode', False)
        Application.getInstance().getPreferences().addPreference('layerview/layer_view_type', 1)
        Application.getInstance().getPreferences().addPreference('layerview/extruder_opacities', '')
        Application.getInstance().getPreferences().addPreference('layerview/show_travel_moves', False)
        Application.getInstance().getPreferences().addPreference('layerview/show_helpers', True)
        Application.getInstance().getPreferences().addPreference('layerview/show_skin', True)
        Application.getInstance().getPreferences().addPreference('layerview/show_infill', True)
        Application.getInstance().getPreferences().addPreference('layerview/show_starts', True)
        self.visibleStructuresChanged.connect(self.calculateColorSchemeLimits)
        self._updateWithPreferences()
        self._solid_layers = int(Application.getInstance().getPreferences().getValue('view/top_layer_count'))
        self._only_show_top_layers = bool(Application.getInstance().getPreferences().getValue('view/only_show_top_layers'))
        self._compatibility_mode = self._evaluateCompatibilityMode()
        self._slice_first_warning_message = Message(catalog.i18nc('@info:status', 'Nothing is shown because you need to slice first.'), title=catalog.i18nc('@info:title', 'No layers to show'), option_text=catalog.i18nc('@info:option_text', 'Do not show this message again'), option_state=False, message_type=Message.MessageType.WARNING)
        self._slice_first_warning_message.optionToggled.connect(self._onDontAskMeAgain)
        CuraApplication.getInstance().getPreferences().addPreference(self._no_layers_warning_preference, True)
        QtApplication.getInstance().engineCreatedSignal.connect(self._onEngineCreated)

    def _onEngineCreated(self) -> None:
        if False:
            print('Hello World!')
        plugin_path = PluginRegistry.getInstance().getPluginPath(self.getPluginId())
        if plugin_path:
            self.addDisplayComponent('main', os.path.join(plugin_path, 'SimulationViewMainComponent.qml'))
            self.addDisplayComponent('menu', os.path.join(plugin_path, 'SimulationViewMenuComponent.qml'))
        else:
            Logger.log('e', 'Unable to find the path for %s', self.getPluginId())

    def _evaluateCompatibilityMode(self) -> bool:
        if False:
            while True:
                i = 10
        return OpenGLContext.isLegacyOpenGL() or bool(Application.getInstance().getPreferences().getValue('view/force_layer_view_compatibility_mode'))

    def _resetSettings(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._layer_view_type = 0
        self._extruder_count = 0
        self._extruder_opacity = [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
        self._show_travel_moves = False
        self._show_helpers = True
        self._show_skin = True
        self._show_infill = True
        self._show_starts = True
        self.resetLayerData()

    def getActivity(self) -> bool:
        if False:
            while True:
                i = 10
        return self._activity

    def setActivity(self, activity: bool) -> None:
        if False:
            while True:
                i = 10
        if self._activity == activity:
            return
        self._activity = activity
        self._updateSliceWarningVisibility()
        self.activityChanged.emit()

    def getSimulationPass(self) -> SimulationPass:
        if False:
            return 10
        if not self._layer_pass:
            self._layer_pass = SimulationPass(1, 1)
            self._compatibility_mode = self._evaluateCompatibilityMode()
            self._layer_pass.setSimulationView(self)
        return self._layer_pass

    def getCurrentLayer(self) -> int:
        if False:
            while True:
                i = 10
        return self._current_layer_num

    def getMinimumLayer(self) -> int:
        if False:
            while True:
                i = 10
        return self._minimum_layer_num

    def getMaxLayers(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self._max_layers

    def getCurrentPath(self) -> int:
        if False:
            return 10
        return self._current_path_num

    def getMinimumPath(self) -> int:
        if False:
            i = 10
            return i + 15
        return self._minimum_path_num

    def getMaxPaths(self) -> int:
        if False:
            print('Hello World!')
        return self._max_paths

    def getNozzleNode(self) -> NozzleNode:
        if False:
            i = 10
            return i + 15
        if not self._nozzle_node:
            self._nozzle_node = NozzleNode()
        return self._nozzle_node

    def _onSceneChanged(self, node: 'SceneNode') -> None:
        if False:
            return 10
        if node.getMeshData() is None:
            return
        self.setActivity(False)
        self.calculateColorSchemeLimits()
        self.calculateMaxLayers()
        self.calculateMaxPathsOnLayer(self._current_layer_num)

    def isBusy(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self._busy

    def setBusy(self, busy: bool) -> None:
        if False:
            return 10
        if busy != self._busy:
            self._busy = busy
            self.busyChanged.emit()

    def isSimulationRunning(self) -> bool:
        if False:
            while True:
                i = 10
        return self._simulation_running

    def setSimulationRunning(self, running: bool) -> None:
        if False:
            while True:
                i = 10
        self._simulation_running = running

    def resetLayerData(self) -> None:
        if False:
            while True:
                i = 10
        self._current_layer_mesh = None
        self._current_layer_jumps = None

    def beginRendering(self) -> None:
        if False:
            i = 10
            return i + 15
        scene = self.getController().getScene()
        renderer = self.getRenderer()
        if renderer is None:
            return
        if not self._ghost_shader:
            self._ghost_shader = OpenGL.getInstance().createShaderProgram(Resources.getPath(Resources.Shaders, 'color.shader'))
            theme = CuraApplication.getInstance().getTheme()
            if theme is not None:
                self._ghost_shader.setUniformValue('u_color', Color(*theme.getColor('layerview_ghost').getRgb()))
        for node in DepthFirstIterator(scene.getRoot()):
            if type(node) is ConvexHullNode and (not Selection.isSelected(cast(ConvexHullNode, node).getWatchedNode())):
                continue
            if not node.render(renderer):
                if node.getMeshData() and node.isVisible():
                    renderer.queueNode(node, transparent=True, shader=self._ghost_shader)

    def setLayer(self, value: int) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Set the upper end of the range of visible layers.\n\n        If setting it below the lower end of the range, the lower end is lowered so that 1 layer stays visible.\n        :param value: The new layer number to show, 0-indexed.\n        '
        if self._current_layer_num != value:
            self._current_layer_num = min(max(value, 0), self._max_layers)
            self._minimum_layer_num = min(self._current_layer_num, self._minimum_layer_num)
            self._startUpdateTopLayers()
            self.currentLayerNumChanged.emit()

    def setMinimumLayer(self, value: int) -> None:
        if False:
            return 10
        '\n        Set the lower end of the range of visible layers.\n\n        If setting it above the upper end of the range, the upper end is increased so that 1 layer stays visible.\n        :param value: The new lower end of the range of visible layers, 0-indexed.\n        '
        if self._minimum_layer_num != value:
            self._minimum_layer_num = min(max(value, 0), self._max_layers)
            self._current_layer_num = max(self._current_layer_num, self._minimum_layer_num)
            self._startUpdateTopLayers()
            self.currentLayerNumChanged.emit()

    def setPath(self, value: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the upper end of the range of visible paths on the current layer.\n\n        If setting it below the lower end of the range, the lower end is lowered so that 1 path stays visible.\n        :param value: The new path index to show, 0-indexed.\n        '
        if self._current_path_num != value:
            self._current_path_num = min(max(value, 0), self._max_paths)
            self._minimum_path_num = min(self._minimum_path_num, self._current_path_num)
            self._startUpdateTopLayers()
            self.currentPathNumChanged.emit()

    def setMinimumPath(self, value: int) -> None:
        if False:
            while True:
                i = 10
        '\n        Set the lower end of the range of visible paths on the current layer.\n\n        If setting it above the upper end of the range, the upper end is increased so that 1 path stays visible.\n        :param value: The new lower end of the range of visible paths, 0-indexed.\n        '
        if self._minimum_path_num != value:
            self._minimum_path_num = min(max(value, 0), self._max_paths)
            self._current_path_num = max(self._current_path_num, self._minimum_path_num)
            self._startUpdateTopLayers()
            self.currentPathNumChanged.emit()

    def setSimulationViewType(self, layer_view_type: int) -> None:
        if False:
            print('Hello World!')
        'Set the layer view type\n\n        :param layer_view_type: integer as in SimulationView.qml and this class\n        '
        if layer_view_type != self._layer_view_type:
            self._layer_view_type = layer_view_type
            self.currentLayerNumChanged.emit()

    def getSimulationViewType(self) -> int:
        if False:
            while True:
                i = 10
        'Return the layer view type, integer as in SimulationView.qml and this class'
        return self._layer_view_type

    def setExtruderOpacity(self, extruder_nr: int, opacity: float) -> None:
        if False:
            i = 10
            return i + 15
        'Set the extruder opacity\n\n        :param extruder_nr: 0..15\n        :param opacity: 0.0 .. 1.0\n        '
        if 0 <= extruder_nr <= 15:
            self._extruder_opacity[extruder_nr // 4][extruder_nr % 4] = opacity
            self.currentLayerNumChanged.emit()

    def getExtruderOpacities(self) -> Matrix:
        if False:
            for i in range(10):
                print('nop')
        return Matrix(self._extruder_opacity)

    def setShowTravelMoves(self, show: bool) -> None:
        if False:
            return 10
        if show == self._show_travel_moves:
            return
        self._show_travel_moves = show
        self.currentLayerNumChanged.emit()
        self.visibleStructuresChanged.emit()

    def getShowTravelMoves(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self._show_travel_moves

    def setShowHelpers(self, show: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        if show == self._show_helpers:
            return
        self._show_helpers = show
        self.currentLayerNumChanged.emit()
        self.visibleStructuresChanged.emit()

    def getShowHelpers(self) -> bool:
        if False:
            print('Hello World!')
        return self._show_helpers

    def setShowSkin(self, show: bool) -> None:
        if False:
            while True:
                i = 10
        if show == self._show_skin:
            return
        self._show_skin = show
        self.currentLayerNumChanged.emit()
        self.visibleStructuresChanged.emit()

    def getShowSkin(self) -> bool:
        if False:
            print('Hello World!')
        return self._show_skin

    def setShowInfill(self, show: bool) -> None:
        if False:
            print('Hello World!')
        if show == self._show_infill:
            return
        self._show_infill = show
        self.currentLayerNumChanged.emit()
        self.visibleStructuresChanged.emit()

    def getShowInfill(self) -> bool:
        if False:
            print('Hello World!')
        return self._show_infill

    def setShowStarts(self, show: bool) -> None:
        if False:
            print('Hello World!')
        if show == self._show_starts:
            return
        self._show_starts = show
        self.currentLayerNumChanged.emit()
        self.visibleStructuresChanged.emit()

    def getShowStarts(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self._show_starts

    def getCompatibilityMode(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self._compatibility_mode

    def getExtruderCount(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self._extruder_count

    def getMinFeedrate(self) -> float:
        if False:
            return 10
        if abs(self._min_feedrate - sys.float_info.max) < 10:
            return 0.0
        return self._min_feedrate

    def getMaxFeedrate(self) -> float:
        if False:
            i = 10
            return i + 15
        return self._max_feedrate

    def getMinThickness(self) -> float:
        if False:
            while True:
                i = 10
        if abs(self._min_thickness - sys.float_info.max) < 10:
            return 0.0
        return self._min_thickness

    def getMaxThickness(self) -> float:
        if False:
            return 10
        return self._max_thickness

    def getMaxLineWidth(self) -> float:
        if False:
            while True:
                i = 10
        return self._max_line_width

    def getMinLineWidth(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        if abs(self._min_line_width - sys.float_info.max) < 10:
            return 0.0
        return self._min_line_width

    def getMaxFlowRate(self) -> float:
        if False:
            return 10
        return self._max_flow_rate

    def getMinFlowRate(self) -> float:
        if False:
            print('Hello World!')
        if abs(self._min_flow_rate - sys.float_info.max) < 10:
            return 0.0
        return self._min_flow_rate

    def calculateMaxLayers(self) -> None:
        if False:
            return 10
        '\n        Calculates number of layers, triggers signals if the number of layers changed and makes sure the top layers are\n        recalculated for legacy layer view.\n        '
        scene = self.getController().getScene()
        self._old_max_layers = self._max_layers
        new_max_layers = -1
        for node in DepthFirstIterator(scene.getRoot()):
            layer_data = node.callDecoration('getLayerData')
            if not layer_data:
                continue
            self.setActivity(True)
            min_layer_number = sys.maxsize
            max_layer_number = -sys.maxsize
            for layer_id in layer_data.getLayers():
                if len(layer_data.getLayer(layer_id).polygons) < 1:
                    continue
                if max_layer_number < layer_id:
                    max_layer_number = layer_id
                if min_layer_number > layer_id:
                    min_layer_number = layer_id
            layer_count = max_layer_number - min_layer_number
            if new_max_layers < layer_count:
                new_max_layers = layer_count
        if new_max_layers >= 0 and new_max_layers != self._old_max_layers:
            self._max_layers = new_max_layers
            if new_max_layers > self._current_layer_num:
                self.maxLayersChanged.emit()
                self.setLayer(int(self._max_layers))
            else:
                self.setLayer(int(self._max_layers))
                self.maxLayersChanged.emit()
        self._startUpdateTopLayers()

    def calculateColorSchemeLimits(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Calculates the limits of the colour schemes, depending on the layer view data that is visible to the user.\n        '
        old_min_feedrate = self._min_feedrate
        old_max_feedrate = self._max_feedrate
        old_min_linewidth = self._min_line_width
        old_max_linewidth = self._max_line_width
        old_min_thickness = self._min_thickness
        old_max_thickness = self._max_thickness
        old_min_flow_rate = self._min_flow_rate
        old_max_flow_rate = self._max_flow_rate
        self._min_feedrate = sys.float_info.max
        self._max_feedrate = sys.float_info.min
        self._min_line_width = sys.float_info.max
        self._max_line_width = sys.float_info.min
        self._min_thickness = sys.float_info.max
        self._max_thickness = sys.float_info.min
        self._min_flow_rate = sys.float_info.max
        self._max_flow_rate = sys.float_info.min
        visible_line_types = []
        if self.getShowSkin():
            visible_line_types.append(LayerPolygon.SkinType)
            visible_line_types.append(LayerPolygon.Inset0Type)
            visible_line_types.append(LayerPolygon.InsetXType)
        if self.getShowInfill():
            visible_line_types.append(LayerPolygon.InfillType)
        if self.getShowHelpers():
            visible_line_types.append(LayerPolygon.PrimeTowerType)
            visible_line_types.append(LayerPolygon.SkirtType)
            visible_line_types.append(LayerPolygon.SupportType)
            visible_line_types.append(LayerPolygon.SupportInfillType)
            visible_line_types.append(LayerPolygon.SupportInterfaceType)
        visible_line_types_with_extrusion = visible_line_types.copy()
        if self.getShowTravelMoves():
            visible_line_types.append(LayerPolygon.MoveCombingType)
            visible_line_types.append(LayerPolygon.MoveRetractionType)
        for node in DepthFirstIterator(self.getController().getScene().getRoot()):
            layer_data = node.callDecoration('getLayerData')
            if not layer_data:
                continue
            for layer_index in layer_data.getLayers():
                for polyline in layer_data.getLayer(layer_index).polygons:
                    is_visible = numpy.isin(polyline.types, visible_line_types)
                    visible_indices = numpy.where(is_visible)[0]
                    visible_indicies_with_extrusion = numpy.where(numpy.isin(polyline.types, visible_line_types_with_extrusion))[0]
                    if visible_indices.size == 0:
                        continue
                    visible_feedrates = numpy.take(polyline.lineFeedrates, visible_indices)
                    visible_feedrates_with_extrusion = numpy.take(polyline.lineFeedrates, visible_indicies_with_extrusion)
                    visible_linewidths = numpy.take(polyline.lineWidths, visible_indices)
                    visible_linewidths_with_extrusion = numpy.take(polyline.lineWidths, visible_indicies_with_extrusion)
                    visible_thicknesses = numpy.take(polyline.lineThicknesses, visible_indices)
                    visible_thicknesses_with_extrusion = numpy.take(polyline.lineThicknesses, visible_indicies_with_extrusion)
                    self._max_feedrate = max(float(visible_feedrates.max()), self._max_feedrate)
                    if visible_feedrates_with_extrusion.size != 0:
                        flow_rates = visible_feedrates_with_extrusion * visible_linewidths_with_extrusion * visible_thicknesses_with_extrusion
                        self._min_flow_rate = min(float(flow_rates.min()), self._min_flow_rate)
                        self._max_flow_rate = max(float(flow_rates.max()), self._max_flow_rate)
                    self._min_feedrate = min(float(visible_feedrates.min()), self._min_feedrate)
                    self._max_line_width = max(float(visible_linewidths.max()), self._max_line_width)
                    self._min_line_width = min(float(visible_linewidths.min()), self._min_line_width)
                    self._max_thickness = max(float(visible_thicknesses.max()), self._max_thickness)
                    try:
                        self._min_thickness = min(float(visible_thicknesses[numpy.nonzero(visible_thicknesses)].min()), self._min_thickness)
                    except ValueError:
                        Logger.log('w', "Min thickness can't be calculated because all the values are zero")
        if old_min_feedrate != self._min_feedrate or old_max_feedrate != self._max_feedrate or old_min_linewidth != self._min_line_width or (old_max_linewidth != self._max_line_width) or (old_min_thickness != self._min_thickness) or (old_max_thickness != self._max_thickness) or (old_min_flow_rate != self._min_flow_rate) or (old_max_flow_rate != self._max_flow_rate):
            self.colorSchemeLimitsChanged.emit()

    def calculateMaxPathsOnLayer(self, layer_num: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        scene = self.getController().getScene()
        for node in DepthFirstIterator(scene.getRoot()):
            layer_data = node.callDecoration('getLayerData')
            if not layer_data:
                continue
            layer = layer_data.getLayer(layer_num)
            if layer is None:
                return
            new_max_paths = layer.lineMeshElementCount()
            if new_max_paths >= 0 and new_max_paths != self._max_paths:
                self._max_paths = new_max_paths
                self.maxPathsChanged.emit()
            self.setPath(int(new_max_paths))
    maxLayersChanged = Signal()
    maxPathsChanged = Signal()
    currentLayerNumChanged = Signal()
    currentPathNumChanged = Signal()
    globalStackChanged = Signal()
    preferencesChanged = Signal()
    busyChanged = Signal()
    activityChanged = Signal()
    visibleStructuresChanged = Signal()
    colorSchemeLimitsChanged = Signal()

    def getProxy(self, engine, script_engine):
        if False:
            print('Hello World!')
        'Hackish way to ensure the proxy is already created\n\n        which ensures that the layerview.qml is already created as this caused some issues.\n        '
        if self._proxy is None:
            self._proxy = SimulationViewProxy(self)
        return self._proxy

    def endRendering(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def event(self, event) -> bool:
        if False:
            for i in range(10):
                print('nop')
        modifiers = QApplication.keyboardModifiers()
        ctrl_is_active = modifiers & Qt.KeyboardModifier.ControlModifier
        shift_is_active = modifiers & Qt.KeyboardModifier.ShiftModifier
        if event.type == Event.KeyPressEvent and ctrl_is_active:
            amount = 10 if shift_is_active else 1
            if event.key == KeyEvent.UpKey:
                self.setLayer(self._current_layer_num + amount)
                return True
            if event.key == KeyEvent.DownKey:
                self.setLayer(self._current_layer_num - amount)
                return True
        if event.type == Event.ViewActivateEvent:
            Application.getInstance().getPreferences().preferenceChanged.connect(self._onPreferencesChanged)
            self._controller.getScene().getRoot().childrenChanged.connect(self._onSceneChanged)
            self.calculateColorSchemeLimits()
            self.calculateMaxLayers()
            self.calculateMaxPathsOnLayer(self._current_layer_num)
            if Platform.isOSX():
                if QOpenGLContext.currentContext() is None:
                    Logger.log('d', 'current context of OpenGL is empty on Mac OS X, will try to create shaders later')
                    CuraApplication.getInstance().callLater(lambda e=event: self.event(e))
                    return False
            layer_pass = self.getSimulationPass()
            renderer = self.getRenderer()
            if renderer is None:
                return False
            renderer.addRenderPass(layer_pass)
            nozzle = self.getNozzleNode()
            nozzle.setParent(self.getController().getScene().getRoot())
            nozzle.setVisible(False)
            Application.getInstance().globalContainerStackChanged.connect(self._onGlobalStackChanged)
            self._onGlobalStackChanged()
            if not self._simulationview_composite_shader:
                plugin_path = cast(str, PluginRegistry.getInstance().getPluginPath('SimulationView'))
                self._simulationview_composite_shader = OpenGL.getInstance().createShaderProgram(os.path.join(plugin_path, 'simulationview_composite.shader'))
                theme = CuraApplication.getInstance().getTheme()
                if theme is not None:
                    self._simulationview_composite_shader.setUniformValue('u_background_color', Color(*theme.getColor('viewport_background').getRgb()))
                    self._simulationview_composite_shader.setUniformValue('u_outline_color', Color(*theme.getColor('model_selection_outline').getRgb()))
            if not self._composite_pass:
                self._composite_pass = cast(CompositePass, renderer.getRenderPass('composite'))
            self._old_layer_bindings = self._composite_pass.getLayerBindings()[:]
            self._composite_pass.getLayerBindings().append('simulationview')
            self._old_composite_shader = self._composite_pass.getCompositeShader()
            self._composite_pass.setCompositeShader(self._simulationview_composite_shader)
            self._updateSliceWarningVisibility()
        elif event.type == Event.ViewDeactivateEvent:
            self._controller.getScene().getRoot().childrenChanged.disconnect(self._onSceneChanged)
            Application.getInstance().getPreferences().preferenceChanged.disconnect(self._onPreferencesChanged)
            self._slice_first_warning_message.hide()
            Application.getInstance().globalContainerStackChanged.disconnect(self._onGlobalStackChanged)
            if self._nozzle_node:
                self._nozzle_node.setParent(None)
            renderer = self.getRenderer()
            if renderer is None:
                return False
            if self._layer_pass is not None:
                renderer.removeRenderPass(self._layer_pass)
            if self._composite_pass:
                self._composite_pass.setLayerBindings(cast(List[str], self._old_layer_bindings))
                self._composite_pass.setCompositeShader(cast(ShaderProgram, self._old_composite_shader))
        return False

    def getCurrentLayerMesh(self):
        if False:
            return 10
        return self._current_layer_mesh

    def getCurrentLayerJumps(self):
        if False:
            print('Hello World!')
        return self._current_layer_jumps

    def _onGlobalStackChanged(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._global_container_stack = Application.getInstance().getGlobalContainerStack()
        if self._global_container_stack:
            self._extruder_count = self._global_container_stack.getProperty('machine_extruder_count', 'value')
            self.globalStackChanged.emit()

    def _onCurrentLayerNumChanged(self) -> None:
        if False:
            return 10
        self.calculateMaxPathsOnLayer(self._current_layer_num)
        scene = Application.getInstance().getController().getScene()
        scene.sceneChanged.emit(scene.getRoot())

    def _startUpdateTopLayers(self) -> None:
        if False:
            print('Hello World!')
        if not self._compatibility_mode:
            return
        if self._top_layers_job:
            self._top_layers_job.finished.disconnect(self._updateCurrentLayerMesh)
            self._top_layers_job.cancel()
        self.setBusy(True)
        self._top_layers_job = _CreateTopLayersJob(self._controller.getScene(), self._current_layer_num, self._solid_layers)
        self._top_layers_job.finished.connect(self._updateCurrentLayerMesh)
        self._top_layers_job.start()

    def _updateCurrentLayerMesh(self, job: '_CreateTopLayersJob') -> None:
        if False:
            return 10
        self.setBusy(False)
        if not job.getResult():
            return
        self.resetLayerData()
        self._current_layer_mesh = job.getResult().get('layers')
        if self._show_travel_moves:
            self._current_layer_jumps = job.getResult().get('jumps')
        self._controller.getScene().sceneChanged.emit(self._controller.getScene().getRoot())
        self._top_layers_job = None

    def _updateWithPreferences(self) -> None:
        if False:
            i = 10
            return i + 15
        self._solid_layers = int(Application.getInstance().getPreferences().getValue('view/top_layer_count'))
        self._only_show_top_layers = bool(Application.getInstance().getPreferences().getValue('view/only_show_top_layers'))
        self._compatibility_mode = self._evaluateCompatibilityMode()
        self.setSimulationViewType(int(float(Application.getInstance().getPreferences().getValue('layerview/layer_view_type'))))
        for (extruder_nr, extruder_opacity) in enumerate(Application.getInstance().getPreferences().getValue('layerview/extruder_opacities').split('|')):
            try:
                opacity = float(extruder_opacity)
            except ValueError:
                opacity = 1.0
            self.setExtruderOpacity(extruder_nr, opacity)
        self.setShowTravelMoves(bool(Application.getInstance().getPreferences().getValue('layerview/show_travel_moves')))
        self.setShowHelpers(bool(Application.getInstance().getPreferences().getValue('layerview/show_helpers')))
        self.setShowSkin(bool(Application.getInstance().getPreferences().getValue('layerview/show_skin')))
        self.setShowInfill(bool(Application.getInstance().getPreferences().getValue('layerview/show_infill')))
        self.setShowStarts(bool(Application.getInstance().getPreferences().getValue('layerview/show_starts')))
        self._startUpdateTopLayers()
        self.preferencesChanged.emit()

    def _onPreferencesChanged(self, preference: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        if preference not in {'view/top_layer_count', 'view/only_show_top_layers', 'view/force_layer_view_compatibility_mode', 'layerview/layer_view_type', 'layerview/extruder_opacities', 'layerview/show_travel_moves', 'layerview/show_helpers', 'layerview/show_skin', 'layerview/show_infill', 'layerview/show_starts'}:
            return
        self._updateWithPreferences()

    def _updateSliceWarningVisibility(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.getActivity() and (not CuraApplication.getInstance().getPreferences().getValue('general/auto_slice')) and CuraApplication.getInstance().getPreferences().getValue(self._no_layers_warning_preference):
            self._slice_first_warning_message.show()
        else:
            self._slice_first_warning_message.hide()

    def _onDontAskMeAgain(self, checked: bool) -> None:
        if False:
            i = 10
            return i + 15
        CuraApplication.getInstance().getPreferences().setValue(self._no_layers_warning_preference, not checked)

class _CreateTopLayersJob(Job):

    def __init__(self, scene: 'Scene', layer_number: int, solid_layers: int) -> None:
        if False:
            return 10
        super().__init__()
        self._scene = scene
        self._layer_number = layer_number
        self._solid_layers = solid_layers
        self._cancel = False

    def run(self) -> None:
        if False:
            while True:
                i = 10
        layer_data = None
        for node in DepthFirstIterator(self._scene.getRoot()):
            layer_data = node.callDecoration('getLayerData')
            if layer_data:
                break
        if self._cancel or not layer_data:
            return
        layer_mesh = MeshBuilder()
        for i in range(self._solid_layers):
            layer_number = self._layer_number - i
            if layer_number < 0:
                continue
            try:
                layer = layer_data.getLayer(layer_number).createMesh()
            except Exception:
                Logger.logException('w', 'An exception occurred while creating layer mesh.')
                return
            if not layer or layer.getVertices() is None:
                continue
            layer_mesh.addIndices(layer_mesh.getVertexCount() + layer.getIndices())
            layer_mesh.addVertices(layer.getVertices())
            brightness = numpy.ones((1, 4), dtype=numpy.float32) * (2.0 - i / self._solid_layers) / 2.0
            brightness[0, 3] = 1.0
            layer_mesh.addColors(layer.getColors() * brightness)
            if self._cancel:
                return
            Job.yieldThread()
        if self._cancel:
            return
        Job.yieldThread()
        jump_mesh = layer_data.getLayer(self._layer_number).createJumps()
        if not jump_mesh or jump_mesh.getVertices() is None:
            jump_mesh = None
        self.setResult({'layers': layer_mesh.build(), 'jumps': jump_mesh})

    def cancel(self) -> None:
        if False:
            i = 10
            return i + 15
        self._cancel = True
        super().cancel()