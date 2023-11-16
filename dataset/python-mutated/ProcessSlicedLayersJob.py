import gc
import sys
from UM.Job import Job
from UM.Application import Application
from UM.Mesh.MeshData import MeshData
from UM.View.GL.OpenGLContext import OpenGLContext
from UM.Message import Message
from UM.i18n import i18nCatalog
from UM.Logger import Logger
from UM.Math.Vector import Vector
from cura.Scene.BuildPlateDecorator import BuildPlateDecorator
from cura.Scene.CuraSceneNode import CuraSceneNode
from cura.Settings.ExtruderManager import ExtruderManager
from cura import LayerDataBuilder
from cura import LayerDataDecorator
from cura import LayerPolygon
import numpy
from time import time
from cura.Machines.Models.ExtrudersModel import ExtrudersModel
catalog = i18nCatalog('cura')

def colorCodeToRGBA(color_code):
    if False:
        return 10
    'Return a 4-tuple with floats 0-1 representing the html color code\n\n    :param color_code: html color code, i.e. "#FF0000" -> red\n    '
    if color_code is None:
        Logger.log('w', 'Unable to convert color code, returning default')
        return [0, 0, 0, 1]
    return [int(color_code[1:3], 16) / 255, int(color_code[3:5], 16) / 255, int(color_code[5:7], 16) / 255, 1.0]

class ProcessSlicedLayersJob(Job):

    def __init__(self, layers):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._layers = layers
        self._scene = Application.getInstance().getController().getScene()
        self._progress_message = Message(catalog.i18nc('@info:status', 'Processing Layers'), 0, False, -1)
        self._abort_requested = False
        self._build_plate_number = None

    def abort(self):
        if False:
            print('Hello World!')
        'Aborts the processing of layers.\n\n        This abort is made on a best-effort basis, meaning that the actual\n        job thread will check once in a while to see whether an abort is\n        requested and then stop processing by itself. There is no guarantee\n        that the abort will stop the job any time soon or even at all.\n        '
        self._abort_requested = True

    def setBuildPlate(self, new_value):
        if False:
            return 10
        self._build_plate_number = new_value

    def getBuildPlate(self):
        if False:
            i = 10
            return i + 15
        return self._build_plate_number

    def run(self):
        if False:
            i = 10
            return i + 15
        Logger.log('d', 'Processing new layer for build plate %s...' % self._build_plate_number)
        start_time = time()
        view = Application.getInstance().getController().getActiveView()
        if view.getPluginId() == 'SimulationView':
            view.resetLayerData()
            self._progress_message.show()
            Job.yieldThread()
            if self._abort_requested:
                if self._progress_message:
                    self._progress_message.hide()
                return
        Application.getInstance().getController().activeViewChanged.connect(self._onActiveViewChanged)
        new_node = CuraSceneNode(no_setting_override=True)
        new_node.addDecorator(BuildPlateDecorator(self._build_plate_number))
        gc.collect()
        mesh = MeshData()
        layer_data = LayerDataBuilder.LayerDataBuilder()
        layer_count = len(self._layers)
        min_layer_number = sys.maxsize
        negative_layers = 0
        for layer in self._layers:
            if layer.repeatedMessageCount('path_segment') > 0:
                if layer.id < min_layer_number:
                    min_layer_number = layer.id
                if layer.id < 0:
                    negative_layers += 1
        current_layer = 0
        for layer in self._layers:
            if layer.id < min_layer_number:
                continue
            abs_layer_number = layer.id - min_layer_number
            if layer.id >= 0 and negative_layers != 0:
                abs_layer_number += min_layer_number + negative_layers
            layer_data.addLayer(abs_layer_number)
            this_layer = layer_data.getLayer(abs_layer_number)
            layer_data.setLayerHeight(abs_layer_number, layer.height)
            layer_data.setLayerThickness(abs_layer_number, layer.thickness)
            for p in range(layer.repeatedMessageCount('path_segment')):
                polygon = layer.getRepeatedMessage('path_segment', p)
                extruder = polygon.extruder
                line_types = numpy.fromstring(polygon.line_type, dtype='u1')
                line_types = line_types.reshape((-1, 1))
                points = numpy.fromstring(polygon.points, dtype='f4')
                if polygon.point_type == 0:
                    points = points.reshape((-1, 2))
                else:
                    points = points.reshape((-1, 3))
                line_widths = numpy.fromstring(polygon.line_width, dtype='f4')
                line_widths = line_widths.reshape((-1, 1))
                line_thicknesses = numpy.fromstring(polygon.line_thickness, dtype='f4')
                line_thicknesses = line_thicknesses.reshape((-1, 1))
                line_feedrates = numpy.fromstring(polygon.line_feedrate, dtype='f4')
                line_feedrates = line_feedrates.reshape((-1, 1))
                new_points = numpy.empty((len(points), 3), numpy.float32)
                if polygon.point_type == 0:
                    new_points[:, 0] = points[:, 0]
                    new_points[:, 1] = layer.height / 1000
                    new_points[:, 2] = -points[:, 1]
                else:
                    new_points[:, 0] = points[:, 0]
                    new_points[:, 1] = points[:, 2]
                    new_points[:, 2] = -points[:, 1]
                this_poly = LayerPolygon.LayerPolygon(extruder, line_types, new_points, line_widths, line_thicknesses, line_feedrates)
                this_poly.buildCache()
                this_layer.polygons.append(this_poly)
                Job.yieldThread()
            Job.yieldThread()
            current_layer += 1
            progress = current_layer / layer_count * 99
            if self._abort_requested:
                if self._progress_message:
                    self._progress_message.hide()
                return
            if self._progress_message:
                self._progress_message.setProgress(progress)
        global_container_stack = Application.getInstance().getGlobalContainerStack()
        manager = ExtruderManager.getInstance()
        extruders = manager.getActiveExtruderStacks()
        if extruders:
            material_color_map = numpy.zeros((len(extruders), 4), dtype=numpy.float32)
            for extruder in extruders:
                position = int(extruder.getMetaDataEntry('position', default='0'))
                try:
                    default_color = ExtrudersModel.defaultColors[position]
                except IndexError:
                    default_color = '#e0e000'
                color_code = extruder.material.getMetaDataEntry('color_code', default=default_color)
                color = colorCodeToRGBA(color_code)
                material_color_map[position, :] = color
        else:
            material_color_map = numpy.zeros((1, 4), dtype=numpy.float32)
            color_code = global_container_stack.material.getMetaDataEntry('color_code', default='#e0e000')
            color = colorCodeToRGBA(color_code)
            material_color_map[0, :] = color
        if OpenGLContext.isLegacyOpenGL() or bool(Application.getInstance().getPreferences().getValue('view/force_layer_view_compatibility_mode')):
            line_type_brightness = 0.5
        else:
            line_type_brightness = 1.0
        layer_mesh = layer_data.build(material_color_map, line_type_brightness)
        if self._abort_requested:
            if self._progress_message:
                self._progress_message.hide()
            return
        decorator = LayerDataDecorator.LayerDataDecorator()
        decorator.setLayerData(layer_mesh)
        new_node.addDecorator(decorator)
        new_node.setMeshData(mesh)
        new_node_parent = Application.getInstance().getBuildVolume()
        new_node.setParent(new_node_parent)
        settings = Application.getInstance().getGlobalContainerStack()
        if not settings.getProperty('machine_center_is_zero', 'value'):
            new_node.setPosition(Vector(-settings.getProperty('machine_width', 'value') / 2, 0.0, settings.getProperty('machine_depth', 'value') / 2))
        if self._progress_message:
            self._progress_message.setProgress(100)
        if self._progress_message:
            self._progress_message.hide()
        self._layers = None
        Logger.log('d', 'Processing layers took %s seconds', time() - start_time)

    def _onActiveViewChanged(self):
        if False:
            while True:
                i = 10
        if self.isRunning():
            if Application.getInstance().getController().getActiveView().getPluginId() == 'SimulationView':
                if not self._progress_message:
                    self._progress_message = Message(catalog.i18nc('@info:status', 'Processing Layers'), 0, False, 0, catalog.i18nc('@info:title', 'Information'))
                if self._progress_message.getProgress() != 100:
                    self._progress_message.show()
            elif self._progress_message:
                self._progress_message.hide()