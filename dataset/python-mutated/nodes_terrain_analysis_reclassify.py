import os
import math
import bpy
from mathutils import Vector
from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty, EnumProperty, CollectionProperty, FloatVectorProperty
from bpy.types import PropertyGroup, UIList, Panel, Operator
from bpy.app.handlers import persistent
import logging
log = logging.getLogger(__name__)
from .utils import getBBOX
from ..core.utils.gradient import Color, Stop, Gradient
from ..core.maths.interpo import scale
from ..core.maths.kmeans1D import kmeans1d, getBreaks
svgGradientFolder = os.path.dirname(os.path.realpath(__file__)) + os.sep + 'rsrc' + os.sep + 'gradients' + os.sep
inMin = 0
inMax = 0
scn = None
obj = None
mat = None
node = None

class RECLASS_PG_color(PropertyGroup):

    def updStop(item, context):
        if False:
            print('Hello World!')
        if context.space_data is not None:
            if context.space_data.type == 'NODE_EDITOR':
                v = item.val
                i = item.idx
                node = context.active_node
                cr = node.color_ramp
                stops = cr.elements
                newPos = scale(v, inMin, inMax, 0, 1)
                if i + 1 == len(stops):
                    nextPos = 1
                else:
                    nextPos = stops[i + 1].position
                if i == 0:
                    prevPos = 0
                else:
                    prevPos = stops[i - 1].position
                if newPos > nextPos:
                    stops[i].position = nextPos
                    item.val = scale(nextPos, 0, 1, inMin, inMax)
                elif newPos < prevPos:
                    stops[i].position = prevPos
                    item.val = scale(prevPos, 0, 1, inMin, inMax)
                else:
                    stops[i].position = newPos

    def updColor(item, context):
        if False:
            i = 10
            return i + 15
        if context.space_data is not None:
            if context.space_data.type == 'NODE_EDITOR':
                color = item.color
                i = item.idx
                node = context.active_node
                cr = node.color_ramp
                stops = cr.elements
                stops[i].color = color
    idx: IntProperty()
    val: FloatProperty(update=updStop)
    color: FloatVectorProperty(subtype='COLOR', min=0, max=1, update=updColor, size=4)

def populateList(colorRampNode):
    if False:
        print('Hello World!')
    setBounds()
    if colorRampNode is not None:
        if colorRampNode.bl_idname == 'ShaderNodeValToRGB':
            bpy.context.scene.uiListCollec.clear()
            cr = colorRampNode.color_ramp
            for (i, stop) in enumerate(cr.elements):
                v = scale(stop.position, 0, 1, inMin, inMax)
                item = bpy.context.scene.uiListCollec.add()
                item.idx = i
                item.val = v
                item.color = stop.color

def updateAnalysisMode(scn, context):
    if False:
        for i in range(10):
            print('nop')
    if context.space_data.type == 'NODE_EDITOR':
        node = context.active_node
        populateList(node)

def setBounds():
    if False:
        i = 10
        return i + 15
    scn = bpy.context.scene
    mode = scn.analysisMode
    global inMin
    global inMax
    global obj
    if mode == 'HEIGHT':
        obj = bpy.context.view_layer.objects.active
        bbox = getBBOX.fromObj(obj)
        inMin = bbox['zmin']
        inMax = bbox['zmax']
    elif mode == 'SLOPE':
        inMin = 0
        inMax = 100
    elif mode == 'ASPECT':
        inMin = 0
        inMax = 360

@persistent
def scene_update(scn):
    if False:
        print('Hello World!')
    'keep colorramp node and reclass panel in synch'
    global obj
    global mat
    global node
    activeObj = bpy.context.view_layer.objects.active
    if activeObj is not None:
        activeMat = activeObj.active_material
        if activeMat is not None and activeMat.use_nodes:
            activeNode = activeMat.node_tree.nodes.active
            "\n\t\t\tdepsgraph = bpy.context.evaluated_depsgraph_get() #cause recursion depth error\n\t\t\tif depsgraph.id_type_updated('MATERIAL'):\n\t\t\t\tpopulateList(activeNode)\n\t\t\t"
            if obj != activeObj:
                obj = activeObj
                populateList(activeNode)
            if mat != activeMat:
                mat = activeMat
                populateList(activeNode)
            if node != activeNode:
                node = activeNode
                populateList(activeNode)

class RECLASS_UL_stops(UIList):

    def getAspectLabels(self):
        if False:
            print('Hello World!')
        vals = [round(item.val, 2) for item in bpy.context.scene.uiListCollec]
        if vals == [0, 45, 135, 225, 315]:
            return ['N', 'E', 'S', 'W', 'N']
        elif vals == [0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]:
            return ['N', 'N-E', 'E', 'S-E', 'S', 'S-W', 'W', 'N-W', 'N']
        elif vals == [0, 30, 90, 150, 210, 270, 330]:
            return ['N', 'N-E', 'S-E', 'S', 'S-W', 'N-W', 'N']
        elif vals == [0, 60, 120, 180, 240, 300, 360]:
            return ['N-E', 'E', 'S-E', 'S-W', 'W', 'N-W', 'N-E']
        elif vals == [0, 90, 270]:
            return ['N', 'S', 'N']
        elif vals == [0, 180]:
            return ['E', 'W']
        else:
            return False

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if False:
            i = 10
            return i + 15
        '\n\t\tcalled for each item of the collection visible in the list\n\t\tmust handle the three layout types \'DEFAULT\', \'COMPACT\' and \'GRID\'\n\t\tdata is the object containing the collection (in our case, the scene)\n\t\titem is the current drawn item of the collection (in our case a propertyGroup "customItem")\n\t\tindex is index of the current item in the collection (optional)\n\t\t'
        scn = bpy.context.scene
        mode = scn.analysisMode
        self.use_filter_show = False
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            if mode == 'ASPECT':
                aspectLabels = self.getAspectLabels()
                split = layout.split(factor=0.2)
                if aspectLabels:
                    split.label(text=aspectLabels[item.idx])
                else:
                    split.label(text=str(item.idx + 1))
                split = split.split(factor=0.4)
                split.prop(item, 'color', text='')
                split.prop(item, 'val', text='')
            else:
                split = layout.split(factor=0.2)
                split.label(text=str(item.idx + 1))
                split = split.split(factor=0.4)
                split.prop(item, 'color', text='')
                split.prop(item, 'val', text='')
        elif self.layout_type in {'GRID'}:
            layout.alignment = 'CENTER'

class RECLASS_PT_reclassify(Panel):
    """Creates a panel in the properties of node editor"""
    bl_label = 'Reclassify'
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'Item'

    def draw(self, context):
        if False:
            print('Hello World!')
        node = context.active_node
        if node is not None:
            if node.bl_idname == 'ShaderNodeValToRGB':
                layout = self.layout
                scn = context.scene
                layout.prop(scn, 'analysisMode')
                row = layout.row()
                row.template_list('RECLASS_UL_stops', '', scn, 'uiListCollec', scn, 'uiListIndex', rows=10)
                col = row.column(align=True)
                col.operator('reclass.list_add', text='', icon='ADD')
                col.operator('reclass.list_rm', text='', icon='REMOVE')
                col.operator('reclass.list_clear', text='', icon='FILE_PARENT')
                col.separator()
                col.operator('reclass.list_refresh', text='', icon='FILE_REFRESH')
                col.separator()
                col.operator('reclass.switch_interpolation', text='', icon='SMOOTHCURVE')
                col.operator('reclass.flip', text='', icon='ARROW_LEFTRIGHT')
                col.operator('reclass.quick_gradient', text='', icon='COLOR')
                col.operator('reclass.svg_gradient', text='', icon='GROUP_VCOL')
                col.operator('reclass.export_svg', text='', icon='FORWARD')
                col.separator()
                col.operator('reclass.auto', text='', icon='FULLSCREEN_ENTER')
                row = layout.row()
                row.label(text='min = ' + str(round(inMin, 2)))
                row.label(text='max = ' + str(round(inMax, 2)))
                row = layout.row()
                row.label(text='delta = ' + str(round(inMax - inMin, 2)))

class RECLASS_OT_switch_interpolation(Operator):
    """Switch color interpolation (continuous / discrete)"""
    bl_idname = 'reclass.switch_interpolation'
    bl_label = 'Switch color interpolation (continuous or discrete)'

    def execute(self, context):
        if False:
            return 10
        node = context.active_node
        cr = node.color_ramp
        cr.color_mode = 'RGB'
        if cr.interpolation != 'CONSTANT':
            cr.interpolation = 'CONSTANT'
        else:
            cr.interpolation = 'LINEAR'
        return {'FINISHED'}

class RECLASS_OT_flip(Operator):
    """Flip color ramp"""
    bl_idname = 'reclass.flip'
    bl_label = 'Flip color ramp'

    def execute(self, context):
        if False:
            while True:
                i = 10
        node = context.active_node
        cr = node.color_ramp
        stops = cr.elements
        revStops = []
        for (i, stop) in reversed(list(enumerate(stops))):
            revPos = 1 - stop.position
            color = tuple(stop.color)
            revStops.append((revPos, color))
        for (i, stop) in enumerate(stops):
            stop.color = revStops[i][1]
        populateList(node)
        return {'FINISHED'}

class RECLASS_OT_refresh(Operator):
    """Refresh list to match node setting"""
    bl_idname = 'reclass.list_refresh'
    bl_label = 'Populate list'

    def execute(self, context):
        if False:
            i = 10
            return i + 15
        node = context.active_node
        populateList(node)
        return {'FINISHED'}

class RECLASS_OT_clear(Operator):
    """Clear color ramp"""
    bl_idname = 'reclass.list_clear'
    bl_label = 'Clear list'

    def execute(self, context):
        if False:
            while True:
                i = 10
        node = context.active_node
        cr = node.color_ramp
        stops = cr.elements
        for stop in reversed(stops):
            if len(stops) > 1:
                stops.remove(stop)
            else:
                stop.position = 0
        populateList(node)
        return {'FINISHED'}

class RECLASS_OT_add(Operator):
    """Add stop"""
    bl_idname = 'reclass.list_add'
    bl_label = 'Add stop'

    def execute(self, context):
        if False:
            print('Hello World!')
        lst = bpy.context.scene.uiListCollec
        currentIdx = bpy.context.scene.uiListIndex
        if currentIdx > len(lst) - 1:
            currentIdx = 0
        node = context.active_node
        cr = node.color_ramp
        stops = cr.elements
        if len(stops) >= 32:
            self.report({'ERROR'}, 'Ramp is limited to 32 colors')
            return {'CANCELLED'}
        currentPos = stops[currentIdx].position
        if currentIdx == len(stops) - 1:
            nextPos = 1.0
        else:
            nextPos = stops[currentIdx + 1].position
        newPos = currentPos + (nextPos - currentPos) / 2
        stops.new(newPos)
        populateList(node)
        bpy.context.scene.uiListIndex = currentIdx + 1
        return {'FINISHED'}

class RECLASS_OT_rm(Operator):
    """Remove stop"""
    bl_idname = 'reclass.list_rm'
    bl_label = 'Remove Stop'

    def execute(self, context):
        if False:
            while True:
                i = 10
        currentIdx = bpy.context.scene.uiListIndex
        lst = bpy.context.scene.uiListCollec
        if currentIdx > len(lst) - 1:
            return {'CANCELLED'}
        node = context.active_node
        cr = node.color_ramp
        stops = cr.elements
        if len(stops) > 1:
            stops.remove(stops[currentIdx])
        populateList(node)
        if currentIdx > len(lst) - 1:
            bpy.context.scene.uiListIndex = currentIdx - 1
        return {'FINISHED'}

def clearRamp(stops, startColor=(0, 0, 0, 1), endColor=(1, 1, 1, 1), startPos=0, endPos=1):
    if False:
        i = 10
        return i + 15
    for stop in reversed(stops):
        if len(stops) > 1:
            stops.remove(stop)
        else:
            first = stop
            first.position = startPos
            first.color = startColor
    last = stops.new(endPos)
    last.color = endColor
    return (first, last)

def getValues():
    if False:
        return 10
    'Return mesh data values (z, slope or az) for classification'
    scn = bpy.context.scene
    obj = bpy.context.view_layer.objects.active
    mesh = obj.to_mesh()
    mesh.transform(obj.matrix_world)
    mode = scn.analysisMode
    if mode == 'HEIGHT':
        values = [vertex.co.z for vertex in mesh.vertices]
    elif mode == 'SLOPE':
        z = Vector((0, 0, 1))
        m = obj.matrix_world
        values = [math.degrees(z.angle(m * face.normal)) for face in mesh.polygons]
    elif mode == 'ASPECT':
        y = Vector((0, 1, 0))
        m = obj.matrix_world
        values = []
        for face in mesh.polygons:
            normal = face.normal.copy()
            normal.z = 0
            try:
                a = math.degrees(y.angle(m * normal))
            except ValueError:
                pass
            else:
                if normal.x < 0:
                    a = 360 - a
                values.append(a)
    values.sort()
    obj.to_mesh_clear()
    return values

class RECLASS_OT_auto(Operator):
    """Auto reclass by equal interval or fixed classe number"""
    bl_idname = 'reclass.auto'
    bl_label = 'Reclass by equal interval or fixed classe number'
    autoReclassMode: EnumProperty(name='Mode', description='Select auto reclassify mode', items=[('CLASSES_NB', 'Fixed classes number', 'Define the expected number of classes'), ('EQUAL_STEP', 'Equal interval value', 'Define step value between classes'), ('TARGET_STEP', 'Target interval value', 'Define target step value that stops will match'), ('QUANTILE', 'Quantile', 'Assigns the same number of data values to each class.'), ('1DKMEANS', 'Natural breaks', 'kmeans clustering optimized for one dimensional data'), ('ASPECT', 'Aspect reclassification', 'Value define the number of azimuth')])
    color1: FloatVectorProperty(name='Start color', subtype='COLOR', min=0, max=1, size=4)
    color2: FloatVectorProperty(name='End color', subtype='COLOR', min=0, max=1, size=4)
    value: IntProperty(name='Value', default=4)

    def invoke(self, context, event):
        if False:
            return 10
        node = context.active_node
        cr = node.color_ramp
        stops = cr.elements
        self.color1 = stops[0].color
        self.color2 = stops[len(stops) - 1].color
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def execute(self, context):
        if False:
            i = 10
            return i + 15
        node = context.active_node
        cr = node.color_ramp
        cr.color_mode = 'RGB'
        cr.interpolation = 'LINEAR'
        stops = cr.elements
        startColor = self.color1
        endColor = self.color2
        if self.autoReclassMode == 'TARGET_STEP':
            interval = self.value
            delta = inMax - inMin
            nbClasses = math.ceil(delta / interval)
            if nbClasses >= 32:
                self.report({'ERROR'}, 'Ramp is limited to 32 colors')
                return {'CANCELLED'}
            clearRamp(stops, startColor, endColor)
            nextStop = inMin + interval - inMin % interval
            while nextStop < inMax:
                position = scale(nextStop, inMin, inMax, 0, 1)
                stop = stops.new(position)
                nextStop += interval
        if self.autoReclassMode == 'EQUAL_STEP':
            interval = self.value
            delta = inMax - inMin
            nbClasses = math.ceil(delta / interval)
            if nbClasses >= 32:
                self.report({'ERROR'}, 'Ramp is limited to 32 colors')
                return {'CANCELLED'}
            clearRamp(stops, startColor, endColor)
            val = inMin
            for i in range(nbClasses - 1):
                val += interval
                position = scale(val, inMin, inMax, 0, 1)
                stop = stops.new(position)
        if self.autoReclassMode == 'CLASSES_NB':
            nbClasses = self.value
            if nbClasses >= 32:
                self.report({'ERROR'}, 'Ramp is limited to 32 colors')
                return {'CANCELLED'}
            delta = inMax - inMin
            if nbClasses >= delta:
                self.report({'ERROR'}, 'Too many classes')
                return {'CANCELLED'}
            clearRamp(stops, startColor, endColor)
            interval = delta / nbClasses
            val = inMin
            for i in range(nbClasses - 1):
                val += interval
                position = scale(val, inMin, inMax, 0, 1)
                stop = stops.new(position)
        if self.autoReclassMode == 'ASPECT':
            bpy.context.scene.analysisMode = 'ASPECT'
            delta = inMax - inMin
            interval = 360 / self.value
            nbClasses = self.value
            if nbClasses >= 32:
                self.report({'ERROR'}, 'Ramp is limited to 32 colors')
                return {'CANCELLED'}
            (first, last) = clearRamp(stops, startColor, endColor)
            offset = interval / 2
            intervalNorm = scale(interval, inMin, inMax, 0, 1)
            offsetNorm = scale(offset, inMin, inMax, 0, 1)
            last.position -= intervalNorm + offsetNorm
            val = 0
            for i in range(nbClasses - 2):
                if i == 0:
                    val += offset
                else:
                    val += interval
                position = scale(val, inMin, inMax, 0, 1)
                stop = stops.new(position)
            stop = stops.new(1 - offsetNorm)
            stop.color = first.color
            cr.interpolation = 'CONSTANT'
        if self.autoReclassMode == 'QUANTILE':
            nbClasses = self.value
            values = getValues()
            if nbClasses >= 32:
                self.report({'ERROR'}, 'Ramp is limited to 32 colors')
                return {'CANCELLED'}
            if nbClasses >= len(values):
                self.report({'ERROR'}, 'Too many classes')
                return {'CANCELLED'}
            clearRamp(stops, startColor, endColor)
            n = len(values)
            q = int(n / nbClasses)
            cumulative_q = q
            previousVal = scale(0, 0, 1, inMin, inMax)
            for i in range(nbClasses - 1):
                val = values[cumulative_q]
                if val != previousVal:
                    position = scale(val, inMin, inMax, 0, 1)
                    stop = stops.new(position)
                    previousVal = val
                cumulative_q += q
        if self.autoReclassMode == '1DKMEANS':
            nbClasses = self.value
            values = getValues()
            if nbClasses >= 32:
                self.report({'ERROR'}, 'Ramp is limited to 32 colors')
                return {'CANCELLED'}
            if nbClasses >= len(values):
                self.report({'ERROR'}, 'Too many classes')
                return {'CANCELLED'}
            clearRamp(stops, startColor, endColor)
            clusters = kmeans1d(values, nbClasses)
            for val in getBreaks(values, clusters):
                position = scale(val, inMin, inMax, 0, 1)
                stop = stops.new(position)
        populateList(node)
        return {'FINISHED'}
colorSpaces = [('RGB', 'RGB', 'RGB color space'), ('HSV', 'HSV', 'HSV color space')]
interpoMethods = [('LINEAR', 'Linear', 'Linear interpolation'), ('SPLINE', 'Spline', "Spline interpolation (Akima's method)"), ('DISCRETE', 'Discrete', 'No interpolation (return previous color)'), ('NEAREST', 'Nearest', 'No interpolation (return nearest color)')]

class RECLASS_PG_color_preview(PropertyGroup):
    color: FloatVectorProperty(subtype='COLOR', min=0, max=1, size=4)

class RECLASS_OT_quick_gradient(Operator):
    """Quick colors gradient edit"""
    bl_idname = 'reclass.quick_gradient'
    bl_label = 'Quick colors gradient edit'
    colorSpace: EnumProperty(name='Space', description='Select interpolation color space', items=colorSpaces)
    method: EnumProperty(name='Method', description='Select interpolation method', items=interpoMethods)

    def check(self, context):
        if False:
            i = 10
            return i + 15
        return True

    def initPreview(self, context):
        if False:
            for i in range(10):
                print('nop')
        context.scene.colorRampPreview.clear()
        node = context.active_node
        cr = node.color_ramp
        stops = cr.elements
        if self.fitGradient:
            (minPos, maxPos) = (stops[0].position, stops[-1].position)
            delta = maxPos - minPos
        else:
            delta = 1
        offset = delta / (self.nbColors - 1)
        position = 0
        for i in range(self.nbColors):
            item = bpy.context.scene.colorRampPreview.add()
            item.color = cr.evaluate(position)
            position += offset
        return

    def updatePreview(self, context):
        if False:
            while True:
                i = 10
        colorItems = bpy.context.scene.colorRampPreview
        nb = len(colorItems)
        if nb == self.nbColors:
            return
        delta = abs(self.nbColors - nb)
        for i in range(delta):
            if self.nbColors > nb:
                item = colorItems.add()
                item.color = colorItems[-2].color
            else:
                colorItems.remove(nb - 1)
    fitGradient: BoolProperty(update=initPreview)
    nbColors: IntProperty(name='Number of colors', description='Set the number of colors needed to define the quick quadient', min=2, default=4, update=updatePreview)

    def invoke(self, context, event):
        if False:
            print('Hello World!')
        self.initPreview(context)
        wm = context.window_manager
        return wm.invoke_props_dialog(self, width=200, height=200)

    def draw(self, context):
        if False:
            print('Hello World!')
        layout = self.layout
        layout.prop(self, 'colorSpace', text='Space')
        layout.prop(self, 'method', text='Method')
        layout.prop(self, 'fitGradient', text='Fit gradient to min/max positions')
        layout.prop(self, 'nbColors', text='Number of colors')
        row = layout.row(align=True)
        colorItems = context.scene.colorRampPreview
        for i in range(self.nbColors):
            colorItem = colorItems[i]
            row.prop(colorItem, 'color', text='')

    def execute(self, context):
        if False:
            for i in range(10):
                print('nop')
        colorList = context.scene.colorRampPreview
        colorRamp = Gradient()
        nbColors = len(colorList)
        offset = 1 / (nbColors - 1)
        position = 0
        for (i, item) in enumerate(colorList):
            color = Color(list(item.color), 'rgb')
            colorRamp.addStop(round(position, 4), color)
            position += offset
        node = context.active_node
        cr = node.color_ramp
        stops = cr.elements
        if self.fitGradient:
            (minPos, maxPos) = (stops[0].position, stops[-1].position)
            colorRamp.rescale(minPos, maxPos)
        for stop in stops:
            stop.color = colorRamp.evaluate(stop.position, self.colorSpace, self.method).rgba
        if self.colorSpace == 'HSV':
            cr.color_mode = 'HSV'
        else:
            cr.color_mode = 'RGB'
        populateList(node)
        return {'FINISHED'}

def filesList(inFolder, ext):
    if False:
        return 10
    if not os.path.exists(inFolder):
        return []
    lst = os.listdir(inFolder)
    extLst = [elem for elem in lst if os.path.splitext(elem)[1] == ext]
    extLst.sort()
    return extLst
svgFiles = filesList(svgGradientFolder, '.svg')
colorPreviewRange = 20

class RECLASS_OT_svg_gradient(Operator):
    """Define colors gradient with presets"""
    bl_idname = 'reclass.svg_gradient'
    bl_label = 'Define colors gradient with presets'

    def listSVG(self, context):
        if False:
            while True:
                i = 10
        svgs = []
        for (index, svg) in enumerate(svgFiles):
            svgs.append((str(index), os.path.splitext(svg)[0], svgGradientFolder + svg))
        return svgs

    def updatePreview(self, context):
        if False:
            i = 10
            return i + 15
        if len(self.colorPresets) == 0:
            return
        enumIdx = int(self.colorPresets)
        path = svgGradientFolder + svgFiles[enumIdx]
        colorRamp = Gradient(path)
        nbColors = colorPreviewRange
        interpoGradient = colorRamp.getRangeColor(nbColors, self.colorSpace, self.method)
        for (i, stop) in enumerate(interpoGradient.stops):
            item = bpy.context.scene.colorRampPreview[i]
            item.color = stop.color.rgba
        return
    colorPresets: EnumProperty(name='preset', description='Select a color ramp preset', items=listSVG, update=updatePreview)
    colorSpace: EnumProperty(name='Space', description='Select interpolation color space', items=colorSpaces, update=updatePreview)
    method: EnumProperty(name='Method', description='Select interpolation method', items=interpoMethods, update=updatePreview)
    fitGradient: BoolProperty()

    def invoke(self, context, event):
        if False:
            for i in range(10):
                print('nop')
        context.scene.colorRampPreview.clear()
        for i in range(colorPreviewRange):
            bpy.context.scene.colorRampPreview.add()
        self.updatePreview(context)
        wm = context.window_manager
        return wm.invoke_props_dialog(self, width=200, height=200)

    def draw(self, context):
        if False:
            i = 10
            return i + 15
        layout = self.layout
        layout.prop(self, 'colorSpace')
        layout.prop(self, 'method')
        layout.prop(self, 'colorPresets', text='')
        row = layout.row(align=True)
        row.enabled = False
        for item in context.scene.colorRampPreview:
            row.prop(item, 'color', text='')
        row = layout.row()
        row.prop(self, 'fitGradient', text='Fit gradient to min/max positions')

    def execute(self, context):
        if False:
            for i in range(10):
                print('nop')
        if len(self.colorPresets) == 0:
            return {'CANCELLED'}
        enumIdx = int(self.colorPresets)
        path = svgGradientFolder + svgFiles[enumIdx]
        colorRamp = Gradient(path)
        node = context.active_node
        cr = node.color_ramp
        stops = cr.elements
        if self.fitGradient:
            (minPos, maxPos) = (stops[0].position, stops[-1].position)
            colorRamp.rescale(minPos, maxPos)
        for stop in stops:
            stop.color = colorRamp.evaluate(stop.position, self.colorSpace, self.method).rgba
        if self.colorSpace == 'HSV':
            cr.color_mode = 'HSV'
        else:
            cr.color_mode = 'RGB'
        populateList(node)
        return {'FINISHED'}

class RECLASS_OT_export_svg(Operator):
    """Export current gradient to SVG file"""
    bl_idname = 'reclass.export_svg'
    bl_label = 'Export current gradient to SVG file'
    name: StringProperty(description='Put name of SVG file')
    n: IntProperty(default=5, description='Select expected number of interpolate colors')
    gradientType: EnumProperty(name='Build method', description='Select methods to build gradient', items=[('SELF_STOPS', 'Use actual stops', ''), ('INTERPOLATE', 'Interpolate n colors', '')])
    makeDiscrete: BoolProperty(name='Make discrete', description='Build discrete svg gradient')
    colorSpace: EnumProperty(name='Color space', description='Select interpolation color space', items=colorSpaces)
    method: EnumProperty(name='Interp. method', description='Select interpolation method', items=interpoMethods)

    def check(self, context):
        if False:
            i = 10
            return i + 15
        return True

    def invoke(self, context, event):
        if False:
            i = 10
            return i + 15
        wm = context.window_manager
        return wm.invoke_props_dialog(self, width=250, height=200)

    def draw(self, context):
        if False:
            for i in range(10):
                print('nop')
        layout = self.layout
        layout.prop(self, 'name', text='Name')
        layout.prop(self, 'gradientType')
        layout.prop(self, 'makeDiscrete')
        if self.gradientType == 'INTERPOLATE':
            layout.separator()
            layout.label(text='Interpolation options')
            layout.prop(self, 'colorSpace', text='Color space')
            layout.prop(self, 'method', text='Method')
            layout.prop(self, 'n', text='Number of colors')

    def execute(self, context):
        if False:
            return 10
        node = context.active_node
        cr = node.color_ramp
        stops = cr.elements
        colorRamp = Gradient()
        for stop in stops:
            color = Color(list(stop.color), 'rgba')
            colorRamp.addStop(stop.position, color)
        svgPath = svgGradientFolder + self.name + '.svg'
        if self.gradientType == 'INTERPOLATE':
            interpoGradient = colorRamp.getRangeColor(self.n, self.colorSpace, self.method)
            interpoGradient.exportSVG(svgPath, self.makeDiscrete)
        elif self.gradientType == 'SELF_STOPS':
            colorRamp.exportSVG(svgPath, self.makeDiscrete)
        global svgFiles
        svgFiles = filesList(svgGradientFolder, '.svg')
        return {'FINISHED'}
classes = [RECLASS_PG_color, RECLASS_PG_color_preview, RECLASS_UL_stops, RECLASS_PT_reclassify, RECLASS_OT_switch_interpolation, RECLASS_OT_flip, RECLASS_OT_refresh, RECLASS_OT_clear, RECLASS_OT_add, RECLASS_OT_rm, RECLASS_OT_auto, RECLASS_OT_quick_gradient, RECLASS_OT_svg_gradient, RECLASS_OT_export_svg]

def register():
    if False:
        i = 10
        return i + 15
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError as e:
            log.warning('{} is already registered, now unregister and retry... '.format(cls))
            bpy.utils.unregister_class(cls)
            bpy.utils.register_class(cls)
    bpy.types.Scene.uiListCollec = CollectionProperty(type=RECLASS_PG_color)
    bpy.types.Scene.uiListIndex = IntProperty()
    bpy.types.Scene.colorRampPreview = CollectionProperty(type=RECLASS_PG_color_preview)
    bpy.app.handlers.depsgraph_update_post.append(scene_update)
    bpy.types.Scene.analysisMode = EnumProperty(name='Mode', description='Choose the type of analysis this material do', items=[('HEIGHT', 'Height', 'Height analysis'), ('SLOPE', 'Slope', 'Slope analysis'), ('ASPECT', 'Aspect', 'Aspect analysis')], update=updateAnalysisMode)

def unregister():
    if False:
        i = 10
        return i + 15
    del bpy.types.Scene.analysisMode
    del bpy.types.Scene.uiListCollec
    del bpy.types.Scene.uiListIndex
    del bpy.types.Scene.colorRampPreview
    bpy.app.handlers.depsgraph_update_post.clear()
    for cls in classes:
        bpy.utils.unregister_class(cls)