"""

The FilterManager is a convenience class that helps with the creation
of render-to-texture buffers for image postprocessing applications.

See :ref:`generalized-image-filters` for information on how to use this class.

Still need to implement:

* Make sure sort-order of buffers is correct.
* Matching buffer size to original region instead of original window.
* Intermediate layer creation.
* Handling of window clears.
* Resizing of windows.
* Do something about window-size roundoff problems.

"""
from panda3d.core import NodePath
from panda3d.core import Texture
from panda3d.core import CardMaker
from panda3d.core import GraphicsPipe, GraphicsOutput
from panda3d.core import WindowProperties, FrameBufferProperties
from panda3d.core import Camera
from panda3d.core import OrthographicLens
from panda3d.core import AuxBitplaneAttrib
from panda3d.core import LightRampAttrib
from direct.directnotify.DirectNotifyGlobal import directNotify
from direct.showbase.DirectObject import DirectObject
__all__ = ['FilterManager']

class FilterManager(DirectObject):
    notify = None

    def __init__(self, win, cam, forcex=0, forcey=0):
        if False:
            while True:
                i = 10
        " The FilterManager constructor requires you to provide\n        a window which is rendering a scene, and the camera which is\n        used by that window to render the scene.  These are henceforth\n        called the 'original window' and the 'original camera.' "
        if FilterManager.notify is None:
            FilterManager.notify = directNotify.newCategory('FilterManager')
        region = None
        for dr in win.getDisplayRegions():
            drcam = dr.getCamera()
            if drcam == cam:
                region = dr
        if region is None:
            self.notify.error('Could not find appropriate DisplayRegion to filter')
            return
        self.win = win
        self.forcex = forcex
        self.forcey = forcey
        self.engine = win.getGsg().getEngine()
        self.region = region
        self.wclears = self.getClears(self.win)
        self.rclears = self.getClears(self.region)
        self.camera = cam
        self.caminit = cam.node().getInitialState()
        self.camstate = self.caminit
        self.buffers = []
        self.sizes = []
        self.nextsort = self.win.getSort() - 9
        self.basex = 0
        self.basey = 0
        self.accept('window-event', self.windowEvent)

    def getClears(self, region):
        if False:
            return 10
        clears = []
        for i in range(GraphicsOutput.RTPCOUNT):
            clears.append((region.getClearActive(i), region.getClearValue(i)))
        return clears

    def setClears(self, region, clears):
        if False:
            i = 10
            return i + 15
        for i in range(GraphicsOutput.RTPCOUNT):
            (active, value) = clears[i]
            region.setClearActive(i, active)
            region.setClearValue(i, value)

    def setStackedClears(self, region, clears0, clears1):
        if False:
            print('Hello World!')
        clears = []
        for i in range(GraphicsOutput.RTPCOUNT):
            (active, value) = clears0[i]
            if not active:
                (active, value) = clears1[i]
            region.setClearActive(i, active)
            region.setClearValue(i, value)
        return clears

    def isFullscreen(self):
        if False:
            return 10
        return self.region.getLeft() == 0.0 and self.region.getRight() == 1.0 and (self.region.getBottom() == 0.0) and (self.region.getTop() == 1.0)

    def getScaledSize(self, mul, div, align):
        if False:
            while True:
                i = 10
        ' Calculate the size of the desired window. Not public. '
        winx = self.forcex
        winy = self.forcey
        if winx == 0:
            winx = self.win.getXSize()
        if winy == 0:
            winy = self.win.getYSize()
        if div != 1:
            winx = (winx + align - 1) // align * align
            winy = (winy + align - 1) // align * align
            winx = winx // div
            winy = winy // div
        if mul != 1:
            winx = int(round(winx * mul))
            winy = int(round(winy * mul))
        return (winx, winy)

    def renderSceneInto(self, depthtex=None, colortex=None, auxtex=None, auxbits=0, textures=None, fbprops=None, clamping=None):
        if False:
            return 10
        ' Causes the scene to be rendered into the supplied textures\n        instead of into the original window.  Puts a fullscreen quad\n        into the original window to show the render-to-texture results.\n        Returns the quad.  Normally, the caller would then apply a\n        shader to the quad.\n\n        To elaborate on how this all works:\n\n        * An offscreen buffer is created.  It is set up to mimic\n          the original display region - it is the same size,\n          uses the same clear colors, and contains a DisplayRegion\n          that uses the original camera.\n\n        * A fullscreen quad and an orthographic camera to render\n          that quad are both created.  The original camera is\n          removed from the original window, and in its place, the\n          orthographic quad-camera is installed.\n\n        * The fullscreen quad is textured with the data from the\n          offscreen buffer.  A shader is applied that tints the\n          results pink.\n\n        * Automatic shader generation NOT enabled.\n          If you have a filter that depends on a render target from\n          the auto-shader, you either need to set an auto-shader\n          attrib on the main camera or scene, or, you need to provide\n          these outputs in your own shader.\n\n        * All clears are disabled on the original display region.\n          If the display region fills the whole window, then clears\n          are disabled on the original window as well.  It is\n          assumed that rendering the full-screen quad eliminates\n          the need to do clears.\n\n        Hence, the original window which used to contain the actual\n        scene, now contains a pink-tinted quad with a texture of the\n        scene.  It is assumed that the user will replace the shader\n        on the quad with a more interesting filter. '
        if textures:
            colortex = textures.get('color', None)
            depthtex = textures.get('depth', None)
            auxtex = textures.get('aux', None)
            auxtex0 = textures.get('aux0', auxtex)
            auxtex1 = textures.get('aux1', None)
        else:
            auxtex0 = auxtex
            auxtex1 = None
        if colortex is None:
            colortex = Texture('filter-base-color')
            colortex.setWrapU(Texture.WMClamp)
            colortex.setWrapV(Texture.WMClamp)
        texgroup = (depthtex, colortex, auxtex0, auxtex1)
        (winx, winy) = self.getScaledSize(1, 1, 1)
        if fbprops is not None:
            buffer = self.createBuffer('filter-base', winx, winy, texgroup, fbprops=fbprops)
        else:
            buffer = self.createBuffer('filter-base', winx, winy, texgroup)
        if buffer is None:
            return None
        cm = CardMaker('filter-base-quad')
        cm.setFrameFullscreenQuad()
        quad = NodePath(cm.generate())
        quad.setDepthTest(0)
        quad.setDepthWrite(0)
        quad.setTexture(colortex)
        quad.setColor(1, 0.5, 0.5, 1)
        cs = NodePath('dummy')
        cs.setState(self.camstate)
        if auxbits:
            cs.setAttrib(AuxBitplaneAttrib.make(auxbits))
        if clamping is False:
            cs.setAttrib(LightRampAttrib.make_identity())
        self.camera.node().setInitialState(cs.getState())
        quadcamnode = Camera('filter-quad-cam')
        lens = OrthographicLens()
        lens.setFilmSize(2, 2)
        lens.setFilmOffset(0, 0)
        lens.setNearFar(-1000, 1000)
        quadcamnode.setLens(lens)
        quadcam = quad.attachNewNode(quadcamnode)
        self.region.setCamera(quadcam)
        self.setStackedClears(buffer, self.rclears, self.wclears)
        if auxtex0:
            buffer.setClearActive(GraphicsOutput.RTPAuxRgba0, 1)
            buffer.setClearValue(GraphicsOutput.RTPAuxRgba0, (0.5, 0.5, 1.0, 0.0))
        if auxtex1:
            buffer.setClearActive(GraphicsOutput.RTPAuxRgba1, 1)
        self.region.disableClears()
        if self.isFullscreen():
            self.win.disableClears()
        dr = buffer.makeDisplayRegion()
        dr.disableClears()
        dr.setCamera(self.camera)
        dr.setActive(1)
        self.buffers.append(buffer)
        self.sizes.append((1, 1, 1))
        return quad

    def renderQuadInto(self, name='filter-stage', mul=1, div=1, align=1, depthtex=None, colortex=None, auxtex0=None, auxtex1=None, fbprops=None):
        if False:
            for i in range(10):
                print('nop')
        " Creates an offscreen buffer for an intermediate\n        computation. Installs a quad into the buffer.  Returns\n        the fullscreen quad.  The size of the buffer is initially\n        equal to the size of the main window.  The parameters 'mul',\n        'div', and 'align' can be used to adjust that size. "
        texgroup = (depthtex, colortex, auxtex0, auxtex1)
        (winx, winy) = self.getScaledSize(mul, div, align)
        depthbits = int(depthtex is not None)
        if fbprops is not None:
            buffer = self.createBuffer(name, winx, winy, texgroup, depthbits, fbprops=fbprops)
        else:
            buffer = self.createBuffer(name, winx, winy, texgroup, depthbits)
        if buffer is None:
            return None
        cm = CardMaker('filter-stage-quad')
        cm.setFrameFullscreenQuad()
        quad = NodePath(cm.generate())
        quad.setDepthTest(0)
        quad.setDepthWrite(0)
        quad.setColor(1, 0.5, 0.5, 1)
        quadcamnode = Camera('filter-quad-cam')
        lens = OrthographicLens()
        lens.setFilmSize(2, 2)
        lens.setFilmOffset(0, 0)
        lens.setNearFar(-1000, 1000)
        quadcamnode.setLens(lens)
        quadcam = quad.attachNewNode(quadcamnode)
        dr = buffer.makeDisplayRegion((0, 1, 0, 1))
        dr.disableClears()
        dr.setCamera(quadcam)
        dr.setActive(True)
        dr.setScissorEnabled(False)
        buffer.setClearColor((0, 0, 0, 1))
        buffer.setClearColorActive(True)
        self.buffers.append(buffer)
        self.sizes.append((mul, div, align))
        return quad

    def createBuffer(self, name, xsize, ysize, texgroup, depthbits=True, fbprops=None):
        if False:
            for i in range(10):
                print('nop')
        ' Low-level buffer creation.  Not intended for public use. '
        winprops = WindowProperties()
        winprops.setSize(xsize, ysize)
        props = FrameBufferProperties(FrameBufferProperties.getDefault())
        props.setBackBuffers(0)
        props.setRgbColor(1)
        if depthbits is True:
            if props.getDepthBits() == 0:
                props.setDepthBits(1)
        else:
            props.setDepthBits(depthbits)
        props.setStereo(self.win.isStereo())
        if fbprops is not None:
            props.addProperties(fbprops)
        (depthtex, colortex, auxtex0, auxtex1) = texgroup
        if auxtex0 is not None:
            props.setAuxRgba(1)
        if auxtex1 is not None:
            props.setAuxRgba(2)
        buffer = self.engine.makeOutput(self.win.getPipe(), name, -1, props, winprops, GraphicsPipe.BFRefuseWindow | GraphicsPipe.BFResizeable, self.win.getGsg(), self.win)
        if buffer is None:
            return buffer
        if depthtex:
            buffer.addRenderTexture(depthtex, GraphicsOutput.RTMBindOrCopy, GraphicsOutput.RTPDepth)
        if colortex:
            buffer.addRenderTexture(colortex, GraphicsOutput.RTMBindOrCopy, GraphicsOutput.RTPColor)
        if auxtex0:
            buffer.addRenderTexture(auxtex0, GraphicsOutput.RTMBindOrCopy, GraphicsOutput.RTPAuxRgba0)
        if auxtex1:
            buffer.addRenderTexture(auxtex1, GraphicsOutput.RTMBindOrCopy, GraphicsOutput.RTPAuxRgba1)
        buffer.setSort(self.nextsort)
        buffer.disableClears()
        self.nextsort += 1
        return buffer

    def windowEvent(self, win):
        if False:
            print('Hello World!')
        ' When the window changes size, automatically resize all buffers '
        self.resizeBuffers()

    def resizeBuffers(self):
        if False:
            for i in range(10):
                print('nop')
        ' Resize all buffers to match the size of the window. '
        for (i, buffer) in enumerate(self.buffers):
            (mul, div, align) = self.sizes[i]
            (xsize, ysize) = self.getScaledSize(mul, div, align)
            buffer.setSize(xsize, ysize)

    def cleanup(self):
        if False:
            while True:
                i = 10
        ' Restore everything to its original state, deleting any\n        new buffers in the process. '
        for buffer in self.buffers:
            buffer.clearRenderTextures()
            self.engine.removeWindow(buffer)
        self.buffers = []
        self.sizes = []
        self.setClears(self.win, self.wclears)
        self.setClears(self.region, self.rclears)
        self.camstate = self.caminit
        self.camera.node().setInitialState(self.caminit)
        self.region.setCamera(self.camera)
        if hasattr(self.region, 'clearCullResult'):
            self.region.clearCullResult()
        self.nextsort = self.win.getSort() - 9
        self.basex = 0
        self.basey = 0
    is_fullscreen = isFullscreen
    resize_buffers = resizeBuffers
    set_stacked_clears = setStackedClears
    render_scene_into = renderSceneInto
    get_scaled_size = getScaledSize
    render_quad_into = renderQuadInto
    get_clears = getClears
    set_clears = setClears
    create_buffer = createBuffer
    window_event = windowEvent