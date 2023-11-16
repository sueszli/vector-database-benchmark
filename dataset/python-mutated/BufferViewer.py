"""Contains the BufferViewer class, which is used as a debugging aid
when debugging render-to-texture effects.  It shows different views at
the bottom of the screen showing the various render targets.

When using ShowBase, the normal way to enable the BufferViewer is using the
following code::

    base.bufferViewer.toggleEnable()

Or, you can enable the following variable in your Config.prc::

    show-buffers true
"""
__all__ = ['BufferViewer']
from panda3d.core import CardMaker, ConfigVariableBool, ConfigVariableDouble, ConfigVariableString, Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat, GeomVertexWriter, GraphicsEngine, GraphicsOutput, NodePath, Point3, SamplerState, Texture, Vec3, Vec3F
from direct.task import Task
from direct.task.TaskManagerGlobal import taskMgr
from direct.directnotify.DirectNotifyGlobal import directNotify
from direct.showbase.DirectObject import DirectObject
import math

class BufferViewer(DirectObject):
    notify = directNotify.newCategory('BufferViewer')

    def __init__(self, win, parent):
        if False:
            print('Hello World!')
        'Access: private.  Constructor.'
        self.enabled = 0
        size = ConfigVariableDouble('buffer-viewer-size', '0 0')
        self.sizex = size[0]
        self.sizey = size[1]
        self.position = ConfigVariableString('buffer-viewer-position', 'lrcorner').getValue()
        self.layout = ConfigVariableString('buffer-viewer-layout', 'hline').getValue()
        self.include = 'all'
        self.exclude = 'none'
        self.cullbin = 'fixed'
        self.cullsort = 10000
        self.win = win
        self.engine = GraphicsEngine.getGlobalPtr()
        self.renderParent = parent
        self.cards = []
        self.cardindex = 0
        self.cardmaker = CardMaker('cubemaker')
        self.cardmaker.setFrame(-1, 1, -1, 1)
        self.task = 0
        self.dirty = 1
        self.accept('render-texture-targets-changed', self.refreshReadout)
        if ConfigVariableBool('show-buffers', 0):
            self.enable(1)

    def refreshReadout(self):
        if False:
            return 10
        'Force the readout to be refreshed.  This is usually invoked\n        by GraphicsOutput::add_render_texture (via an event handler).\n        However, it is also possible to invoke it manually.  Currently,\n        the only time I know of that this is necessary is after a\n        window resize (and I ought to fix that).'
        self.dirty = 1
        self.enable(self.enabled)

    def isValidTextureSet(self, x):
        if False:
            for i in range(10):
                print('nop')
        "Access: private. Returns true if the parameter is a\n        list of GraphicsOutput and Texture, or the keyword 'all'."
        if isinstance(x, list):
            for elt in x:
                if not self.isValidTextureSet(elt):
                    return 0
        else:
            return x == 'all' or isinstance(x, Texture) or isinstance(x, GraphicsOutput)

    def isEnabled(self):
        if False:
            i = 10
            return i + 15
        'Returns true if the buffer viewer is currently enabled.'
        return self.enabled

    def enable(self, x):
        if False:
            for i in range(10):
                print('nop')
        "Turn the buffer viewer on or off.  The initial state of the\n        buffer viewer depends on the Config variable 'show-buffers'."
        if x != 0 and x != 1:
            BufferViewer.notify.error('invalid parameter to BufferViewer.enable')
            return
        self.enabled = x
        self.dirty = 1
        if x and self.task == 0:
            self.task = taskMgr.add(self.maintainReadout, 'buffer-viewer-maintain-readout', priority=1)

    def toggleEnable(self):
        if False:
            while True:
                i = 10
        "Toggle the buffer viewer on or off.  The initial state of the\n        enable flag depends on the Config variable 'show-buffers'."
        self.enable(1 - self.enabled)

    def setCardSize(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        'Set the size of each card.  The units are relative to\n        render2d (ie, 1x1 card is not square).  If one of the\n        dimensions is zero, then the viewer will choose a value\n        for that dimension so as to ensure that the aspect ratio\n        of the card matches the aspect ratio of the source-window.\n        If both dimensions are zero, the viewer uses a heuristic\n        to choose a reasonable size for the card.  The initial\n        value is (0, 0).'
        if x < 0 or y < 0:
            BufferViewer.notify.error('invalid parameter to BufferViewer.setCardSize')
            return
        self.sizex = x
        self.sizey = y
        self.dirty = 1

    def setPosition(self, pos):
        if False:
            for i in range(10):
                print('nop')
        "Set the position of the cards.  The valid values are:\n\n        - *llcorner* - put them in the lower-left  corner of the window\n        - *lrcorner* - put them in the lower-right corner of the window\n        - *ulcorner* - put them in the upper-left  corner of the window\n        - *urcorner* - put them in the upper-right corner of the window\n        - *window* - put them in a separate window\n\n        The initial value is 'lrcorner'."
        valid = ['llcorner', 'lrcorner', 'ulcorner', 'urcorner', 'window']
        if valid.count(pos) == 0:
            BufferViewer.notify.error('invalid parameter to BufferViewer.setPosition')
            BufferViewer.notify.error('valid parameters are: llcorner, lrcorner, ulcorner, urcorner, window')
            return
        if pos == 'window':
            BufferViewer.notify.error('BufferViewer.setPosition - "window" mode not implemented yet.')
            return
        self.position = pos
        self.dirty = 1

    def setLayout(self, lay):
        if False:
            print('Hello World!')
        "Set the layout of the cards.  The valid values are:\n\n        - *vline* - display them in a vertical line\n        - *hline* - display them in a horizontal line\n        - *vgrid* - display them in a vertical grid\n        - *hgrid* - display them in a horizontal grid\n        - *cycle* - display one card at a time, using selectCard/advanceCard\n\n        The default value is 'hline'."
        valid = ['vline', 'hline', 'vgrid', 'hgrid', 'cycle']
        if valid.count(lay) == 0:
            BufferViewer.notify.error('invalid parameter to BufferViewer.setLayout')
            BufferViewer.notify.error('valid parameters are: vline, hline, vgrid, hgrid, cycle')
            return
        self.layout = lay
        self.dirty = 1

    def selectCard(self, i):
        if False:
            while True:
                i = 10
        "Only useful when using setLayout('cycle').  Sets the index\n        that selects which card to display.  The index is taken modulo\n        the actual number of cards."
        self.cardindex = i
        self.dirty = 1

    def advanceCard(self):
        if False:
            for i in range(10):
                print('nop')
        "Only useful when using setLayout('cycle').  Increments the index\n        that selects which card to display.  The index is taken modulo\n        the actual number of cards."
        self.cardindex += 1
        self.dirty = 1

    def setInclude(self, x):
        if False:
            return 10
        "Set the include-set for the buffer viewer.  The include-set\n        specifies which of the render-to-texture targets to display.\n        Valid inputs are the string 'all' (display every render-to-texture\n        target), or a list of GraphicsOutputs or Textures.  The initial\n        value is 'all'."
        if not self.isValidTextureSet(x):
            BufferViewer.notify.error('setInclude: must be list of textures and buffers, or "all"')
            return
        self.include = x
        self.dirty = 1

    def setExclude(self, x):
        if False:
            while True:
                i = 10
        'Set the exclude-set for the buffer viewer.  The exclude-set\n        should be a list of GraphicsOutputs and Textures to ignore.\n        The exclude-set is subtracted from the include-set (so the excludes\n        effectively override the includes.)  The initial value is the\n        empty list.'
        if not self.isValidTextureSet(x):
            BufferViewer.notify.error('setExclude: must be list of textures and buffers')
            return
        self.exclude = x
        self.dirty = 1

    def setSort(self, bin, sort):
        if False:
            while True:
                i = 10
        "Set the cull-bin and sort-order for the output cards.  The\n        default value is 'fixed', 10000."
        self.cullbin = bin
        self.cullsort = sort
        self.dirty = 1

    def setRenderParent(self, renderParent):
        if False:
            return 10
        'Set the scene graph root to which the output cards should\n        be parented.  The default is render2d. '
        self.renderParent = renderParent
        self.dirty = 1

    def analyzeTextureSet(self, x, set):
        if False:
            i = 10
            return i + 15
        'Access: private.  Converts a list of GraphicsObject,\n        GraphicsEngine, and Texture into a table of Textures.'
        if isinstance(x, list):
            for elt in x:
                self.analyzeTextureSet(elt, set)
        elif isinstance(x, Texture):
            set[x] = 1
        elif isinstance(x, GraphicsOutput):
            for itex in range(x.countTextures()):
                tex = x.getTexture(itex)
                set[tex] = 1
        elif isinstance(x, GraphicsEngine):
            for iwin in range(x.getNumWindows()):
                win = x.getWindow(iwin)
                self.analyzeTextureSet(win, set)
        elif x == 'all':
            self.analyzeTextureSet(self.engine, set)
        else:
            return

    def makeFrame(self, sizex, sizey):
        if False:
            for i in range(10):
                print('nop')
        "Access: private.  Each texture card is displayed with\n        a two-pixel wide frame (a ring of black and a ring of white).\n        This routine builds the frame geometry.  It is necessary to\n        be precise so that the frame exactly aligns to pixel\n        boundaries, and so that it doesn't overlap the card at all."
        format = GeomVertexFormat.getV3c()
        vdata = GeomVertexData('card-frame', format, Geom.UHDynamic)
        vwriter = GeomVertexWriter(vdata, 'vertex')
        cwriter = GeomVertexWriter(vdata, 'color')
        ringoffset = [0, 1, 1, 2]
        ringbright = [0, 0, 1, 1]
        for ring in range(4):
            offsetx = ringoffset[ring] * 2.0 / float(sizex)
            offsety = ringoffset[ring] * 2.0 / float(sizey)
            bright = ringbright[ring]
            vwriter.addData3f(Vec3F.rfu(-1 - offsetx, 0, -1 - offsety))
            vwriter.addData3f(Vec3F.rfu(1 + offsetx, 0, -1 - offsety))
            vwriter.addData3f(Vec3F.rfu(1 + offsetx, 0, 1 + offsety))
            vwriter.addData3f(Vec3F.rfu(-1 - offsetx, 0, 1 + offsety))
            cwriter.addData3f(bright, bright, bright)
            cwriter.addData3f(bright, bright, bright)
            cwriter.addData3f(bright, bright, bright)
            cwriter.addData3f(bright, bright, bright)
        triangles = GeomTriangles(Geom.UHStatic)
        for i in range(2):
            delta = i * 8
            triangles.addVertices(0 + delta, 4 + delta, 1 + delta)
            triangles.addVertices(1 + delta, 4 + delta, 5 + delta)
            triangles.addVertices(1 + delta, 5 + delta, 2 + delta)
            triangles.addVertices(2 + delta, 5 + delta, 6 + delta)
            triangles.addVertices(2 + delta, 6 + delta, 3 + delta)
            triangles.addVertices(3 + delta, 6 + delta, 7 + delta)
            triangles.addVertices(3 + delta, 7 + delta, 0 + delta)
            triangles.addVertices(0 + delta, 7 + delta, 4 + delta)
        triangles.closePrimitive()
        geom = Geom(vdata)
        geom.addPrimitive(triangles)
        geomnode = GeomNode('card-frame')
        geomnode.addGeom(geom)
        return NodePath(geomnode)

    def maintainReadout(self, task):
        if False:
            for i in range(10):
                print('nop')
        'Access: private.  Whenever necessary, rebuilds the entire\n        display from scratch.  This is only done when the configuration\n        parameters have changed.'
        if not self.dirty:
            return Task.cont
        self.dirty = 0
        for card in self.cards:
            card.removeNode()
        self.cards = []
        if not self.enabled:
            self.task = 0
            return Task.done
        exclude = {}
        include = {}
        self.analyzeTextureSet(self.exclude, exclude)
        self.analyzeTextureSet(self.include, include)
        sampler = SamplerState()
        sampler.setWrapU(SamplerState.WM_clamp)
        sampler.setWrapV(SamplerState.WM_clamp)
        sampler.setWrapW(SamplerState.WM_clamp)
        sampler.setMinfilter(SamplerState.FT_linear)
        sampler.setMagfilter(SamplerState.FT_nearest)
        cards = []
        wins = []
        for iwin in range(self.engine.getNumWindows()):
            win = self.engine.getWindow(iwin)
            for itex in range(win.countTextures()):
                tex = win.getTexture(itex)
                if tex in include and tex not in exclude:
                    if tex.getTextureType() == Texture.TTCubeMap:
                        for face in range(6):
                            self.cardmaker.setUvRangeCube(face)
                            card = NodePath(self.cardmaker.generate())
                            card.setTexture(tex, sampler)
                            cards.append(card)
                    elif tex.getTextureType() == Texture.TT2dTextureArray:
                        for layer in range(tex.getZSize()):
                            self.cardmaker.setUvRange((0, 1, 1, 0), (0, 0, 1, 1), (layer, layer, layer, layer))
                            card = NodePath(self.cardmaker.generate())
                            card.setShaderAuto()
                            card.setTexture(tex, sampler)
                            cards.append(card)
                    else:
                        card = win.getTextureCard()
                        card.setTexture(tex, sampler)
                        cards.append(card)
                    wins.append(win)
                    exclude[tex] = 1
        self.cards = cards
        if len(cards) == 0:
            self.task = 0
            return Task.done
        ncards = len(cards)
        if self.layout == 'hline':
            rows = 1
            cols = ncards
        elif self.layout == 'vline':
            rows = ncards
            cols = 1
        elif self.layout == 'hgrid':
            rows = int(math.sqrt(ncards))
            cols = rows
            if rows * cols < ncards:
                cols += 1
            if rows * cols < ncards:
                rows += 1
        elif self.layout == 'vgrid':
            rows = int(math.sqrt(ncards))
            cols = rows
            if rows * cols < ncards:
                rows += 1
            if rows * cols < ncards:
                cols += 1
        elif self.layout == 'cycle':
            rows = 1
            cols = 1
        else:
            BufferViewer.notify.error('shouldnt ever get here in BufferViewer.maintainReadout')
        aspectx = wins[0].getXSize()
        aspecty = wins[0].getYSize()
        for win in wins:
            if win.getXSize() * aspecty != win.getYSize() * aspectx:
                aspectx = 1
                aspecty = 1
        bordersize = 4.0
        if float(self.sizex) == 0.0 and float(self.sizey) == 0.0:
            sizey = int(0.4266666667 * self.win.getYSize())
            sizex = sizey * aspectx // aspecty
            v_sizey = (self.win.getYSize() - (rows - 1) - rows * 2) // rows
            v_sizex = v_sizey * aspectx // aspecty
            if v_sizey < sizey or v_sizex < sizex:
                sizey = v_sizey
                sizex = v_sizex
            adjustment = 2
            h_sizex = float(self.win.getXSize() - adjustment) / float(cols)
            h_sizex -= bordersize
            if h_sizex < 1.0:
                h_sizex = 1.0
            h_sizey = h_sizex * aspecty // aspectx
            if h_sizey < sizey or h_sizex < sizex:
                sizey = h_sizey
                sizex = h_sizex
        else:
            sizex = int(self.sizex * 0.5 * self.win.getXSize())
            sizey = int(self.sizey * 0.5 * self.win.getYSize())
            if sizex == 0:
                sizex = sizey * aspectx // aspecty
            if sizey == 0:
                sizey = sizex * aspecty // aspectx
        fsizex = 2.0 * sizex / float(self.win.getXSize())
        fsizey = 2.0 * sizey / float(self.win.getYSize())
        fpixelx = 2.0 / float(self.win.getXSize())
        fpixely = 2.0 / float(self.win.getYSize())
        if self.position == 'llcorner':
            dirx = -1.0
            diry = -1.0
        elif self.position == 'lrcorner':
            dirx = 1.0
            diry = -1.0
        elif self.position == 'ulcorner':
            dirx = -1.0
            diry = 1.0
        elif self.position == 'urcorner':
            dirx = 1.0
            diry = 1.0
        else:
            BufferViewer.notify.error('window mode not implemented yet')
        frame = self.makeFrame(sizex, sizey)
        for r in range(rows):
            for c in range(cols):
                index = c + r * cols
                if index < ncards:
                    index = (index + self.cardindex) % len(cards)
                    posx = dirx * (1.0 - (c + 0.5) * (fsizex + fpixelx * bordersize)) - fpixelx * dirx
                    posy = diry * (1.0 - (r + 0.5) * (fsizey + fpixely * bordersize)) - fpixely * diry
                    placer = NodePath('card-structure')
                    placer.setPos(Point3.rfu(posx, 0, posy))
                    placer.setScale(Vec3.rfu(fsizex * 0.5, 1.0, fsizey * 0.5))
                    placer.setBin(self.cullbin, self.cullsort)
                    placer.reparentTo(self.renderParent)
                    frame.instanceTo(placer)
                    cards[index].reparentTo(placer)
                    cards[index] = placer
        return Task.cont
    advance_card = advanceCard
    analyze_texture_set = analyzeTextureSet
    is_enabled = isEnabled
    is_valid_texture_set = isValidTextureSet
    maintain_readout = maintainReadout
    make_frame = makeFrame
    refresh_readout = refreshReadout
    select_card = selectCard
    set_card_size = setCardSize
    set_exclude = setExclude
    set_include = setInclude
    set_layout = setLayout
    set_position = setPosition
    set_render_parent = setRenderParent
    set_sort = setSort
    toggle_enable = toggleEnable