"""Contains the DirectWaitBar class, a progress bar widget.

See the :ref:`directwaitbar` page in the programming manual for a more
in-depth explanation and an example of how to use this class.
"""
__all__ = ['DirectWaitBar']
from panda3d.core import PGFrameStyle, PGWaitBar
from . import DirectGuiGlobals as DGG
from .DirectFrame import DirectFrame

class DirectWaitBar(DirectFrame):
    """ DirectWaitBar - A DirectWidget that shows progress completed
    towards a task.  """

    def __init__(self, parent=None, **kw):
        if False:
            for i in range(10):
                print('nop')
        optiondefs = (('pgFunc', PGWaitBar, None), ('frameSize', (-1, 1, -0.08, 0.08), None), ('borderWidth', (0, 0), None), ('range', 100, self.setRange), ('value', 0, self.setValue), ('barBorderWidth', (0, 0), self.setBarBorderWidth), ('barColor', (1, 0, 0, 1), self.setBarColor), ('barTexture', None, self.setBarTexture), ('barRelief', DGG.FLAT, self.setBarRelief), ('sortOrder', DGG.NO_FADE_SORT_INDEX, None))
        if 'text' in kw:
            textoptiondefs = (('text_pos', (0, -0.025), None), ('text_scale', 0.1, None))
        else:
            textoptiondefs = ()
        self.defineoptions(kw, optiondefs + textoptiondefs)
        DirectFrame.__init__(self, parent)
        self.barStyle = PGFrameStyle()
        self.initialiseoptions(DirectWaitBar)
        self.updateBarStyle()

    def destroy(self):
        if False:
            for i in range(10):
                print('nop')
        self.barStyle = None
        DirectFrame.destroy(self)

    def setRange(self):
        if False:
            i = 10
            return i + 15
        "Updates the bar range which you can set using bar['range'].\n        This is the value at which the WaitBar indicates 100%."
        self.guiItem.setRange(self['range'])

    def setValue(self):
        if False:
            i = 10
            return i + 15
        "Updates the bar value which you can set using bar['value'].\n        The value should range between 0 and bar['range']."
        self.guiItem.setValue(self['value'])

    def getPercent(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the percentage complete.'
        return self.guiItem.getPercent()

    def updateBarStyle(self):
        if False:
            return 10
        if not self.fInit:
            self.guiItem.setBarStyle(self.barStyle)

    def setBarRelief(self):
        if False:
            return 10
        "Updates the bar relief, which you can set using bar['barRelief']."
        self.barStyle.setType(self['barRelief'])
        self.updateBarStyle()

    def setBarBorderWidth(self):
        if False:
            i = 10
            return i + 15
        "Updates the bar's border width, which you can set using bar['barBorderWidth']."
        self.barStyle.setWidth(*self['barBorderWidth'])
        self.updateBarStyle()

    def setBarColor(self):
        if False:
            while True:
                i = 10
        "Updates the bar color, which you can set using bar['barColor']."
        color = self['barColor']
        self.barStyle.setColor(color[0], color[1], color[2], color[3])
        self.updateBarStyle()

    def setBarTexture(self):
        if False:
            return 10
        "Updates the bar texture, which you can set using bar['barTexture']."
        texture = self['barTexture']
        if isinstance(texture, str):
            texture = base.loader.loadTexture(texture)
        if texture:
            self.barStyle.setTexture(texture)
        else:
            self.barStyle.clearTexture()
        self.updateBarStyle()

    def update(self, value):
        if False:
            for i in range(10):
                print('nop')
        'Updates the bar with the given value and renders a frame.'
        self['value'] = value
        base.graphicsEngine.renderFrame()

    def finish(self, N=10):
        if False:
            for i in range(10):
                print('nop')
        'Fill the bar in N frames. This call is blocking.'
        remaining = self['range'] - self['value']
        if remaining:
            step = max(1, int(remaining / N))
            count = self['value']
            while count != self['range']:
                count += step
                if count > self['range']:
                    count = self['range']
                self.update(count)