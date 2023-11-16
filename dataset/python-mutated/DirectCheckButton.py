"""A DirectCheckButton is a type of button that toggles between two states
when clicked.  It also has a separate indicator that can be modified
separately.

See the :ref:`directcheckbutton` page in the programming manual for a more
in-depth explanation and an example of how to use this class.
"""
__all__ = ['DirectCheckButton']
from panda3d.core import PGFrameStyle, VBase4
from .DirectButton import DirectButton
from .DirectLabel import DirectLabel

class DirectCheckButton(DirectButton):
    """
    DirectCheckButton(parent) - Create a DirectGuiWidget which responds
    to mouse clicks by setting a state of on or off and execute a callback
    function (passing that state through) if defined
    """

    def __init__(self, parent=None, **kw):
        if False:
            print('Hello World!')
        self.colors = None
        optiondefs = (('indicatorValue', 0, self.setIndicatorValue), ('boxBorder', 0, None), ('boxPlacement', 'left', None), ('boxImage', None, None), ('boxImageScale', 1, None), ('boxImageColor', None, None), ('boxRelief', 'sunken', None))
        self.defineoptions(kw, optiondefs)
        DirectButton.__init__(self, parent)
        self.indicator = self.createcomponent('indicator', (), None, DirectLabel, (self,), numStates=2, image=self['boxImage'], image_scale=self['boxImageScale'], image_color=self['boxImageColor'], state='disabled', text=('X', 'X'), relief=self['boxRelief'])
        self.initialiseoptions(DirectCheckButton)
        if self['boxImage'] is None:
            self.indicator['text'] = (' ', '*')
            self.indicator['text_pos'] = (0, -0.2)
        else:
            self.indicator['text'] = (' ', ' ')
        if self['boxImageColor'] is not None and self['boxImage'] is not None:
            self.colors = [VBase4(0, 0, 0, 0), self['boxImageColor']]
            self.component('indicator')['image_color'] = VBase4(0, 0, 0, 0)

    def resetFrameSize(self):
        if False:
            print('Hello World!')
        self.setFrameSize(fClearFrame=1)

    def setFrameSize(self, fClearFrame=0):
        if False:
            print('Hello World!')
        if self['frameSize']:
            self.bounds = self['frameSize']
            frameType = self.frameStyle[0].getType()
            ibw = self.indicator['borderWidth']
        else:
            frameType = self.frameStyle[0].getType()
            if fClearFrame and frameType != PGFrameStyle.TNone:
                self.frameStyle[0].setType(PGFrameStyle.TNone)
                self.guiItem.setFrameStyle(0, self.frameStyle[0])
                self.guiItem.getStateDef(0)
            self.getBounds()
            if frameType != PGFrameStyle.TNone:
                self.frameStyle[0].setType(frameType)
                self.guiItem.setFrameStyle(0, self.frameStyle[0])
            ibw = self.indicator['borderWidth']
            indicatorWidth = self.indicator.getWidth() + 2 * ibw[0]
            indicatorHeight = self.indicator.getHeight() + 2 * ibw[1]
            diff = indicatorHeight + 2 * self['boxBorder'] - (self.bounds[3] - self.bounds[2])
            if diff > 0:
                if self['boxPlacement'] == 'left':
                    self.bounds[0] += -(indicatorWidth + 2 * self['boxBorder'])
                    self.bounds[3] += diff / 2
                    self.bounds[2] -= diff / 2
                elif self['boxPlacement'] == 'below':
                    self.bounds[2] += -(indicatorHeight + 2 * self['boxBorder'])
                elif self['boxPlacement'] == 'right':
                    self.bounds[1] += indicatorWidth + 2 * self['boxBorder']
                    self.bounds[3] += diff / 2
                    self.bounds[2] -= diff / 2
                else:
                    self.bounds[3] += indicatorHeight + 2 * self['boxBorder']
            elif self['boxPlacement'] == 'left':
                self.bounds[0] += -(indicatorWidth + 2 * self['boxBorder'])
            elif self['boxPlacement'] == 'below':
                self.bounds[2] += -(indicatorHeight + 2 * self['boxBorder'])
            elif self['boxPlacement'] == 'right':
                self.bounds[1] += indicatorWidth + 2 * self['boxBorder']
            else:
                self.bounds[3] += indicatorHeight + 2 * self['boxBorder']
        if frameType != PGFrameStyle.TNone and frameType != PGFrameStyle.TFlat:
            bw = self['borderWidth']
        else:
            bw = (0, 0)
        self.guiItem.setFrame(self.bounds[0] - bw[0], self.bounds[1] + bw[0], self.bounds[2] - bw[1], self.bounds[3] + bw[1])
        if not self.indicator['pos']:
            bbounds = self.bounds
            lbounds = self.indicator.bounds
            newpos = [0, 0, 0]
            if self['boxPlacement'] == 'left':
                newpos[0] += bbounds[0] - lbounds[0] + self['boxBorder'] + ibw[0]
                dropValue = (bbounds[3] - bbounds[2] - lbounds[3] + lbounds[2]) / 2 + self['boxBorder']
                newpos[2] += bbounds[3] - lbounds[3] + self['boxBorder'] - dropValue
            elif self['boxPlacement'] == 'right':
                newpos[0] += bbounds[1] - lbounds[1] - self['boxBorder'] - ibw[0]
                dropValue = (bbounds[3] - bbounds[2] - lbounds[3] + lbounds[2]) / 2 + self['boxBorder']
                newpos[2] += bbounds[3] - lbounds[3] + self['boxBorder'] - dropValue
            elif self['boxPlacement'] == 'above':
                newpos[2] += bbounds[3] - lbounds[3] - self['boxBorder'] - ibw[1]
            else:
                newpos[2] += bbounds[2] - lbounds[2] + self['boxBorder'] + ibw[1]
            self.indicator.setPos(newpos[0], newpos[1], newpos[2])

    def commandFunc(self, event):
        if False:
            print('Hello World!')
        self['indicatorValue'] = 1 - self['indicatorValue']
        if self.colors is not None:
            self.component('indicator')['image_color'] = self.colors[self['indicatorValue']]
        if self['command']:
            self['command'](*[self['indicatorValue']] + self['extraArgs'])

    def setIndicatorValue(self):
        if False:
            while True:
                i = 10
        self.component('indicator').guiItem.setState(self['indicatorValue'])
        if self.colors is not None:
            self.component('indicator')['image_color'] = self.colors[self['indicatorValue']]