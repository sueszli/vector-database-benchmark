"""This module defines various dialog windows for the DirectGUI system.

See the :ref:`directdialog` page in the programming manual for a more
in-depth explanation and an example of how to use this class.
"""
from __future__ import annotations
__all__ = ['findDialog', 'cleanupDialog', 'DirectDialog', 'OkDialog', 'OkCancelDialog', 'YesNoDialog', 'YesNoCancelDialog', 'RetryCancelDialog']
from panda3d.core import NodePath, Point3, TextNode, VBase3
from direct.showbase import ShowBaseGlobal
from . import DirectGuiGlobals as DGG
from .DirectFrame import DirectFrame
from .DirectButton import DirectButton

def findDialog(uniqueName):
    if False:
        print('Hello World!')
    '\n    Returns the panel whose uniqueName is given.  This is mainly\n    useful for debugging, to get a pointer to the current onscreen\n    panel of a particular type.\n    '
    if uniqueName in DirectDialog.AllDialogs:
        return DirectDialog.AllDialogs[uniqueName]
    return None

def cleanupDialog(uniqueName):
    if False:
        return 10
    'cleanupPanel(string uniqueName)\n\n    Cleans up (removes) the panel with the given uniqueName.  This\n    may be useful when some panels know about each other and know\n    that opening panel A should automatically close panel B, for\n    instance.\n    '
    if uniqueName in DirectDialog.AllDialogs:
        DirectDialog.AllDialogs[uniqueName].cleanup()

class DirectDialog(DirectFrame):
    AllDialogs: dict[str, DirectDialog] = {}
    PanelIndex = 0

    def __init__(self, parent=None, **kw):
        if False:
            print('Hello World!')
        'Creates a popup dialog to alert and/or interact with user.\n        Some of the main keywords that can be used to customize the dialog:\n\n        Parameters:\n            text (str): Text message/query displayed to user\n            geom: Geometry to be displayed in dialog\n            buttonTextList: List of text to show on each button\n            buttonGeomList: List of geometry to show on each button\n            buttonImageList: List of images to show on each button\n            buttonValueList: List of values sent to dialog command for\n                each button.  If value is [] then the ordinal rank of\n                the button is used as its value.\n            buttonHotKeyList: List of hotkeys to bind to each button.\n                Typing the hotkey is equivalent to pressing the\n                corresponding button.\n            suppressKeys: Set to true if you wish to suppress keys\n                (i.e. Dialog eats key event), false if you wish Dialog\n                to pass along key event.\n            buttonSize: 4-tuple used to specify custom size for each\n                button (to make bigger then geom/text for example)\n            pad: Space between border and interior graphics\n            topPad: Extra space added above text/geom/image\n            midPad: Extra space added between text/buttons\n            sidePad: Extra space added to either side of text/buttons\n            buttonPadSF: Scale factor used to expand/contract button\n                horizontal spacing\n            command: Callback command used when a button is pressed.\n                Value supplied to command depends on values in\n                buttonValueList.\n\n        Note:\n            The number of buttons on the dialog depends on the maximum\n            length of any button[Text|Geom|Image|Value]List specified.\n            Values of None are substituted for lists that are shorter\n            than the max length\n         '
        optiondefs = (('dialogName', 'DirectDialog_' + repr(DirectDialog.PanelIndex), DGG.INITOPT), ('pos', (0, 0.1, 0), None), ('pad', (0.1, 0.1), None), ('text', '', None), ('text_align', TextNode.ALeft, None), ('text_scale', 0.06, None), ('image', DGG.getDefaultDialogGeom(), None), ('relief', DGG.getDefaultDialogRelief(), None), ('borderWidth', (0.01, 0.01), None), ('buttonTextList', [], DGG.INITOPT), ('buttonGeomList', [], DGG.INITOPT), ('buttonImageList', [], DGG.INITOPT), ('buttonValueList', [], DGG.INITOPT), ('buttonHotKeyList', [], DGG.INITOPT), ('button_borderWidth', (0.01, 0.01), None), ('button_pad', (0.01, 0.01), None), ('button_relief', DGG.RAISED, None), ('button_text_scale', 0.06, None), ('buttonSize', None, DGG.INITOPT), ('topPad', 0.06, DGG.INITOPT), ('midPad', 0.12, DGG.INITOPT), ('sidePad', 0.0, DGG.INITOPT), ('buttonPadSF', 1.1, DGG.INITOPT), ('fadeScreen', 0, None), ('command', None, None), ('extraArgs', [], None), ('sortOrder', DGG.NO_FADE_SORT_INDEX, None))
        self.defineoptions(kw, optiondefs, dynamicGroups=('button',))
        DirectFrame.__init__(self, parent)
        cleanupDialog(self['dialogName'])
        DirectDialog.AllDialogs[self['dialogName']] = self
        DirectDialog.PanelIndex += 1
        self.numButtons = max(len(self['buttonTextList']), len(self['buttonGeomList']), len(self['buttonImageList']), len(self['buttonValueList']))
        self.buttonList = []
        index = 0
        for i in range(self.numButtons):
            name = 'Button' + repr(i)
            try:
                text = self['buttonTextList'][i]
            except IndexError:
                text = None
            try:
                geom = self['buttonGeomList'][i]
            except IndexError:
                geom = None
            try:
                image = self['buttonImageList'][i]
            except IndexError:
                image = None
            try:
                value = self['buttonValueList'][i]
            except IndexError:
                value = i
                self['buttonValueList'].append(i)
            try:
                hotKey = self['buttonHotKeyList'][i]
            except IndexError:
                hotKey = None
            button = self.createcomponent(name, (), 'button', DirectButton, (self,), text=text, geom=geom, image=image, suppressKeys=self['suppressKeys'], frameSize=self['buttonSize'], command=lambda s=self, v=value: s.buttonCommand(v))
            self.buttonList.append(button)
        self.postInitialiseFuncList.append(self.configureDialog)
        self.initialiseoptions(DirectDialog)

    def configureDialog(self):
        if False:
            while True:
                i = 10
        bindList = zip(self.buttonList, self['buttonHotKeyList'], self['buttonValueList'])
        for (button, hotKey, value) in bindList:
            if isinstance(hotKey, (list, tuple)):
                for key in hotKey:
                    button.bind('press-' + key + '-', self.buttonCommand, extraArgs=[value])
                    self.bind('press-' + key + '-', self.buttonCommand, extraArgs=[value])
            else:
                button.bind('press-' + hotKey + '-', self.buttonCommand, extraArgs=[value])
                self.bind('press-' + hotKey + '-', self.buttonCommand, extraArgs=[value])
        pad = self['pad']
        if self.hascomponent('image0'):
            image = self.component('image0')
        else:
            image = None
        if image:
            image.reparentTo(ShowBaseGlobal.hidden)
        bounds = self.stateNodePath[0].getTightBounds()
        if image:
            image.reparentTo(self.stateNodePath[0])
        if bounds is None:
            l = 0
            r = 0
            b = 0
            t = 0
        else:
            l = bounds[0][0]
            r = bounds[1][0]
            b = bounds[0][2]
            t = bounds[1][2]
        xOffset = -(l + r) * 0.5
        zOffset = -(b + t) * 0.5
        l += xOffset
        r += xOffset
        b += zOffset
        t += zOffset
        if self['text']:
            self['text_pos'] = (self['text_pos'][0] + xOffset, self['text_pos'][1] + zOffset)
        if self['geom']:
            self['geom_pos'] = Point3(self['geom_pos'][0] + xOffset, self['geom_pos'][1], self['geom_pos'][2] + zOffset)
        if self.numButtons != 0:
            bpad = self['button_pad']
            if self['buttonSize']:
                buttonSize = self['buttonSize']
                bl = buttonSize[0]
                br = buttonSize[1]
                bb = buttonSize[2]
                bt = buttonSize[3]
            else:
                bl = br = bb = bt = 0
                for button in self.buttonList:
                    bounds = button.stateNodePath[0].getTightBounds()
                    if bounds is None:
                        bl = 0
                        br = 0
                        bb = 0
                        bt = 0
                    else:
                        bl = min(bl, bounds[0][0])
                        br = max(br, bounds[1][0])
                        bb = min(bb, bounds[0][2])
                        bt = max(bt, bounds[1][2])
                bl -= bpad[0]
                br += bpad[0]
                bb -= bpad[1]
                bt += bpad[1]
                for button in self.buttonList:
                    button['frameSize'] = (bl, br, bb, bt)
            scale = self['button_scale']
            if isinstance(scale, (VBase3, list, tuple)):
                sx = scale[0]
                sz = scale[2]
            elif isinstance(scale, (int, float)):
                sx = sz = scale
            else:
                sx = sz = 1
            bl *= sx
            br *= sx
            bb *= sz
            bt *= sz
            bHeight = bt - bb
            bWidth = br - bl
            bSpacing = self['buttonPadSF'] * bWidth
            bPos = -bSpacing * (self.numButtons - 1) * 0.5
            index = 0
            for button in self.buttonList:
                button.setPos(bPos + index * bSpacing, 0, b - self['midPad'] - bpad[1] - bt)
                index += 1
            bMax = bPos + bSpacing * (self.numButtons - 1)
        else:
            bpad = 0
            bl = br = bb = bt = 0
            bPos = 0
            bMax = 0
            bpad = (0, 0)
            bHeight = bWidth = 0
        l = min(bPos + bl, l) - pad[0]
        r = max(bMax + br, r) + pad[0]
        sidePad = self['sidePad']
        l -= sidePad
        r += sidePad
        b = min(b - self['midPad'] - bpad[1] - bHeight - bpad[1], b) - pad[1]
        t = t + self['topPad'] + pad[1]
        if self['frameSize'] is None:
            self['frameSize'] = (l, r, b, t)
        self['image_scale'] = (r - l, 1, t - b)
        self['image_pos'] = ((l + r) * 0.5, 0.0, (b + t) * 0.5)
        self.resetFrameSize()

    def show(self):
        if False:
            return 10
        if self['fadeScreen']:
            base.transitions.fadeScreen(self['fadeScreen'])
            self.setBin('gui-popup', 0)
        NodePath.show(self)

    def hide(self):
        if False:
            print('Hello World!')
        if self['fadeScreen']:
            base.transitions.noTransitions()
        NodePath.hide(self)

    def buttonCommand(self, value, event=None):
        if False:
            for i in range(10):
                print('nop')
        if self['command']:
            self['command'](value, *self['extraArgs'])

    def setMessage(self, message):
        if False:
            print('Hello World!')
        self['text'] = message
        self.configureDialog()

    def cleanup(self):
        if False:
            return 10
        uniqueName = self['dialogName']
        if uniqueName in DirectDialog.AllDialogs:
            del DirectDialog.AllDialogs[uniqueName]
        self.destroy()

    def destroy(self):
        if False:
            while True:
                i = 10
        if self['fadeScreen']:
            base.transitions.noTransitions()
        for button in self.buttonList:
            button.destroy()
        DirectFrame.destroy(self)

class OkDialog(DirectDialog):

    def __init__(self, parent=None, **kw):
        if False:
            for i in range(10):
                print('nop')
        optiondefs = (('buttonTextList', ['OK'], DGG.INITOPT), ('buttonValueList', [DGG.DIALOG_OK], DGG.INITOPT))
        self.defineoptions(kw, optiondefs)
        DirectDialog.__init__(self, parent)
        self.initialiseoptions(OkDialog)

class OkCancelDialog(DirectDialog):

    def __init__(self, parent=None, **kw):
        if False:
            for i in range(10):
                print('nop')
        optiondefs = (('buttonTextList', ['OK', 'Cancel'], DGG.INITOPT), ('buttonValueList', [DGG.DIALOG_OK, DGG.DIALOG_CANCEL], DGG.INITOPT))
        self.defineoptions(kw, optiondefs)
        DirectDialog.__init__(self, parent)
        self.initialiseoptions(OkCancelDialog)

class YesNoDialog(DirectDialog):

    def __init__(self, parent=None, **kw):
        if False:
            return 10
        optiondefs = (('buttonTextList', ['Yes', 'No'], DGG.INITOPT), ('buttonValueList', [DGG.DIALOG_YES, DGG.DIALOG_NO], DGG.INITOPT))
        self.defineoptions(kw, optiondefs)
        DirectDialog.__init__(self, parent)
        self.initialiseoptions(YesNoDialog)

class YesNoCancelDialog(DirectDialog):

    def __init__(self, parent=None, **kw):
        if False:
            return 10
        optiondefs = (('buttonTextList', ['Yes', 'No', 'Cancel'], DGG.INITOPT), ('buttonValueList', [DGG.DIALOG_YES, DGG.DIALOG_NO, DGG.DIALOG_CANCEL], DGG.INITOPT))
        self.defineoptions(kw, optiondefs)
        DirectDialog.__init__(self, parent)
        self.initialiseoptions(YesNoCancelDialog)

class RetryCancelDialog(DirectDialog):

    def __init__(self, parent=None, **kw):
        if False:
            print('Hello World!')
        optiondefs = (('buttonTextList', ['Retry', 'Cancel'], DGG.INITOPT), ('buttonValueList', [DGG.DIALOG_RETRY, DGG.DIALOG_CANCEL], DGG.INITOPT))
        self.defineoptions(kw, optiondefs)
        DirectDialog.__init__(self, parent)
        self.initialiseoptions(RetryCancelDialog)