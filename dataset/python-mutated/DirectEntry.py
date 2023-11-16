"""Contains the DirectEntry class, a type of DirectGUI widget that accepts
text entered using the keyboard.

See the :ref:`directentry` page in the programming manual for a more in-depth
explanation and an example of how to use this class.
"""
__all__ = ['DirectEntry']
from panda3d.core import ConfigVariableBool, PGEntry, Point3, TextNode, Vec3
from direct.showbase import ShowBaseGlobal
from . import DirectGuiGlobals as DGG
from .DirectFrame import DirectFrame
from .OnscreenText import OnscreenText
import encodings.utf_8
from direct.showbase.DirectObject import DirectObject
ENTRY_FOCUS_STATE = PGEntry.SFocus
ENTRY_NO_FOCUS_STATE = PGEntry.SNoFocus
ENTRY_INACTIVE_STATE = PGEntry.SInactive

class DirectEntry(DirectFrame):
    """
    DirectEntry(parent) - Create a DirectGuiWidget which responds
    to keyboard buttons
    """
    directWtext = ConfigVariableBool('direct-wtext', True)
    AllowCapNamePrefixes = ('Al', 'Ap', 'Ben', 'De', 'Del', 'Della', 'Delle', 'Der', 'Di', 'Du', 'El', 'Fitz', 'La', 'Las', 'Le', 'Les', 'Lo', 'Los', 'Mac', 'St', 'Te', 'Ten', 'Van', 'Von')
    ForceCapNamePrefixes = ("D'", 'DeLa', "Dell'", "L'", "M'", 'Mc', "O'")

    def __init__(self, parent=None, **kw):
        if False:
            for i in range(10):
                print('nop')
        optiondefs = (('pgFunc', PGEntry, None), ('numStates', 3, None), ('state', DGG.NORMAL, None), ('entryFont', None, DGG.INITOPT), ('width', 10, self.updateWidth), ('numLines', 1, self.updateNumLines), ('focus', 0, self.setFocus), ('cursorKeys', 1, self.setCursorKeysActive), ('obscured', 0, self.setObscureMode), ('backgroundFocus', 0, self.setBackgroundFocus), ('initialText', '', DGG.INITOPT), ('overflow', 0, self.setOverflowMode), ('command', None, None), ('extraArgs', [], None), ('failedCommand', None, None), ('failedExtraArgs', [], None), ('focusInCommand', None, None), ('focusInExtraArgs', [], None), ('focusOutCommand', None, None), ('focusOutExtraArgs', [], None), ('rolloverSound', DGG.getDefaultRolloverSound(), self.setRolloverSound), ('clickSound', DGG.getDefaultClickSound(), self.setClickSound), ('autoCapitalize', 0, self.autoCapitalizeFunc), ('autoCapitalizeAllowPrefixes', DirectEntry.AllowCapNamePrefixes, None), ('autoCapitalizeForcePrefixes', DirectEntry.ForceCapNamePrefixes, None))
        self.defineoptions(kw, optiondefs)
        DirectFrame.__init__(self, parent)
        if self['entryFont'] is None:
            font = DGG.getDefaultFont()
        else:
            font = self['entryFont']
        self.onscreenText = self.createcomponent('text', (), None, OnscreenText, (), parent=ShowBaseGlobal.hidden, text='', align=TextNode.ALeft, font=font, scale=1, mayChange=1)
        self.onscreenText.removeNode()
        self.bind(DGG.ACCEPT, self.commandFunc)
        self.bind(DGG.ACCEPTFAILED, self.failedCommandFunc)
        self.accept(self.guiItem.getFocusInEvent(), self.focusInCommandFunc)
        self.accept(self.guiItem.getFocusOutEvent(), self.focusOutCommandFunc)
        self._autoCapListener = DirectObject()
        self.initialiseoptions(DirectEntry)
        if not hasattr(self, 'autoCapitalizeAllowPrefixes'):
            self.autoCapitalizeAllowPrefixes = DirectEntry.AllowCapNamePrefixes
        if not hasattr(self, 'autoCapitalizeForcePrefixes'):
            self.autoCapitalizeForcePrefixes = DirectEntry.ForceCapNamePrefixes
        for i in range(self['numStates']):
            self.guiItem.setTextDef(i, self.onscreenText.textNode)
        self.setup()
        self.unicodeText = 0
        if self['initialText']:
            self.enterText(self['initialText'])

    def destroy(self):
        if False:
            i = 10
            return i + 15
        self.ignoreAll()
        self._autoCapListener.ignoreAll()
        DirectFrame.destroy(self)

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.guiItem.setupMinimal(self['width'], self['numLines'])

    def updateWidth(self):
        if False:
            for i in range(10):
                print('nop')
        self.guiItem.setMaxWidth(self['width'])

    def updateNumLines(self):
        if False:
            i = 10
            return i + 15
        self.guiItem.setNumLines(self['numLines'])

    def setFocus(self):
        if False:
            return 10
        PGEntry.setFocus(self.guiItem, self['focus'])

    def setCursorKeysActive(self):
        if False:
            for i in range(10):
                print('nop')
        PGEntry.setCursorKeysActive(self.guiItem, self['cursorKeys'])

    def setOverflowMode(self):
        if False:
            for i in range(10):
                print('nop')
        PGEntry.set_overflow_mode(self.guiItem, self['overflow'])

    def setObscureMode(self):
        if False:
            for i in range(10):
                print('nop')
        PGEntry.setObscureMode(self.guiItem, self['obscured'])

    def setBackgroundFocus(self):
        if False:
            for i in range(10):
                print('nop')
        PGEntry.setBackgroundFocus(self.guiItem, self['backgroundFocus'])

    def setRolloverSound(self):
        if False:
            for i in range(10):
                print('nop')
        rolloverSound = self['rolloverSound']
        if rolloverSound:
            self.guiItem.setSound(DGG.ENTER + self.guiId, rolloverSound)
        else:
            self.guiItem.clearSound(DGG.ENTER + self.guiId)

    def setClickSound(self):
        if False:
            print('Hello World!')
        clickSound = self['clickSound']
        if clickSound:
            self.guiItem.setSound(DGG.ACCEPT + self.guiId, clickSound)
        else:
            self.guiItem.clearSound(DGG.ACCEPT + self.guiId)

    def commandFunc(self, event):
        if False:
            for i in range(10):
                print('nop')
        if self['command']:
            self['command'](*[self.get()] + self['extraArgs'])

    def failedCommandFunc(self, event):
        if False:
            print('Hello World!')
        if self['failedCommand']:
            self['failedCommand'](*[self.get()] + self['failedExtraArgs'])

    def autoCapitalizeFunc(self):
        if False:
            for i in range(10):
                print('nop')
        if self['autoCapitalize']:
            self._autoCapListener.accept(self.guiItem.getTypeEvent(), self._handleTyping)
            self._autoCapListener.accept(self.guiItem.getEraseEvent(), self._handleErasing)
        else:
            self._autoCapListener.ignore(self.guiItem.getTypeEvent())
            self._autoCapListener.ignore(self.guiItem.getEraseEvent())

    def focusInCommandFunc(self):
        if False:
            for i in range(10):
                print('nop')
        if self['focusInCommand']:
            self['focusInCommand'](*self['focusInExtraArgs'])
        if self['autoCapitalize']:
            self.accept(self.guiItem.getTypeEvent(), self._handleTyping)
            self.accept(self.guiItem.getEraseEvent(), self._handleErasing)

    def _handleTyping(self, guiEvent):
        if False:
            while True:
                i = 10
        self._autoCapitalize()

    def _handleErasing(self, guiEvent):
        if False:
            while True:
                i = 10
        self._autoCapitalize()

    def _autoCapitalize(self):
        if False:
            print('Hello World!')
        name = self.guiItem.getWtext()
        capName = ''
        wordSoFar = ''
        wasNonWordChar = True
        for (i, character) in enumerate(name):
            if character.lower() == character.upper() and character != "'":
                wordSoFar = ''
                wasNonWordChar = True
            else:
                capitalize = False
                if wasNonWordChar:
                    capitalize = True
                elif character == character.upper() and len(self.autoCapitalizeAllowPrefixes) > 0 and (wordSoFar in self.autoCapitalizeAllowPrefixes):
                    capitalize = True
                elif len(self.autoCapitalizeForcePrefixes) > 0 and wordSoFar in self.autoCapitalizeForcePrefixes:
                    capitalize = True
                if capitalize:
                    character = character.upper()
                else:
                    character = character.lower()
                wordSoFar += character
                wasNonWordChar = False
            capName += character
        self.guiItem.setWtext(capName)
        self.guiItem.setCursorPosition(self.guiItem.getNumCharacters())

    def focusOutCommandFunc(self):
        if False:
            i = 10
            return i + 15
        if self['focusOutCommand']:
            self['focusOutCommand'](*self['focusOutExtraArgs'])
        if self['autoCapitalize']:
            self.ignore(self.guiItem.getTypeEvent())
            self.ignore(self.guiItem.getEraseEvent())

    def set(self, text):
        if False:
            print('Hello World!')
        ' Changes the text currently showing in the typable region;\n        does not change the current cursor position.  Also see\n        enterText(). '
        assert not isinstance(text, bytes)
        self.unicodeText = True
        self.guiItem.setWtext(text)

    def get(self, plain=False):
        if False:
            return 10
        ' Returns the text currently showing in the typable region.\n        If plain is True, the returned text will not include any\n        formatting characters like nested color-change codes. '
        wantWide = self.unicodeText or self.guiItem.isWtext()
        if not self.directWtext.getValue():
            wantWide = False
        if plain:
            if wantWide:
                return self.guiItem.getPlainWtext()
            else:
                return self.guiItem.getPlainText()
        elif wantWide:
            return self.guiItem.getWtext()
        else:
            return self.guiItem.getText()

    def getCursorPosition(self):
        if False:
            i = 10
            return i + 15
        return self.guiItem.getCursorPosition()

    def setCursorPosition(self, pos):
        if False:
            while True:
                i = 10
        if pos < 0:
            self.guiItem.setCursorPosition(self.guiItem.getNumCharacters() + pos)
        else:
            self.guiItem.setCursorPosition(pos)

    def getNumCharacters(self):
        if False:
            print('Hello World!')
        return self.guiItem.getNumCharacters()

    def enterText(self, text):
        if False:
            while True:
                i = 10
        " sets the entry's text, and moves the cursor to the end "
        self.set(text)
        self.setCursorPosition(self.guiItem.getNumCharacters())

    def getFont(self):
        if False:
            return 10
        return self.onscreenText.getFont()

    def getBounds(self, state=0):
        if False:
            for i in range(10):
                print('nop')
        tn = self.onscreenText.textNode
        mat = tn.getTransform()
        align = tn.getAlign()
        lineHeight = tn.getLineHeight()
        numLines = self['numLines']
        width = self['width']
        if align == TextNode.ALeft:
            left = 0.0
            right = width
        elif align == TextNode.ACenter:
            left = -width / 2.0
            right = width / 2.0
        elif align == TextNode.ARight:
            left = -width
            right = 0.0
        bottom = -0.3 * lineHeight - lineHeight * (numLines - 1)
        top = lineHeight
        self.ll.set(left, 0.0, bottom)
        self.ur.set(right, 0.0, top)
        self.ll = mat.xformPoint(Point3.rfu(left, 0.0, bottom))
        self.ur = mat.xformPoint(Point3.rfu(right, 0.0, top))
        vec_right = Vec3.right()
        vec_up = Vec3.up()
        left = vec_right[0] * self.ll[0] + vec_right[1] * self.ll[1] + vec_right[2] * self.ll[2]
        right = vec_right[0] * self.ur[0] + vec_right[1] * self.ur[1] + vec_right[2] * self.ur[2]
        bottom = vec_up[0] * self.ll[0] + vec_up[1] * self.ll[1] + vec_up[2] * self.ll[2]
        top = vec_up[0] * self.ur[0] + vec_up[1] * self.ur[1] + vec_up[2] * self.ur[2]
        self.ll = Point3(left, 0.0, bottom)
        self.ur = Point3(right, 0.0, top)
        pad = self['pad']
        borderWidth = self['borderWidth']
        self.bounds = [self.ll[0] - pad[0] - borderWidth[0], self.ur[0] + pad[0] + borderWidth[0], self.ll[2] - pad[1] - borderWidth[1], self.ur[2] + pad[1] + borderWidth[1]]
        return self.bounds