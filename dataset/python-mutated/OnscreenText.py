"""OnscreenText module: contains the OnscreenText class.

See the :ref:`onscreentext` page in the programming manual for explanation of
this class.
"""
__all__ = ['OnscreenText', 'Plain', 'ScreenTitle', 'ScreenPrompt', 'NameConfirm', 'BlackOnWhite']
from panda3d.core import LColor, Mat4, NodePath, Point3, TextNode, TextProperties, Vec3
from . import DirectGuiGlobals as DGG
import warnings
Plain = 1
ScreenTitle = 2
ScreenPrompt = 3
NameConfirm = 4
BlackOnWhite = 5

class OnscreenText(NodePath):

    def __init__(self, text='', style=Plain, pos=(0, 0), roll=0, scale=None, fg=None, bg=None, shadow=None, shadowOffset=(0.04, 0.04), frame=None, align=None, wordwrap=None, drawOrder=None, decal=0, font=None, parent=None, sort=0, mayChange=True, direction=None):
        if False:
            while True:
                i = 10
        "\n        Make a text node from string, put it into the 2d sg and set it\n        up with all the indicated parameters.\n\n        Parameters:\n\n          text: the actual text to display.  This may be omitted and\n              specified later via setText() if you don't have it\n              available, but it is better to specify it up front.\n\n          style: one of the pre-canned style parameters defined at the\n              head of this file.  This sets up the default values for\n              many of the remaining parameters if they are\n              unspecified; however, a parameter may still be specified\n              to explicitly set it, overriding the pre-canned style.\n\n          pos: the x, y position of the text on the screen.\n\n          scale: the size of the text.  This may either be a single\n              float (and it will usually be a small number like 0.07)\n              or it may be a 2-tuple of floats, specifying a different\n              x, y scale.\n\n          fg: the (r, g, b, a) foreground color of the text.  This is\n              normally a 4-tuple of floats or ints.\n\n          bg: the (r, g, b, a) background color of the text.  If the\n              fourth value, a, is nonzero, a card is created to place\n              behind the text and set to the given color.\n\n          shadow: the (r, g, b, a) color of the shadow behind the text.\n              If the fourth value, a, is nonzero, a little drop shadow\n              is created and placed behind the text.\n\n          frame: the (r, g, b, a) color of the frame drawn around the\n              text.  If the fourth value, a, is nonzero, a frame is\n              created around the text.\n\n          align: one of TextNode.ALeft, TextNode.ARight, or TextNode.ACenter.\n\n          wordwrap: either the width to wordwrap the text at, or None\n              to specify no automatic word wrapping.\n\n          drawOrder: the drawing order of this text with respect to\n              all other things in the 'fixed' bin within render2d.\n              The text will actually use drawOrder through drawOrder +\n              2.\n\n          decal: if this is True, the text is decalled onto its\n              background card.  Useful when the text will be parented\n              into the 3-D scene graph.\n\n          font: the font to use for the text.\n\n          parent: the NodePath to parent the text to initially.\n\n          mayChange: pass true if the text or its properties may need\n              to be changed at runtime, false if it is static once\n              created (which leads to better memory optimization).\n\n          direction: this can be set to 'ltr' or 'rtl' to override the\n              direction of the text.\n        "
        if parent is None:
            from direct.showbase import ShowBaseGlobal
            parent = ShowBaseGlobal.aspect2d
        textNode = TextNode('')
        self.textNode = textNode
        NodePath.__init__(self)
        if style == Plain:
            scale = scale or 0.07
            fg = fg or (0, 0, 0, 1)
            bg = bg or (0, 0, 0, 0)
            shadow = shadow or (0, 0, 0, 0)
            frame = frame or (0, 0, 0, 0)
            if align is None:
                align = TextNode.ACenter
        elif style == ScreenTitle:
            scale = scale or 0.15
            fg = fg or (1, 0.2, 0.2, 1)
            bg = bg or (0, 0, 0, 0)
            shadow = shadow or (0, 0, 0, 1)
            frame = frame or (0, 0, 0, 0)
            if align is None:
                align = TextNode.ACenter
        elif style == ScreenPrompt:
            scale = scale or 0.1
            fg = fg or (1, 1, 0, 1)
            bg = bg or (0, 0, 0, 0)
            shadow = shadow or (0, 0, 0, 1)
            frame = frame or (0, 0, 0, 0)
            if align is None:
                align = TextNode.ACenter
        elif style == NameConfirm:
            scale = scale or 0.1
            fg = fg or (0, 1, 0, 1)
            bg = bg or (0, 0, 0, 0)
            shadow = shadow or (0, 0, 0, 0)
            frame = frame or (0, 0, 0, 0)
            if align is None:
                align = TextNode.ACenter
        elif style == BlackOnWhite:
            scale = scale or 0.1
            fg = fg or (0, 0, 0, 1)
            bg = bg or (1, 1, 1, 1)
            shadow = shadow or (0, 0, 0, 0)
            frame = frame or (0, 0, 0, 0)
            if align is None:
                align = TextNode.ACenter
        else:
            raise ValueError
        if not isinstance(scale, tuple):
            scale = (scale, scale)
        self.__scale = scale
        self.__pos = pos
        self.__roll = roll
        self.__wordwrap = wordwrap
        if decal:
            textNode.setCardDecal(True)
        if font is None:
            font = DGG.getDefaultFont()
        textNode.setFont(font)
        textNode.setTextColor(fg[0], fg[1], fg[2], fg[3])
        textNode.setAlign(align)
        if wordwrap:
            textNode.setWordwrap(wordwrap)
        if bg[3] != 0:
            textNode.setCardColor(bg[0], bg[1], bg[2], bg[3])
            textNode.setCardAsMargin(0.1, 0.1, 0.1, 0.1)
        if shadow[3] != 0:
            textNode.setShadowColor(shadow[0], shadow[1], shadow[2], shadow[3])
            textNode.setShadow(*shadowOffset)
        if frame[3] != 0:
            textNode.setFrameColor(frame[0], frame[1], frame[2], frame[3])
            textNode.setFrameAsMargin(0.1, 0.1, 0.1, 0.1)
        if direction is not None:
            if isinstance(direction, str):
                direction = direction.lower()
                if direction == 'rtl':
                    direction = TextProperties.D_rtl
                elif direction == 'ltr':
                    direction = TextProperties.D_ltr
                else:
                    raise ValueError('invalid direction')
            textNode.setDirection(direction)
        self.updateTransformMat()
        if drawOrder is not None:
            textNode.setBin('fixed')
            textNode.setDrawOrder(drawOrder)
        self.setText(text)
        if not text:
            self.mayChange = 1
        else:
            self.mayChange = mayChange
        if not self.mayChange:
            self.textNode = textNode.generate()
        self.isClean = 0
        self.assign(parent.attachNewNode(self.textNode, sort))

    def cleanup(self):
        if False:
            print('Hello World!')
        self.textNode = None
        if self.isClean == 0:
            self.isClean = 1
            self.removeNode()

    def destroy(self):
        if False:
            for i in range(10):
                print('nop')
        self.cleanup()

    def freeze(self):
        if False:
            print('Hello World!')
        pass

    def thaw(self):
        if False:
            return 10
        pass

    def setDecal(self, decal):
        if False:
            for i in range(10):
                print('nop')
        self.textNode.setCardDecal(decal)

    def getDecal(self):
        if False:
            return 10
        return self.textNode.getCardDecal()
    decal = property(getDecal, setDecal)

    def setFont(self, font):
        if False:
            print('Hello World!')
        self.textNode.setFont(font)

    def getFont(self):
        if False:
            print('Hello World!')
        return self.textNode.getFont()
    font = property(getFont, setFont)

    def clearText(self):
        if False:
            for i in range(10):
                print('nop')
        self.textNode.clearText()

    def setText(self, text):
        if False:
            while True:
                i = 10
        assert not isinstance(text, bytes)
        self.textNode.setWtext(text)

    def appendText(self, text):
        if False:
            while True:
                i = 10
        assert not isinstance(text, bytes)
        self.textNode.appendWtext(text)

    def getText(self):
        if False:
            while True:
                i = 10
        return self.textNode.getWtext()
    text = property(getText, setText)

    def setTextX(self, x):
        if False:
            print('Hello World!')
        '\n        .. versionadded:: 1.10.8\n        '
        self.setTextPos(x, self.__pos[1])

    def setX(self, x):
        if False:
            i = 10
            return i + 15
        '\n        .. deprecated:: 1.11.0\n           Use `.setTextX()` method instead.\n        '
        if __debug__:
            warnings.warn('Use `.setTextX()` method instead.', DeprecationWarning, stacklevel=2)
        self.setTextPos(x, self.__pos[1])

    def setTextY(self, y):
        if False:
            i = 10
            return i + 15
        '\n        .. versionadded:: 1.10.8\n        '
        self.setTextPos(self.__pos[0], y)

    def setY(self, y):
        if False:
            return 10
        '\n        .. deprecated:: 1.11.0\n           Use `.setTextY()` method instead.\n        '
        if __debug__:
            warnings.warn('Use `.setTextY()` method instead.', DeprecationWarning, stacklevel=2)
        self.setTextPos(self.__pos[0], y)

    def setTextPos(self, x, y=None):
        if False:
            i = 10
            return i + 15
        '\n        Position the onscreen text in 2d screen space\n\n        .. versionadded:: 1.10.8\n        '
        if y is None:
            self.__pos = tuple(x)
        else:
            self.__pos = (x, y)
        self.updateTransformMat()

    def getTextPos(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        .. versionadded:: 1.10.8\n        '
        return self.__pos
    text_pos = property(getTextPos, setTextPos)

    def setPos(self, x, y):
        if False:
            print('Hello World!')
        'setPos(self, float, float)\n        Position the onscreen text in 2d screen space\n\n        .. deprecated:: 1.11.0\n           Use `.setTextPos()` method or `.text_pos` property instead.\n        '
        if __debug__:
            warnings.warn('Use `.setTextPos()` method or `.text_pos` property instead.', DeprecationWarning, stacklevel=2)
        self.__pos = (x, y)
        self.updateTransformMat()

    def getPos(self):
        if False:
            i = 10
            return i + 15
        '\n        .. deprecated:: 1.11.0\n           Use `.getTextPos()` method or `.text_pos` property instead.\n        '
        if __debug__:
            warnings.warn('Use `.getTextPos()` method or `.text_pos` property instead.', DeprecationWarning, stacklevel=2)
        return self.__pos
    pos = property(getPos)

    def setTextR(self, r):
        if False:
            while True:
                i = 10
        "setTextR(self, float)\n        Rotates the text around the screen's normal.\n\n        .. versionadded:: 1.10.8\n        "
        self.__roll = -r
        self.updateTransformMat()

    def getTextR(self):
        if False:
            print('Hello World!')
        return -self.__roll
    text_r = property(getTextR, setTextR)

    def setRoll(self, roll):
        if False:
            for i in range(10):
                print('nop')
        "setRoll(self, float)\n        Rotate the onscreen text around the screen's normal.\n\n        .. deprecated:: 1.11.0\n           Use ``setTextR(-roll)`` instead (note the negated sign).\n        "
        if __debug__:
            warnings.warn('Use ``setTextR(-roll)`` instead (note the negated sign).', DeprecationWarning, stacklevel=2)
        self.__roll = roll
        self.updateTransformMat()

    def getRoll(self):
        if False:
            while True:
                i = 10
        '\n        .. deprecated:: 1.11.0\n           Use ``-getTextR()`` instead (note the negated sign).\n        '
        if __debug__:
            warnings.warn('Use ``-getTextR()`` instead (note the negated sign).', DeprecationWarning, stacklevel=2)
        return self.__roll
    roll = property(getRoll, setRoll)

    def setTextScale(self, sx, sy=None):
        if False:
            while True:
                i = 10
        'setTextScale(self, float, float)\n        Scale the text in 2d space.  You may specify either a single\n        uniform scale, or two scales, or a tuple of two scales.\n\n        .. versionadded:: 1.10.8\n        '
        if sy is None:
            if isinstance(sx, tuple):
                self.__scale = sx
            else:
                self.__scale = (sx, sx)
        else:
            self.__scale = (sx, sy)
        self.updateTransformMat()

    def getTextScale(self):
        if False:
            return 10
        '\n        .. versionadded:: 1.10.8\n        '
        return self.__scale
    text_scale = property(getTextScale, setTextScale)

    def setScale(self, sx, sy=None):
        if False:
            i = 10
            return i + 15
        'setScale(self, float, float)\n        Scale the text in 2d space.  You may specify either a single\n        uniform scale, or two scales, or a tuple of two scales.\n\n        .. deprecated:: 1.11.0\n           Use `.setTextScale()` method or `.text_scale` property instead.\n        '
        if __debug__:
            warnings.warn('Use `.setTextScale()` method or `.text_scale` property instead.', DeprecationWarning, stacklevel=2)
        if sy is None:
            if isinstance(sx, tuple):
                self.__scale = sx
            else:
                self.__scale = (sx, sx)
        else:
            self.__scale = (sx, sy)
        self.updateTransformMat()

    def getScale(self):
        if False:
            return 10
        '\n        .. deprecated:: 1.11.0\n           Use `.getTextScale()` method or `.text_scale` property instead.\n        '
        if __debug__:
            warnings.warn('Use `.getTextScale()` method or `.text_scale` property instead.', DeprecationWarning, stacklevel=2)
        return self.__scale
    scale = property(getScale, setScale)

    def updateTransformMat(self):
        if False:
            print('Hello World!')
        assert isinstance(self.textNode, TextNode)
        mat = Mat4.scaleMat(Vec3.rfu(self.__scale[0], 1, self.__scale[1])) * Mat4.rotateMat(self.__roll, Vec3.back()) * Mat4.translateMat(Point3.rfu(self.__pos[0], 0, self.__pos[1]))
        self.textNode.setTransform(mat)

    def setWordwrap(self, wordwrap):
        if False:
            return 10
        self.__wordwrap = wordwrap
        if wordwrap:
            self.textNode.setWordwrap(wordwrap)
        else:
            self.textNode.clearWordwrap()

    def getWordwrap(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__wordwrap
    wordwrap = property(getWordwrap, setWordwrap)

    def __getFg(self):
        if False:
            for i in range(10):
                print('nop')
        return self.textNode.getTextColor()

    def setFg(self, fg):
        if False:
            print('Hello World!')
        self.textNode.setTextColor(fg[0], fg[1], fg[2], fg[3])
    fg = property(__getFg, setFg)

    def __getBg(self):
        if False:
            i = 10
            return i + 15
        if self.textNode.hasCard():
            return self.textNode.getCardColor()
        else:
            return LColor(0)

    def setBg(self, bg):
        if False:
            while True:
                i = 10
        if bg[3] != 0:
            self.textNode.setCardColor(bg[0], bg[1], bg[2], bg[3])
            self.textNode.setCardAsMargin(0.1, 0.1, 0.1, 0.1)
        else:
            self.textNode.clearCard()
    bg = property(__getBg, setBg)

    def __getShadow(self):
        if False:
            for i in range(10):
                print('nop')
        return self.textNode.getShadowColor()

    def setShadow(self, shadow):
        if False:
            print('Hello World!')
        if shadow[3] != 0:
            self.textNode.setShadowColor(shadow[0], shadow[1], shadow[2], shadow[3])
            self.textNode.setShadow(0.04, 0.04)
        else:
            self.textNode.clearShadow()
    shadow = property(__getShadow, setShadow)

    def __getFrame(self):
        if False:
            while True:
                i = 10
        return self.textNode.getFrameColor()

    def setFrame(self, frame):
        if False:
            i = 10
            return i + 15
        if frame[3] != 0:
            self.textNode.setFrameColor(frame[0], frame[1], frame[2], frame[3])
            self.textNode.setFrameAsMargin(0.1, 0.1, 0.1, 0.1)
        else:
            self.textNode.clearFrame()
    frame = property(__getFrame, setFrame)

    def configure(self, option=None, **kw):
        if False:
            return 10
        if not self.mayChange:
            print('OnscreenText.configure: mayChange == 0')
            return
        for (option, value) in kw.items():
            try:
                if option == 'pos':
                    self.setTextPos(value[0], value[1])
                elif option == 'roll':
                    self.setTextR(-value)
                elif option == 'scale':
                    self.setTextScale(value)
                elif option == 'x':
                    self.setTextX(value)
                elif option == 'y':
                    self.setTextY(value)
                else:
                    setter = getattr(self, 'set' + option[0].upper() + option[1:])
                    setter(value)
            except AttributeError:
                print('OnscreenText.configure: invalid option: %s' % option)

    def __setitem__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        self.configure(*(), **{key: value})

    def cget(self, option):
        if False:
            i = 10
            return i + 15
        if option == 'pos':
            return self.__pos
        elif option == 'roll':
            return self.__roll
        elif option == 'scale':
            return self.__scale
        elif option == 'x':
            return self.__pos[0]
        elif option == 'y':
            return self.__pos[1]
        getter = getattr(self, 'get' + option[0].upper() + option[1:])
        return getter()

    def __getAlign(self):
        if False:
            print('Hello World!')
        return self.textNode.getAlign()

    def setAlign(self, align):
        if False:
            i = 10
            return i + 15
        self.textNode.setAlign(align)
    align = property(__getAlign, setAlign)
    __getitem__ = cget