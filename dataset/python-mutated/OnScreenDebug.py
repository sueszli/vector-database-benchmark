"""Contains the OnScreenDebug class."""
__all__ = ['OnScreenDebug']
from panda3d.core import ConfigVariableBool, ConfigVariableDouble, ConfigVariableString, TextNode, Vec4
from direct.gui import OnscreenText
from direct.directtools import DirectUtil

class OnScreenDebug:
    enabled = ConfigVariableBool('on-screen-debug-enabled', False)

    def __init__(self):
        if False:
            while True:
                i = 10
        self.onScreenText = None
        self.frame = 0
        self.text = ''
        self.data = {}

    def load(self):
        if False:
            while True:
                i = 10
        if self.onScreenText:
            return
        fontPath = ConfigVariableString('on-screen-debug-font', 'cmtt12').value
        fontScale = ConfigVariableDouble('on-screen-debug-font-scale', 0.05).value
        color = {'black': Vec4(0, 0, 0, 1), 'white': Vec4(1, 1, 1, 1)}
        fgColor = color[ConfigVariableString('on-screen-debug-fg-color', 'white').value]
        bgColor = color[ConfigVariableString('on-screen-debug-bg-color', 'black').value]
        fgColor.setW(ConfigVariableDouble('on-screen-debug-fg-alpha', 0.85).value)
        bgColor.setW(ConfigVariableDouble('on-screen-debug-bg-alpha', 0.85).value)
        font = base.loader.loadFont(fontPath)
        if not font.isValid():
            print('failed to load OnScreenDebug font %s' % fontPath)
            font = TextNode.getDefaultFont()
        self.onScreenText = OnscreenText.OnscreenText(parent=base.a2dTopLeft, pos=(0.0, -0.1), fg=fgColor, bg=bgColor, scale=(fontScale, fontScale, 0.0), align=TextNode.ALeft, mayChange=1, font=font)
        DirectUtil.useDirectRenderStyle(self.onScreenText)

    def render(self):
        if False:
            while True:
                i = 10
        if not self.enabled:
            return
        if not self.onScreenText:
            self.load()
        self.onScreenText.clearText()
        for (k, v) in sorted(self.data.items()):
            if v[0] == self.frame:
                isNew = '='
            else:
                isNew = '~'
            value = v[1]
            if isinstance(value, float):
                value = '% 10.4f' % (value,)
            self.onScreenText.appendText('%20s %s %-44s\n' % (k, isNew, value))
        self.onScreenText.appendText(self.text)
        self.frame += 1

    def clear(self):
        if False:
            return 10
        self.text = ''
        if self.onScreenText:
            self.onScreenText.clearText()

    def add(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        self.data[key] = (self.frame, value)
        return 1

    def has(self, key):
        if False:
            while True:
                i = 10
        return key in self.data

    def remove(self, key):
        if False:
            for i in range(10):
                print('nop')
        del self.data[key]

    def removeAllWithPrefix(self, prefix):
        if False:
            while True:
                i = 10
        toRemove = []
        for key in list(self.data.keys()):
            if len(key) >= len(prefix):
                if key[:len(prefix)] == prefix:
                    toRemove.append(key)
        for key in toRemove:
            self.remove(key)

    def append(self, text):
        if False:
            print('Hello World!')
        self.text += text