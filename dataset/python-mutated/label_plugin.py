from PyQt5.QtCore import Qt
from PyQt5.QtDesigner import QPyDesignerCustomWidgetPlugin
from qfluentwidgets import BodyLabel, CaptionLabel, StrongBodyLabel, SubtitleLabel, TitleLabel, LargeTitleLabel, DisplayLabel, ImageLabel, AvatarWidget, HyperlinkLabel
from plugin_base import PluginBase

class LabelPlugin(PluginBase):

    def group(self):
        if False:
            while True:
                i = 10
        return super().group() + ' (Label)'

    def domXml(self):
        if False:
            i = 10
            return i + 15
        return f'\n        <widget class="{self.name()}" name="{self.name()}">\n            <property name="text">\n                <string>{self.toolTip()}</string>\n            </property>\n        </widget>\n        '

class CaptionLabelPlugin(LabelPlugin, QPyDesignerCustomWidgetPlugin):
    """ Caption label plugin """

    def createWidget(self, parent):
        if False:
            while True:
                i = 10
        return CaptionLabel(parent)

    def icon(self):
        if False:
            print('Hello World!')
        return super().icon('TextBlock')

    def name(self):
        if False:
            return 10
        return 'CaptionLabel'

class BodyLabelPlugin(LabelPlugin, QPyDesignerCustomWidgetPlugin):
    """ Body label plugin """

    def createWidget(self, parent):
        if False:
            for i in range(10):
                print('nop')
        return BodyLabel(parent)

    def icon(self):
        if False:
            return 10
        return super().icon('TextBlock')

    def name(self):
        if False:
            print('Hello World!')
        return 'BodyLabel'

class StrongBodyLabelPlugin(LabelPlugin, QPyDesignerCustomWidgetPlugin):
    """ Strong body label plugin """

    def createWidget(self, parent):
        if False:
            for i in range(10):
                print('nop')
        return StrongBodyLabel(parent)

    def icon(self):
        if False:
            while True:
                i = 10
        return super().icon('TextBlock')

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'StrongBodyLabel'

class SubtitleLabelPlugin(LabelPlugin, QPyDesignerCustomWidgetPlugin):
    """ Subtitle label plugin """

    def createWidget(self, parent):
        if False:
            return 10
        return SubtitleLabel(parent)

    def icon(self):
        if False:
            while True:
                i = 10
        return super().icon('TextBlock')

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'SubtitleLabel'

class TitleLabelPlugin(LabelPlugin, QPyDesignerCustomWidgetPlugin):
    """ Title label plugin """

    def createWidget(self, parent):
        if False:
            print('Hello World!')
        return TitleLabel(parent)

    def icon(self):
        if False:
            while True:
                i = 10
        return super().icon('TextBlock')

    def name(self):
        if False:
            i = 10
            return i + 15
        return 'TitleLabel'

class LargeTitleLabelPlugin(LabelPlugin, QPyDesignerCustomWidgetPlugin):
    """ Title label plugin """

    def createWidget(self, parent):
        if False:
            i = 10
            return i + 15
        return LargeTitleLabel(parent)

    def icon(self):
        if False:
            i = 10
            return i + 15
        return super().icon('TextBlock')

    def name(self):
        if False:
            return 10
        return 'LargeTitleLabel'

class DisplayLabelPlugin(LabelPlugin, QPyDesignerCustomWidgetPlugin):
    """ Display label plugin """

    def createWidget(self, parent):
        if False:
            return 10
        return DisplayLabel(parent)

    def icon(self):
        if False:
            while True:
                i = 10
        return super().icon('TextBlock')

    def name(self):
        if False:
            print('Hello World!')
        return 'DisplayLabel'

class HyperlinkLabelPlugin(LabelPlugin, QPyDesignerCustomWidgetPlugin):
    """ Hyperlink label plugin """

    def createWidget(self, parent):
        if False:
            i = 10
            return i + 15
        return HyperlinkLabel(parent)

    def icon(self):
        if False:
            i = 10
            return i + 15
        return super().icon('HyperlinkButton')

    def name(self):
        if False:
            while True:
                i = 10
        return 'HyperlinkLabel'

class ImageLabelPlugin(LabelPlugin, QPyDesignerCustomWidgetPlugin):
    """ Image label plugin """

    def createWidget(self, parent):
        if False:
            i = 10
            return i + 15
        return ImageLabel(self.icon().pixmap(72, 72), parent)

    def icon(self):
        if False:
            print('Hello World!')
        return super().icon('Image')

    def name(self):
        if False:
            return 10
        return 'ImageLabel'

    def domXml(self):
        if False:
            print('Hello World!')
        return f'<widget class="{self.name()}" name="{self.name()}"></widget>'

class AvatarPlugin(LabelPlugin, QPyDesignerCustomWidgetPlugin):
    """ Avatar plugin """

    def createWidget(self, parent):
        if False:
            return 10
        return AvatarWidget(self.icon().pixmap(72, 72), parent)

    def icon(self):
        if False:
            return 10
        return super().icon('PersonPicture')

    def name(self):
        if False:
            return 10
        return 'AvatarWidget'

    def domXml(self):
        if False:
            return 10
        return f'<widget class="{self.name()}" name="{self.name()}"></widget>'