from PyQt5.QtWidgets import QWidget
from PyQt5.QtDesigner import QPyDesignerCustomWidgetPlugin, QDesignerFormWindowInterface, QExtensionFactory, QPyDesignerContainerExtension
from qfluentwidgets import ScrollArea, SmoothScrollArea, SingleDirectionScrollArea, OpacityAniStackedWidget, PopUpAniStackedWidget, CardWidget, ElevatedCardWidget, SimpleCardWidget, HeaderCardWidget
from plugin_base import PluginBase

class ContainerPlugin(PluginBase):

    def group(self):
        if False:
            i = 10
            return i + 15
        return super().group() + ' (Container)'

    def isContainer(self):
        if False:
            while True:
                i = 10
        return True

class CardWidgetPlugin(ContainerPlugin, QPyDesignerCustomWidgetPlugin):
    """ Card widget plugin """

    def createWidget(self, parent):
        if False:
            for i in range(10):
                print('nop')
        return CardWidget(parent)

    def icon(self):
        if False:
            while True:
                i = 10
        return super().icon('CommandBar')

    def name(self):
        if False:
            i = 10
            return i + 15
        return 'CardWidget'

class ElevatedCardWidgetPlugin(ContainerPlugin, QPyDesignerCustomWidgetPlugin):
    """ Elevated card widget plugin """

    def createWidget(self, parent):
        if False:
            print('Hello World!')
        return ElevatedCardWidget(parent)

    def icon(self):
        if False:
            for i in range(10):
                print('nop')
        return super().icon('CommandBar')

    def name(self):
        if False:
            print('Hello World!')
        return 'ElevatedCardWidget'

class SimpleCardWidgetPlugin(ContainerPlugin, QPyDesignerCustomWidgetPlugin):
    """ Simple card widget plugin """

    def createWidget(self, parent):
        if False:
            while True:
                i = 10
        return SimpleCardWidget(parent)

    def icon(self):
        if False:
            print('Hello World!')
        return super().icon('CommandBar')

    def name(self):
        if False:
            print('Hello World!')
        return 'SimpleCardWidget'

class HeaderCardWidgetPlugin(ContainerPlugin, QPyDesignerCustomWidgetPlugin):
    """ Header card widget plugin """

    def createWidget(self, parent):
        if False:
            while True:
                i = 10
        return HeaderCardWidget(parent)

    def icon(self):
        if False:
            while True:
                i = 10
        return super().icon('CommandBar')

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'HeaderCardWidget'

class ScrollAreaPluginBase(ContainerPlugin):
    """ Scroll area plugin base """

    def domXml(self):
        if False:
            i = 10
            return i + 15
        return f'\n            <widget class="{self.name()}" name="{self.name()}">\n                <property name="widgetResizable">\n                    <bool>true</bool>\n                </property>\n                <widget class="QWidget" name="scrollAreaWidgetContents" />\n            </widget>\n        '

class ScrollAreaPlugin(ScrollAreaPluginBase, QPyDesignerCustomWidgetPlugin):
    """ Scroll area plugin """

    def createWidget(self, parent):
        if False:
            i = 10
            return i + 15
        return ScrollArea(parent)

    def icon(self):
        if False:
            while True:
                i = 10
        return super().icon('ScrollViewer')

    def name(self):
        if False:
            return 10
        return 'ScrollArea'

    def toolTip(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Smooth scroll area'

class SmoothScrollAreaPlugin(ScrollAreaPluginBase, QPyDesignerCustomWidgetPlugin):
    """ Smooth scroll area plugin """

    def createWidget(self, parent):
        if False:
            while True:
                i = 10
        return SmoothScrollArea(parent)

    def icon(self):
        if False:
            return 10
        return super().icon('ScrollViewer')

    def name(self):
        if False:
            print('Hello World!')
        return 'SmoothScrollArea'

class SingleDirectionScrollAreaPlugin(ScrollAreaPluginBase, QPyDesignerCustomWidgetPlugin):
    """ Single direction scroll area plugin """

    def createWidget(self, parent):
        if False:
            while True:
                i = 10
        return SingleDirectionScrollArea(parent)

    def icon(self):
        if False:
            print('Hello World!')
        return super().icon('ScrollViewer')

    def name(self):
        if False:
            print('Hello World!')
        return 'SingleDirectionScrollArea'

class StackedWidgetPlugin(ContainerPlugin):

    def domXml(self):
        if False:
            for i in range(10):
                print('nop')
        return f'''\n            <widget class="{self.name()}" name="{self.name()}">'\n                <widget class="QWidget" name="page" />'\n            </widget>\n        '''

    def onCurrentIndexChanged(self, index):
        if False:
            while True:
                i = 10
        widget = self.sender()
        form = QDesignerFormWindowInterface.findFormWindow(widget)
        if form:
            form.emitSelectionChanged()

class StackedWidgetExtension(QPyDesignerContainerExtension):
    """ Stacked widget extension """

    def __init__(self, stacked, parent=None) -> None:
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self.stacked = stacked

    def addWidget(self, widget) -> None:
        if False:
            i = 10
            return i + 15
        self.stacked.addWidget(widget)

    def count(self):
        if False:
            print('Hello World!')
        return self.stacked.count()

    def currentIndex(self):
        if False:
            print('Hello World!')
        return self.stacked.currentIndex()

    def insertWidget(self, index, widget):
        if False:
            print('Hello World!')
        self.stacked.insertWidget(index, widget)

    def remove(self, index):
        if False:
            print('Hello World!')
        self.stacked.removeWidget(self.stacked.widget(index))

    def setCurrentIndex(self, index):
        if False:
            for i in range(10):
                print('nop')
        self.stacked.setCurrentIndex(index)

    def widget(self, index):
        if False:
            i = 10
            return i + 15
        return self.stacked.widget(index)

class StackedWidgetExtensionFactory(QExtensionFactory):
    """ Stacked widget extension factory """
    widgets = []
    IID = 'org.qt-project.Qt.Designer.Container'

    def createExtension(self, object, iid, parent):
        if False:
            return 10
        if iid != StackedWidgetExtensionFactory.IID:
            return None
        if object.__class__.__name__ not in self.widgets:
            return None
        return StackedWidgetExtension(object, parent)

    @classmethod
    def register(cls, Plugin):
        if False:
            i = 10
            return i + 15
        if Plugin.__name__ not in cls.widgets:
            cls.widgets.append(Plugin().name())
            Plugin.Factory = cls
        return Plugin

@StackedWidgetExtensionFactory.register
class OpacityAniStackedWidgetPlugin(StackedWidgetPlugin, QPyDesignerCustomWidgetPlugin):
    """ opacity ani stacked widget plugin """

    def createWidget(self, parent):
        if False:
            for i in range(10):
                print('nop')
        w = OpacityAniStackedWidget(parent)
        w.currentChanged.connect(self.onCurrentIndexChanged)
        return w

    def icon(self):
        if False:
            for i in range(10):
                print('nop')
        return super().icon('StackPanel')

    def name(self):
        if False:
            i = 10
            return i + 15
        return 'OpacityAniStackedWidget'

@StackedWidgetExtensionFactory.register
class PopUpAniStackedWidgetPlugin(StackedWidgetPlugin, QPyDesignerCustomWidgetPlugin):
    """ pop up ani stacked widget plugin """

    def createWidget(self, parent):
        if False:
            i = 10
            return i + 15
        w = PopUpAniStackedWidget(parent)
        w.currentChanged.connect(self.onCurrentIndexChanged)
        return w

    def icon(self):
        if False:
            print('Hello World!')
        return super().icon('StackPanel')

    def name(self):
        if False:
            print('Hello World!')
        return 'PopUpAniStackedWidget'