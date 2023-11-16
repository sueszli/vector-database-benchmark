from PyQt5.QtCore import Qt, QSize
from PyQt5.QtDesigner import QPyDesignerCustomWidgetPlugin
from qfluentwidgets import ListWidget, ListView, TreeView, TreeWidget, TableView, TableWidget, HorizontalFlipView, VerticalFlipView, HorizontalPipsPager, VerticalPipsPager
from plugin_base import PluginBase

class ViewPlugin(PluginBase):

    def group(self):
        if False:
            print('Hello World!')
        return super().group() + ' (View)'

class ListWidgetPlugin(ViewPlugin, QPyDesignerCustomWidgetPlugin):
    """ List widget plugin """

    def createWidget(self, parent):
        if False:
            i = 10
            return i + 15
        return ListWidget(parent)

    def icon(self):
        if False:
            return 10
        return super().icon('ListView')

    def name(self):
        if False:
            print('Hello World!')
        return 'ListWidget'

class ListViewPlugin(ViewPlugin, QPyDesignerCustomWidgetPlugin):
    """ List view plugin """

    def createWidget(self, parent):
        if False:
            for i in range(10):
                print('nop')
        return ListView(parent)

    def icon(self):
        if False:
            for i in range(10):
                print('nop')
        return super().icon('ListView')

    def name(self):
        if False:
            i = 10
            return i + 15
        return 'ListView'

class TableWidgetPlugin(ViewPlugin, QPyDesignerCustomWidgetPlugin):
    """ Table widget plugin """

    def createWidget(self, parent):
        if False:
            for i in range(10):
                print('nop')
        return TableWidget(parent)

    def icon(self):
        if False:
            print('Hello World!')
        return super().icon('DataGrid')

    def name(self):
        if False:
            while True:
                i = 10
        return 'TableWidget'

class TableViewPlugin(ViewPlugin, QPyDesignerCustomWidgetPlugin):
    """ Table widget plugin """

    def createWidget(self, parent):
        if False:
            return 10
        return TableView(parent)

    def icon(self):
        if False:
            while True:
                i = 10
        return super().icon('DataGrid')

    def name(self):
        if False:
            i = 10
            return i + 15
        return 'TableView'

class TreeWidgetPlugin(ViewPlugin, QPyDesignerCustomWidgetPlugin):
    """ Tree widget plugin """

    def createWidget(self, parent):
        if False:
            for i in range(10):
                print('nop')
        return TreeWidget(parent)

    def icon(self):
        if False:
            return 10
        return super().icon('TreeView')

    def name(self):
        if False:
            i = 10
            return i + 15
        return 'TreeWidget'

class TreeViewPlugin(ViewPlugin, QPyDesignerCustomWidgetPlugin):
    """ Tree view plugin """

    def createWidget(self, parent):
        if False:
            return 10
        return TreeView(parent)

    def icon(self):
        if False:
            for i in range(10):
                print('nop')
        return super().icon('TreeView')

    def name(self):
        if False:
            i = 10
            return i + 15
        return 'TreeView'

class HorizontalFlipViewPlugin(ViewPlugin, QPyDesignerCustomWidgetPlugin):
    """ Horizontal flip view plugin """

    def createWidget(self, parent):
        if False:
            for i in range(10):
                print('nop')
        return HorizontalFlipView(parent)

    def icon(self):
        if False:
            i = 10
            return i + 15
        return super().icon('FlipView')

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'HorizontalFlipView'

class VerticalFlipViewPlugin(ViewPlugin, QPyDesignerCustomWidgetPlugin):
    """ Vertical flip view plugin """

    def createWidget(self, parent):
        if False:
            for i in range(10):
                print('nop')
        return VerticalFlipView(parent)

    def icon(self):
        if False:
            return 10
        return super().icon('FlipView')

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'VerticalFlipView'

class HorizontalPipsPagerPlugin(ViewPlugin, QPyDesignerCustomWidgetPlugin):
    """ Horizontal flip view plugin """

    def createWidget(self, parent):
        if False:
            while True:
                i = 10
        w = HorizontalPipsPager(parent)
        w.setPageNumber(5)
        return w

    def icon(self):
        if False:
            while True:
                i = 10
        return super().icon('PipsPager')

    def name(self):
        if False:
            i = 10
            return i + 15
        return 'HorizontalPipsPager'

class VerticalPipsPagerPlugin(ViewPlugin, QPyDesignerCustomWidgetPlugin):
    """ Vertical flip view plugin """

    def createWidget(self, parent):
        if False:
            i = 10
            return i + 15
        w = VerticalPipsPager(parent)
        w.setPageNumber(5)
        return w

    def icon(self):
        if False:
            print('Hello World!')
        return super().icon('PipsPager')

    def name(self):
        if False:
            i = 10
            return i + 15
        return 'VerticalPipsPager'