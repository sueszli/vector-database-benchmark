from PyQt5.QtCore import Qt
from PyQt5.QtDesigner import QPyDesignerCustomWidgetPlugin
from qfluentwidgets import CommandBar, Action, FluentIcon
from plugin_base import PluginBase

class ToolBarPlugin(PluginBase):

    def group(self):
        if False:
            for i in range(10):
                print('nop')
        return super().group() + ' (ToolBar)'

class CommandBarPlugin(ToolBarPlugin, QPyDesignerCustomWidgetPlugin):
    """ Command bar plugin """

    def createWidget(self, parent):
        if False:
            i = 10
            return i + 15
        w = CommandBar(parent)
        w.addAction(Action(FluentIcon.SHARE, 'Share'))
        w.addAction(Action(FluentIcon.SAVE, 'Save'))
        w.addAction(Action(FluentIcon.DELETE, 'Delete'))
        return w

    def icon(self):
        if False:
            return 10
        return super().icon('CommandBar')

    def name(self):
        if False:
            return 10
        return 'CommandBar'