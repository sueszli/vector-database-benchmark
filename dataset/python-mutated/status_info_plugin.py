from PyQt5.QtCore import QSize, Qt
from PyQt5.QtDesigner import QPyDesignerCustomWidgetPlugin
from qfluentwidgets import InfoBar, ProgressBar, IndeterminateProgressBar, ProgressRing, StateToolTip, InfoBarPosition, IndeterminateProgressRing, InfoBadge, DotInfoBadge, IconInfoBadge, FluentIcon
from plugin_base import PluginBase

class StatusInfoPlugin(PluginBase):

    def group(self):
        if False:
            return 10
        return super().group() + ' (Status & Info)'

class InfoBarPlugin(StatusInfoPlugin, QPyDesignerCustomWidgetPlugin):
    """ Info bar plugin """

    def createWidget(self, parent):
        if False:
            return 10
        return InfoBar.success(title='Lesson 5', content='最短的捷径就是绕远路，绕远路才是我的最短捷径。', duration=-1, position=InfoBarPosition.NONE, parent=parent)

    def icon(self):
        if False:
            print('Hello World!')
        return super().icon('InfoBar')

    def name(self):
        if False:
            print('Hello World!')
        return 'InfoBar'

class ProgressBarPlugin(StatusInfoPlugin, QPyDesignerCustomWidgetPlugin):
    """ Progress bar plugin """

    def createWidget(self, parent):
        if False:
            for i in range(10):
                print('nop')
        return ProgressBar(parent)

    def icon(self):
        if False:
            return 10
        return super().icon('ProgressBar')

    def name(self):
        if False:
            return 10
        return 'ProgressBar'

class IndeterminateProgressBarPlugin(StatusInfoPlugin, QPyDesignerCustomWidgetPlugin):
    """ Indeterminate progress bar plugin """

    def createWidget(self, parent):
        if False:
            return 10
        return IndeterminateProgressBar(parent)

    def icon(self):
        if False:
            print('Hello World!')
        return super().icon('ProgressBar')

    def name(self):
        if False:
            while True:
                i = 10
        return 'IndeterminateProgressBar'

class ProgressRingPlugin(StatusInfoPlugin, QPyDesignerCustomWidgetPlugin):
    """ Progress ring plugin """

    def createWidget(self, parent):
        if False:
            while True:
                i = 10
        return ProgressRing(parent)

    def icon(self):
        if False:
            print('Hello World!')
        return super().icon('ProgressRing')

    def name(self):
        if False:
            return 10
        return 'ProgressRing'

class IndeterminateProgressRingPlugin(StatusInfoPlugin, QPyDesignerCustomWidgetPlugin):
    """ Progress ring plugin """

    def createWidget(self, parent):
        if False:
            i = 10
            return i + 15
        return IndeterminateProgressRing(parent)

    def icon(self):
        if False:
            i = 10
            return i + 15
        return super().icon('ProgressRing')

    def name(self):
        if False:
            return 10
        return 'IndeterminateProgressRing'

class StateToolTipPlugin(StatusInfoPlugin, QPyDesignerCustomWidgetPlugin):
    """ State tool tip plugin """

    def createWidget(self, parent):
        if False:
            print('Hello World!')
        return StateToolTip('Running', 'Please wait patiently', parent)

    def icon(self):
        if False:
            i = 10
            return i + 15
        return super().icon('ProgressRing')

    def name(self):
        if False:
            return 10
        return 'StateToolTip'

class InfoBadgePlugin(StatusInfoPlugin, QPyDesignerCustomWidgetPlugin):
    """ Info badge plugin """

    def createWidget(self, parent):
        if False:
            for i in range(10):
                print('nop')
        return InfoBadge('10', parent)

    def icon(self):
        if False:
            for i in range(10):
                print('nop')
        return super().icon('InfoBadge')

    def name(self):
        if False:
            print('Hello World!')
        return 'InfoBadge'

class DotInfoBadgePlugin(StatusInfoPlugin, QPyDesignerCustomWidgetPlugin):
    """ Dot info badge plugin """

    def createWidget(self, parent):
        if False:
            while True:
                i = 10
        return DotInfoBadge(parent)

    def icon(self):
        if False:
            for i in range(10):
                print('nop')
        return super().icon('InfoBadge')

    def name(self):
        if False:
            i = 10
            return i + 15
        return 'DotInfoBadge'

class IconInfoBadgePlugin(StatusInfoPlugin, QPyDesignerCustomWidgetPlugin):
    """ Icon info badge plugin """

    def createWidget(self, parent):
        if False:
            while True:
                i = 10
        return IconInfoBadge.success(FluentIcon.ACCEPT_MEDIUM, parent)

    def icon(self):
        if False:
            i = 10
            return i + 15
        return super().icon('InfoBadge')

    def name(self):
        if False:
            print('Hello World!')
        return 'IconInfoBadge'