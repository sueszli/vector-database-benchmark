from PyQt5.QtDesigner import QPyDesignerCustomWidgetPlugin
from qfluentwidgets import PrimaryPushButton, SplitPushButton, DropDownPushButton, ToolButton, SplitToolButton, DropDownToolButton, FluentIcon, ToggleButton, SwitchButton, RadioButton, CheckBox, HyperlinkButton, Slider, ComboBox, IconWidget, EditableComboBox, PixmapLabel, PushButton, PrimaryToolButton, PrimarySplitToolButton, PrimarySplitPushButton, PrimaryDropDownPushButton, PrimaryDropDownToolButton, TransparentToolButton, TransparentPushButton, ToggleToolButton, TransparentToggleToolButton, TransparentTogglePushButton, TransparentDropDownPushButton, TransparentDropDownToolButton, PillPushButton, PillToolButton, HorizontalSeparator, VerticalSeparator
from plugin_base import PluginBase
from task_menu_factory import EditTextTaskMenuFactory

class BasicInputPlugin(PluginBase):

    def group(self):
        if False:
            for i in range(10):
                print('nop')
        return super().group() + ' (Basic Input)'

class TextPlugin(BasicInputPlugin):

    def domXml(self):
        if False:
            i = 10
            return i + 15
        return f'\n        <widget class="{self.name()}" name="{self.name()}">\n            <property name="text">\n                <string>{self.toolTip()}</string>\n            </property>\n        </widget>\n        '

class CheckBoxPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Check box plugin """

    def createWidget(self, parent):
        if False:
            return 10
        return CheckBox(self.toolTip(), parent)

    def icon(self):
        if False:
            while True:
                i = 10
        return super().icon('Checkbox')

    def name(self):
        if False:
            i = 10
            return i + 15
        return 'CheckBox'

class ComboBoxPlugin(BasicInputPlugin, QPyDesignerCustomWidgetPlugin):
    """ Combo box plugin """

    def createWidget(self, parent):
        if False:
            i = 10
            return i + 15
        return ComboBox(parent)

    def icon(self):
        if False:
            for i in range(10):
                print('nop')
        return super().icon('ComboBox')

    def name(self):
        if False:
            i = 10
            return i + 15
        return 'ComboBox'

class EditableComboBoxPlugin(BasicInputPlugin, QPyDesignerCustomWidgetPlugin):
    """ Editable box plugin """

    def createWidget(self, parent):
        if False:
            while True:
                i = 10
        return EditableComboBox(parent)

    def icon(self):
        if False:
            return 10
        return super().icon('ComboBox')

    def name(self):
        if False:
            return 10
        return 'EditableComboBox'

class HyperlinkButtonPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Hyperlink button plugin """

    def createWidget(self, parent):
        if False:
            print('Hello World!')
        return HyperlinkButton('', self.toolTip(), parent)

    def icon(self):
        if False:
            while True:
                i = 10
        return super().icon('HyperlinkButton')

    def name(self):
        if False:
            while True:
                i = 10
        return 'HyperlinkButton'

class PushButtonPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Push button plugin """

    def createWidget(self, parent):
        if False:
            i = 10
            return i + 15
        return PushButton(self.toolTip(), parent)

    def icon(self):
        if False:
            for i in range(10):
                print('nop')
        return super().icon('Button')

    def name(self):
        if False:
            i = 10
            return i + 15
        return 'PushButton'

class PrimaryPushButtonPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Primary push button plugin """

    def createWidget(self, parent):
        if False:
            while True:
                i = 10
        return PrimaryPushButton(self.toolTip(), parent)

    def icon(self):
        if False:
            print('Hello World!')
        return super().icon('Button')

    def name(self):
        if False:
            return 10
        return 'PrimaryPushButton'

class PillPushButtonPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Pill push button plugin """

    def createWidget(self, parent):
        if False:
            return 10
        return PillPushButton(self.toolTip(), parent)

    def icon(self):
        if False:
            i = 10
            return i + 15
        return super().icon('Button')

    def name(self):
        if False:
            return 10
        return 'PillPushButton'

class DropDownPushButtonPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Drop down push button plugin """

    def createWidget(self, parent):
        if False:
            i = 10
            return i + 15
        return DropDownPushButton(self.toolTip(), parent)

    def icon(self):
        if False:
            while True:
                i = 10
        return super().icon('DropDownButton')

    def name(self):
        if False:
            return 10
        return 'DropDownPushButton'

class PrimaryDropDownPushButtonPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Primary drop down push button plugin """

    def createWidget(self, parent):
        if False:
            return 10
        return PrimaryDropDownPushButton(self.toolTip(), parent)

    def icon(self):
        if False:
            for i in range(10):
                print('nop')
        return super().icon('DropDownButton')

    def name(self):
        if False:
            while True:
                i = 10
        return 'PrimaryDropDownPushButton'

@EditTextTaskMenuFactory.register
class SplitPushButtonPlugin(BasicInputPlugin, QPyDesignerCustomWidgetPlugin):
    """ Split push button plugin """

    def createWidget(self, parent):
        if False:
            i = 10
            return i + 15
        return SplitPushButton(self.toolTip(), parent)

    def icon(self):
        if False:
            while True:
                i = 10
        return super().icon('SplitButton')

    def name(self):
        if False:
            return 10
        return 'SplitPushButton'

    def domXml(self):
        if False:
            print('Hello World!')
        return f'\n        <widget class="{self.name()}" name="{self.name()}">\n            <property name="text_">\n                <string>{self.toolTip()}</string>\n            </property>\n        </widget>\n        '

@EditTextTaskMenuFactory.register
class PrimarySplitPushButtonPlugin(BasicInputPlugin, QPyDesignerCustomWidgetPlugin):
    """ Primary color split push button plugin """

    def createWidget(self, parent):
        if False:
            print('Hello World!')
        return PrimarySplitPushButton(self.toolTip(), parent)

    def icon(self):
        if False:
            for i in range(10):
                print('nop')
        return super().icon('SplitButton')

    def name(self):
        if False:
            return 10
        return 'PrimarySplitPushButton'

    def domXml(self):
        if False:
            while True:
                i = 10
        return f'\n        <widget class="{self.name()}" name="{self.name()}">\n            <property name="text_">\n                <string>{self.toolTip()}</string>\n            </property>\n        </widget>\n        '

class ToolButtonPlugin(BasicInputPlugin, QPyDesignerCustomWidgetPlugin):
    """ Tool button plugin """

    def createWidget(self, parent):
        if False:
            while True:
                i = 10
        return ToolButton(FluentIcon.BASKETBALL, parent)

    def icon(self):
        if False:
            return 10
        return super().icon('Button')

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'ToolButton'

class PrimaryToolButtonPlugin(BasicInputPlugin, QPyDesignerCustomWidgetPlugin):
    """ Primary color tool button plugin """

    def createWidget(self, parent):
        if False:
            return 10
        return PrimaryToolButton(FluentIcon.BASKETBALL, parent)

    def icon(self):
        if False:
            i = 10
            return i + 15
        return super().icon('Button')

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'PrimaryToolButton'

class PillToolButtonPlugin(BasicInputPlugin, QPyDesignerCustomWidgetPlugin):
    """ Pill tool button plugin """

    def createWidget(self, parent):
        if False:
            return 10
        return PillToolButton(FluentIcon.BASKETBALL, parent)

    def icon(self):
        if False:
            i = 10
            return i + 15
        return super().icon('Button')

    def name(self):
        if False:
            return 10
        return 'PillToolButton'

class TransparentToolButtonPlugin(BasicInputPlugin, QPyDesignerCustomWidgetPlugin):
    """ Primary color tool button plugin """

    def createWidget(self, parent):
        if False:
            return 10
        return TransparentToolButton(FluentIcon.BASKETBALL, parent)

    def icon(self):
        if False:
            i = 10
            return i + 15
        return super().icon('Button')

    def name(self):
        if False:
            return 10
        return 'TransparentToolButton'

class DropDownToolButtonPlugin(BasicInputPlugin, QPyDesignerCustomWidgetPlugin):
    """ Drop down tool button plugin """

    def createWidget(self, parent):
        if False:
            return 10
        return DropDownToolButton(FluentIcon.BASKETBALL, parent)

    def icon(self):
        if False:
            while True:
                i = 10
        return super().icon('DropDownButton')

    def name(self):
        if False:
            while True:
                i = 10
        return 'DropDownToolButton'

class PrimaryDropDownToolButtonPlugin(BasicInputPlugin, QPyDesignerCustomWidgetPlugin):
    """ Drop down tool button plugin """

    def createWidget(self, parent):
        if False:
            return 10
        return PrimaryDropDownToolButton(FluentIcon.BASKETBALL, parent)

    def icon(self):
        if False:
            print('Hello World!')
        return super().icon('DropDownButton')

    def name(self):
        if False:
            i = 10
            return i + 15
        return 'PrimaryDropDownToolButton'

class SplitToolButtonPlugin(BasicInputPlugin, QPyDesignerCustomWidgetPlugin):
    """ split tool button plugin """

    def createWidget(self, parent):
        if False:
            print('Hello World!')
        return SplitToolButton(FluentIcon.BASKETBALL, parent)

    def icon(self):
        if False:
            while True:
                i = 10
        return super().icon('SplitButton')

    def name(self):
        if False:
            return 10
        return 'SplitToolButton'

class PrimarySplitToolButtonPlugin(BasicInputPlugin, QPyDesignerCustomWidgetPlugin):
    """ Primary color split tool button plugin """

    def createWidget(self, parent):
        if False:
            while True:
                i = 10
        return PrimarySplitToolButton(FluentIcon.BASKETBALL, parent)

    def icon(self):
        if False:
            i = 10
            return i + 15
        return super().icon('SplitButton')

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'PrimarySplitToolButton'

@EditTextTaskMenuFactory.register
class SwitchButtonPlugin(BasicInputPlugin, QPyDesignerCustomWidgetPlugin):
    """ Switch button plugin """

    def createWidget(self, parent):
        if False:
            print('Hello World!')
        return SwitchButton(parent)

    def icon(self):
        if False:
            print('Hello World!')
        return super().icon('ToggleSwitch')

    def name(self):
        if False:
            print('Hello World!')
        return 'SwitchButton'

class RadioButtonPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Radio button plugin """

    def createWidget(self, parent):
        if False:
            while True:
                i = 10
        return RadioButton(self.toolTip(), parent)

    def icon(self):
        if False:
            i = 10
            return i + 15
        return super().icon('RadioButton')

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'RadioButton'

class ToggleButtonPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Toggle push button plugin """

    def createWidget(self, parent):
        if False:
            while True:
                i = 10
        return ToggleButton(self.toolTip(), parent)

    def icon(self):
        if False:
            while True:
                i = 10
        return super().icon('ToggleButton')

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'ToggleButton'

class ToggleToolButtonPlugin(BasicInputPlugin, QPyDesignerCustomWidgetPlugin):
    """ Toggle tool button plugin """

    def createWidget(self, parent):
        if False:
            while True:
                i = 10
        return ToggleToolButton(FluentIcon.BASKETBALL, parent)

    def icon(self):
        if False:
            i = 10
            return i + 15
        return super().icon('ToggleButton')

    def name(self):
        if False:
            while True:
                i = 10
        return 'ToggleToolButton'

class TransparentPushButtonPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Transparent push button plugin """

    def createWidget(self, parent):
        if False:
            while True:
                i = 10
        return TransparentPushButton(self.toolTip(), parent)

    def icon(self):
        if False:
            i = 10
            return i + 15
        return super().icon('Button')

    def name(self):
        if False:
            while True:
                i = 10
        return 'TransparentPushButton'

class TransparentTogglePushButtonPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Transparent toggle push button plugin """

    def createWidget(self, parent):
        if False:
            print('Hello World!')
        return TransparentTogglePushButton(self.toolTip(), parent)

    def icon(self):
        if False:
            i = 10
            return i + 15
        return super().icon('ToggleButton')

    def name(self):
        if False:
            while True:
                i = 10
        return 'TransparentTogglePushButton'

class TransparentDropDownPushButtonPlugin(TextPlugin, QPyDesignerCustomWidgetPlugin):
    """ Transparent drop down push button plugin """

    def createWidget(self, parent):
        if False:
            for i in range(10):
                print('nop')
        return TransparentDropDownPushButton(self.toolTip(), parent)

    def icon(self):
        if False:
            return 10
        return super().icon('DropDownButton')

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'TransparentDropDownPushButton'

class TransparentToggleToolButtonPlugin(BasicInputPlugin, QPyDesignerCustomWidgetPlugin):
    """ Transparent toggle tool button plugin """

    def createWidget(self, parent):
        if False:
            i = 10
            return i + 15
        return TransparentToggleToolButton(FluentIcon.BASKETBALL, parent)

    def icon(self):
        if False:
            print('Hello World!')
        return super().icon('ToggleButton')

    def name(self):
        if False:
            return 10
        return 'TransparentToggleToolButton'

class TransparentDropDownToolButtonPlugin(BasicInputPlugin, QPyDesignerCustomWidgetPlugin):
    """ Transparent drop down tool button plugin """

    def createWidget(self, parent):
        if False:
            for i in range(10):
                print('nop')
        return TransparentDropDownToolButton(FluentIcon.BASKETBALL, parent)

    def icon(self):
        if False:
            print('Hello World!')
        return super().icon('DropDownButton')

    def name(self):
        if False:
            while True:
                i = 10
        return 'TransparentDropDownToolButton'

class SliderPlugin(BasicInputPlugin, QPyDesignerCustomWidgetPlugin):
    """  Slider  plugin """

    def createWidget(self, parent):
        if False:
            return 10
        slider = Slider(parent)
        slider.setRange(0, 100)
        slider.setMinimumWidth(200)
        return slider

    def icon(self):
        if False:
            return 10
        return super().icon('Slider')

    def name(self):
        if False:
            i = 10
            return i + 15
        return 'Slider'

class IconWidgetPlugin(BasicInputPlugin, QPyDesignerCustomWidgetPlugin):
    """ Icon widget plugin """

    def createWidget(self, parent):
        if False:
            for i in range(10):
                print('nop')
        return IconWidget(FluentIcon.EMOJI_TAB_SYMBOLS, parent)

    def icon(self):
        if False:
            i = 10
            return i + 15
        return super().icon('IconElement')

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'IconWidget'

class PixmapLabelPlugin(BasicInputPlugin, QPyDesignerCustomWidgetPlugin):
    """ Pixmap label plugin """

    def createWidget(self, parent):
        if False:
            return 10
        return PixmapLabel(parent)

    def icon(self):
        if False:
            for i in range(10):
                print('nop')
        return super().icon('Image')

    def name(self):
        if False:
            return 10
        return 'PixmapLabel'

class HorizontalSeparatorPlugin(BasicInputPlugin, QPyDesignerCustomWidgetPlugin):
    """ Horizontal separator plugin """

    def createWidget(self, parent):
        if False:
            while True:
                i = 10
        return HorizontalSeparator(parent)

    def icon(self):
        if False:
            for i in range(10):
                print('nop')
        return super().icon('Line')

    def name(self):
        if False:
            return 10
        return 'HorizontalSeparator'

class VerticalSeparatorPlugin(BasicInputPlugin, QPyDesignerCustomWidgetPlugin):
    """ Vertical separator plugin """

    def createWidget(self, parent):
        if False:
            for i in range(10):
                print('nop')
        return VerticalSeparator(parent)

    def icon(self):
        if False:
            while True:
                i = 10
        return super().icon('VerticalLine')

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'VerticalSeparator'