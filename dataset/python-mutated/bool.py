from ...Qt import QtWidgets
from .basetypes import WidgetParameterItem

class BoolParameterItem(WidgetParameterItem):
    """
    Registered parameter type which displays a QCheckBox
    """

    def makeWidget(self):
        if False:
            for i in range(10):
                print('nop')
        w = QtWidgets.QCheckBox()
        w.sigChanged = w.toggled
        w.value = w.isChecked
        w.setValue = w.setChecked
        self.hideWidget = False
        return w