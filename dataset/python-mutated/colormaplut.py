from ... import colormap
from ...widgets.ColorMapButton import ColorMapButton
from .basetypes import Parameter, WidgetParameterItem

class ColorMapLutParameterItem(WidgetParameterItem):

    def makeWidget(self):
        if False:
            return 10
        w = ColorMapButton()
        w.sigChanged = w.sigColorMapChanged
        w.value = w.colorMap
        w.setValue = w.setColorMap
        self.hideWidget = False
        return w

class ColorMapLutParameter(Parameter):
    itemClass = ColorMapLutParameterItem

    def _interpretValue(self, v):
        if False:
            return 10
        if isinstance(v, str):
            v = colormap.get(v)
        if v is not None and (not isinstance(v, colormap.ColorMap)):
            raise TypeError('Cannot set colormap parameter from object %r' % v)
        return v