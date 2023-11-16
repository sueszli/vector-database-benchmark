from ipywidgets import widgets
from ydata_profiling.report.presentation.core import VariableInfo
from ydata_profiling.report.presentation.flavours.html import templates

class WidgetVariableInfo(VariableInfo):

    def render(self) -> widgets.HTML:
        if False:
            for i in range(10):
                print('nop')
        return widgets.HTML(templates.template('variable_info.html').render(**self.content))