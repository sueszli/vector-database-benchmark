from ipywidgets import widgets
from ydata_profiling.report.presentation.core import Variable

class WidgetVariable(Variable):

    def render(self) -> widgets.VBox:
        if False:
            i = 10
            return i + 15
        items = [self.content['top'].render()]
        if self.content['bottom'] is not None:
            items.append(self.content['bottom'].render())
        return widgets.VBox(items)