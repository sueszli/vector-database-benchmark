from ipywidgets import widgets
from ydata_profiling.report.presentation.core.root import Root

class WidgetRoot(Root):

    def render(self, **kwargs) -> widgets.VBox:
        if False:
            i = 10
            return i + 15
        return widgets.VBox([self.content['body'].render(), self.content['footer'].render()])