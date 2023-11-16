from IPython.core.display import display
from ipywidgets import Output, widgets
from ydata_profiling.report.presentation.core.duplicate import Duplicate

class WidgetDuplicate(Duplicate):

    def render(self) -> widgets.VBox:
        if False:
            for i in range(10):
                print('nop')
        out = Output()
        with out:
            display(self.content['duplicate'])
        name = widgets.HTML(f"<h4>{self.content['name']}</h4>")
        return widgets.VBox([name, out])