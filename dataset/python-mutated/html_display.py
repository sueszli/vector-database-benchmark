"""Module for html displayable result."""
import io
import typing as t
from ipywidgets import HTML, VBox, Widget
from deepchecks.core.display import DisplayableResult
from deepchecks.core.serialization.abc import HtmlSerializer, IPythonSerializer, WidgetSerializer
from deepchecks.core.serialization.common import normalize_widget_style
from deepchecks.utils.strings import create_new_file_name

class HtmlDisplayableResult(DisplayableResult):
    """Class which accepts html string and support displaying it in different environments."""

    def __init__(self, html: str):
        if False:
            i = 10
            return i + 15
        self.html = html

    @property
    def widget_serializer(self) -> WidgetSerializer[t.Any]:
        if False:
            for i in range(10):
                print('nop')
        'Return widget serializer.'

        class _WidgetSerializer(WidgetSerializer[t.Any]):

            def serialize(self, **kwargs) -> Widget:
                if False:
                    i = 10
                    return i + 15
                return normalize_widget_style(VBox(children=[HTML(self.value)]))
        return _WidgetSerializer(self.html)

    @property
    def ipython_serializer(self) -> IPythonSerializer[t.Any]:
        if False:
            return 10
        'Return IPython serializer.'

        class _IPythonSerializer(IPythonSerializer[t.Any]):

            def serialize(self, **kwargs) -> t.Any:
                if False:
                    for i in range(10):
                        print('nop')
                return HTML(self.value)
        return _IPythonSerializer(self.html)

    @property
    def html_serializer(self) -> HtmlSerializer[t.Any]:
        if False:
            print('Hello World!')
        'Return HTML serializer.'

        class _HtmlSerializer(HtmlSerializer[t.Any]):

            def serialize(self, **kwargs) -> str:
                if False:
                    while True:
                        i = 10
                return self.value
        return _HtmlSerializer(self.html)

    def to_widget(self, **kwargs) -> Widget:
        if False:
            i = 10
            return i + 15
        'Return the widget representation of the result.'
        return self.widget_serializer.serialize(**kwargs)

    def to_json(self, **kwargs):
        if False:
            i = 10
            return i + 15
        'Not implemented.'
        raise NotImplementedError()

    def to_wandb(self, **kwargs):
        if False:
            while True:
                i = 10
        'Not implemented.'
        raise NotImplementedError()

    def save_as_html(self, file: t.Union[str, io.TextIOWrapper, None]=None, **kwargs) -> t.Optional[str]:
        if False:
            i = 10
            return i + 15
        'Save the html to a file.'
        if file is None:
            file = 'output.html'
        if isinstance(file, str):
            file = create_new_file_name(file)
        if isinstance(file, str):
            with open(file, 'w', encoding='utf-8') as f:
                f.write(self.html)
        elif isinstance(file, io.TextIOWrapper):
            file.write(self.html)
        return file