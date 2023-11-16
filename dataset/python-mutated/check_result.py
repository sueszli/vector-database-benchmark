"""Module containing the check results classes."""
import io
import traceback
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union, cast
import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pd
import pandas as pd
from ipywidgets import Widget
from pandas.io.formats.style import Styler
from plotly.basedatatypes import BaseFigure
from deepchecks.core.condition import ConditionCategory, ConditionResult
from deepchecks.core.display import DisplayableResult, save_as_html
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.core.reduce_classes import ReduceMixin
from deepchecks.core.serialization.abc import HTMLFormatter
from deepchecks.core.serialization.check_failure.html import CheckFailureSerializer as CheckFailureHtmlSerializer
from deepchecks.core.serialization.check_failure.ipython import CheckFailureSerializer as CheckFailureIPythonSerializer
from deepchecks.core.serialization.check_failure.json import CheckFailureSerializer as CheckFailureJsonSerializer
from deepchecks.core.serialization.check_failure.widget import CheckFailureSerializer as CheckFailureWidgetSerializer
from deepchecks.core.serialization.check_result.html import CheckResultSection
from deepchecks.core.serialization.check_result.html import CheckResultSerializer as CheckResultHtmlSerializer
from deepchecks.core.serialization.check_result.ipython import CheckResultSerializer as CheckResultIPythonSerializer
from deepchecks.core.serialization.check_result.json import CheckResultSerializer as CheckResultJsonSerializer
from deepchecks.core.serialization.check_result.widget import CheckResultSerializer as CheckResultWidgetSerializer
from deepchecks.utils.strings import widget_to_html_string
from deepchecks.utils.wandb_utils import wandb_run
jsonpickle_pd.register_handlers()
if TYPE_CHECKING:
    from deepchecks.core.checks import BaseCheck
__all__ = ['CheckResult', 'CheckFailure', 'BaseCheckResult', 'DisplayMap']

class DisplayMap(Dict[str, List['TDisplayItem']]):
    """Class facilitating tabs within check display output."""
    pass
TDisplayCallable = Callable[[], None]
TDisplayItem = Union[str, pd.DataFrame, Styler, BaseFigure, TDisplayCallable, DisplayMap]

class BaseCheckResult:
    """Generic class for any check output, contains some basic functions."""
    check: Optional['BaseCheck']
    header: Optional[str]
    run_time: Optional[int] = 0

    @staticmethod
    def from_json(json_dict: Union[str, Dict]) -> 'BaseCheckResult':
        if False:
            while True:
                i = 10
        'Convert a json object that was returned from CheckResult.to_json or CheckFailure.to_json.\n\n        Parameters\n        ----------\n        json_dict: Union[str, Dict]\n            Json data\n\n        Returns\n        -------\n        BaseCheckResult\n            A check output object.\n        '
        from deepchecks.core.check_json import CheckFailureJson, CheckResultJson
        if isinstance(json_dict, str):
            json_dict = jsonpickle.loads(json_dict)
        check_type = cast(dict, json_dict)['type']
        if check_type == 'CheckFailure':
            return CheckFailureJson(json_dict)
        elif check_type == 'CheckResult':
            return CheckResultJson(json_dict)
        else:
            raise ValueError(f'Excpected json object to be one of [CheckFailure, CheckResult] but recievied: {check_type}')

    def get_header(self) -> str:
        if False:
            while True:
                i = 10
        'Return header for display. if header was defined return it, else extract name of check class.'
        return self.header or self.check.name()

    def get_metadata(self, with_doc_link: bool=False) -> Dict:
        if False:
            print('Hello World!')
        'Return the related check metadata.'
        return {'header': self.get_header(), **self.check.metadata(with_doc_link=with_doc_link)}

    def get_check_id(self, unique_id: str='') -> str:
        if False:
            while True:
                i = 10
        'Return check id (used for href).'
        header = self.get_header().replace(' ', '')
        return f'{header}_{unique_id}'

class CheckResult(BaseCheckResult, DisplayableResult):
    """Class which returns from a check with result that can later be used for automatic pipelines and display value.

    Class containing the result of a check

    The class stores the results and display of the check. Evaluating the result in an IPython console / notebook
    will show the result display output.

    Parameters
    ----------
    value : Any
        Value calculated by check. Can be used to decide if decidable check passed.
    display : List[Union[Callable, str, pd.DataFrame, Styler, BaseFigure]] , default: None
        Dictionary with formatters for display. possible formatters are: 'text/html', 'image/png'
    header : str , default: None
        Header to be displayed in python notebook.
    """
    value: Any
    header: Optional[str]
    display: List[TDisplayItem]
    conditions_results: List[ConditionResult]

    def __init__(self, value, header: Optional[str]=None, display: Optional[List[TDisplayItem]]=None):
        if False:
            while True:
                i = 10
        self.value = value
        self.header = header
        self.conditions_results = []
        if display is not None and (not isinstance(display, List)):
            self.display = [display]
        else:
            self.display = display or []
        for item in self.display:
            if not isinstance(item, (str, pd.DataFrame, Styler, Callable, BaseFigure, DisplayMap)):
                raise DeepchecksValueError(f"Can't display item of type: {type(item)}")

    def process_conditions(self):
        if False:
            return 10
        'Process the conditions results from current result and check.'
        self.conditions_results = self.check.conditions_decision(self)

    def have_conditions(self) -> bool:
        if False:
            return 10
        'Return if this check has condition results.'
        return bool(self.conditions_results)

    def have_display(self) -> bool:
        if False:
            return 10
        'Return if this check has display.'
        return bool(self.display)

    def passed_conditions(self, fail_if_warning=True) -> bool:
        if False:
            print('Hello World!')
        'Return if this check has no passing condition results.'
        return all((r.is_pass(fail_if_warning) for r in self.conditions_results))

    @property
    def priority(self) -> int:
        if False:
            i = 10
            return i + 15
        "Return priority of the current result.\n\n        This value is primarly used to determine suite output order.\n        The logic is next:\n\n        * if at least one condition did not pass and is of category 'FAIL', return 1.\n        * if at least one condition did not pass and is of category 'WARN', return 2.\n        * if at least one condition did not pass and is of category 'ERROR', return 3.\n        * if all conditions passed, return 4.\n        * if check result do not have assigned conditions, return 5.\n\n        Returns\n        -------\n        int\n            priority of the check result.\n        "
        if not self.have_conditions():
            return 5
        for c in self.conditions_results:
            if c.is_pass() is False and c.category == ConditionCategory.FAIL:
                return 1
            if c.is_pass() is False and c.category == ConditionCategory.WARN:
                return 2
            if c.is_pass() is False and c.category == ConditionCategory.ERROR:
                return 3
        return 4

    def reduce_output(self) -> Dict[str, float]:
        if False:
            i = 10
            return i + 15
        'Return the check result as a reduced dict.'
        if isinstance(self.check, ReduceMixin):
            return self.check.reduce_output(self)
        raise DeepchecksValueError('Check needs to be an instance of ReduceMixin to use this function')

    @property
    def widget_serializer(self) -> CheckResultWidgetSerializer:
        if False:
            while True:
                i = 10
        'Return WidgetSerializer instance.'
        return CheckResultWidgetSerializer(self)

    @property
    def ipython_serializer(self) -> CheckResultIPythonSerializer:
        if False:
            print('Hello World!')
        'Return IPythonSerializer instance.'
        return CheckResultIPythonSerializer(self)

    @property
    def html_serializer(self) -> CheckResultHtmlSerializer:
        if False:
            for i in range(10):
                print('nop')
        'Return HtmlSerializer instance.'
        return CheckResultHtmlSerializer(self)

    def display_check(self, unique_id: Optional[str]=None, as_widget: bool=True, show_additional_outputs: bool=True, **kwargs):
        if False:
            print('Hello World!')
        'Display the check result or return the display as widget.\n\n        Parameters\n        ----------\n        unique_id : str\n            unique identifier of the result output\n        as_widget : bool\n            Boolean that controls if to display the check regulary or if to return a widget.\n        show_additional_outputs : bool\n            Boolean that controls if to show additional outputs.\n        '
        self.show(as_widget=as_widget, unique_id=unique_id, show_additional_outputs=show_additional_outputs)

    def save_as_html(self, file: Union[str, io.TextIOWrapper, None]=None, unique_id: Optional[str]=None, show_additional_outputs: bool=True, as_widget: bool=True, requirejs: bool=True, connected: bool=False, **kwargs):
        if False:
            return 10
        "Save a result to an HTML file.\n\n        Parameters\n        ----------\n        file : filename or file-like object\n            the file to write the HTML output to. If None writes to output.html\n        unique_id : Optional[str], default None\n            unique identifier of the result output\n        show_additional_outputs : bool, default True\n            whether to show additional outputs or not\n        as_widget : bool, default True\n            whether to use ipywidgets or not\n        requirejs: bool , default: True\n            whether to include requirejs library into output HTML or not\n        connected: bool , default False\n            indicates whether internet connection is available or not,\n            if 'True' then CDN urls will be used to load javascript otherwise\n            javascript libraries will be injected directly into HTML output.\n            Set to 'False' to make results viewing possible when the internet\n            connection is not available.\n\n        Returns\n        -------\n        Optional[str] :\n            name of newly create file\n        "
        return save_as_html(file=file, serializer=self.widget_serializer if as_widget else self.html_serializer, connected=connected, requirejs=requirejs, output_id=unique_id, check_sections=detalize_additional_output(show_additional_outputs))

    def show(self, as_widget: bool=True, unique_id: Optional[str]=None, show_additional_outputs: bool=True, **kwargs) -> Optional[HTMLFormatter]:
        if False:
            return 10
        'Display the check result.\n\n        Parameters\n        ----------\n        as_widget : bool, default True\n            whether to use ipywidgets or not\n        unique_id : Optional[str], default None\n            unique identifier of the result output\n        show_additional_outputs : bool, default True\n            whether to show additional outputs or not\n\n        Returns\n        -------\n        Optional[HTMLFormatter] :\n            when used by sphinx-gallery\n        '
        return super().show(as_widget=as_widget, unique_id=unique_id, check_sections=detalize_additional_output(show_additional_outputs), **kwargs)

    def to_widget(self, unique_id: Optional[str]=None, show_additional_outputs: bool=True, **kwargs) -> Widget:
        if False:
            for i in range(10):
                print('nop')
        'Return CheckResult as a ipywidgets.Widget instance.\n\n        Parameters\n        ----------\n        unique_id : Optional[str], default None\n            unique identifier of the result output\n        show_additional_outputs : bool, default True\n            whether to show additional outputs or not\n\n        Returns\n        -------\n        Widget\n        '
        return self.widget_serializer.serialize(output_id=unique_id, check_sections=detalize_additional_output(show_additional_outputs))

    def to_wandb(self, **kwargs):
        if False:
            while True:
                i = 10
        'Send result to wandb.\n\n        Parameters\n        ----------\n        kwargs: Keyword arguments to pass to wandb.init.\n                Default project name is deepchecks.\n                Default config is the check metadata (params, train/test/ name etc.).\n        '
        assert self.check is not None
        from .serialization.check_result.wandb import CheckResultSerializer as WandbSerializer
        wandb_kwargs = {'config': {'header': self.get_header(), **self.check.metadata()}}
        wandb_kwargs.update(**kwargs)
        with wandb_run(**wandb_kwargs) as run:
            run.log(WandbSerializer(self).serialize())

    def to_json(self, with_display: bool=True, **kwargs) -> str:
        if False:
            while True:
                i = 10
        'Serialize result into a json string.\n\n        Returned JSON string will have next structure:\n\n        >>    class CheckResultMetadata(TypedDict):\n        >>        type: str\n        >>        check: CheckMetadata\n        >>        value: Any\n        >>        header: str\n        >>        conditions_results: List[Dict[Any, Any]]\n        >>        display: List[Dict[str, Any]]\n\n        >>    class CheckMetadata(TypedDict):\n        >>        name: str\n        >>        params: Dict[Any, Any]\n        >>        summary: str\n\n        Parameters\n        ----------\n        with_display : bool\n            whethere to include display items or not\n\n        Returns\n        -------\n        str\n        '
        return jsonpickle.dumps(CheckResultJsonSerializer(self).serialize(with_display=with_display), unpicklable=False)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        'Return default __repr__ function uses value.'
        return f'{self.get_header()}: {self.value}'

    def _repr_html_(self, unique_id: Optional[str]=None, show_additional_outputs: bool=True, requirejs: bool=False, **kwargs) -> str:
        if False:
            return 10
        'Return html representation of check result.'
        return widget_to_html_string(self.to_widget(unique_id=unique_id, show_additional_outputs=show_additional_outputs), title=self.get_header(), requirejs=requirejs)

    def _repr_json_(self, **kwargs):
        if False:
            i = 10
            return i + 15
        return CheckResultJsonSerializer(self).serialize()

    def _repr_mimebundle_(self, **kwargs):
        if False:
            while True:
                i = 10
        return {'text/html': self._repr_html_(), 'application/json': self._repr_json_()}

    def _ipython_display_(self, unique_id: Optional[str]=None, as_widget: bool=True, show_additional_outputs: bool=True):
        if False:
            for i in range(10):
                print('nop')
        self.show(unique_id=unique_id, as_widget=as_widget, show_additional_outputs=show_additional_outputs)

class CheckFailure(BaseCheckResult, DisplayableResult):
    """Class which holds a check run exception.

    Parameters
    ----------
    check : BaseCheck
    exception : Exception
    header_suffix : str , default ``

    """

    def __init__(self, check: 'BaseCheck', exception: Exception, header_suffix: str=''):
        if False:
            return 10
        self.check = check
        self.exception = exception
        self.header = check.name() + header_suffix

    @property
    def widget_serializer(self) -> CheckFailureWidgetSerializer:
        if False:
            return 10
        'Return WidgetSerializer instance.'
        return CheckFailureWidgetSerializer(self)

    @property
    def ipython_serializer(self) -> CheckFailureIPythonSerializer:
        if False:
            i = 10
            return i + 15
        'Return IPythonSerializer instance.'
        return CheckFailureIPythonSerializer(self)

    @property
    def html_serializer(self) -> CheckFailureHtmlSerializer:
        if False:
            print('Hello World!')
        'Return HtmlSerializer instance.'
        return CheckFailureHtmlSerializer(self)

    def display_check(self, as_widget: bool=True, **kwargs):
        if False:
            return 10
        'Display the check failure or return the display as widget.\n\n        Parameters\n        ----------\n        as_widget : bool, default True\n            whether to use ipywidgets or not\n        '
        self.show(as_widget=as_widget)

    def save_as_html(self, file: Union[str, io.TextIOWrapper, None]=None, as_widget: bool=True, requirejs: bool=True, connected: bool=False, **kwargs) -> Optional[str]:
        if False:
            print('Hello World!')
        "Save output as html file.\n\n        Parameters\n        ----------\n        file : filename or file-like object\n            The file to write the HTML output to. If None writes to output.html\n        as_widget : bool, default True\n            whether to use ipywidgets or not\n        requirejs: bool , default: True\n            whether to include requirejs library into output HTML or not\n        connected: bool , default False\n            indicates whether internet connection is available or not,\n            if 'True' then CDN urls will be used to load javascript otherwise\n            javascript libraries will be injected directly into HTML output.\n            Set to 'False' to make results viewing possible when the internet\n            connection is not available.\n\n        Returns\n        -------\n        Optional[str] :\n            name of newly create file\n        "
        return save_as_html(file=file, serializer=self.widget_serializer if as_widget else self.html_serializer, connected=connected, requirejs=requirejs)

    def to_widget(self, **kwargs) -> Widget:
        if False:
            print('Hello World!')
        'Return CheckFailure as a ipywidgets.Widget instance.'
        return CheckFailureWidgetSerializer(self).serialize()

    def to_json(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Serialize CheckFailure into a json string.\n\n        Returned JSON string will have next structure:\n\n        >>    class CheckFailureMetadata(TypedDict):\n        >>        check: CheckMetadata\n        >>        header: str\n        >>        display: List[Dict[str, str]]\n\n        >>    class CheckMetadata(TypedDict):\n        >>        type: str\n        >>        name: str\n        >>        params: Dict[Any, Any]\n        >>        summary: str\n\n        Returns\n        -------\n        str\n        '
        return jsonpickle.dumps(CheckFailureJsonSerializer(self).serialize(), unpicklable=False)

    def to_wandb(self, **kwargs):
        if False:
            i = 10
            return i + 15
        'Send check result to wandb.\n\n        Parameters\n        ----------\n        kwargs: Keyword arguments to pass to wandb.init.\n                Default project name is deepchecks.\n                Default config is the check metadata (params, train/test/ name etc.).\n        '
        assert self.check is not None
        from .serialization.check_failure.wandb import CheckFailureSerializer as WandbSerializer
        wandb_kwargs = {'config': {'header': self.get_header(), **self.check.metadata()}}
        wandb_kwargs.update(**kwargs)
        with wandb_run(**wandb_kwargs) as run:
            run.log(WandbSerializer(self).serialize())

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return string representation.'
        return self.get_header() + ': ' + str(self.exception)

    def _repr_html_(self):
        if False:
            print('Hello World!')
        return CheckFailureHtmlSerializer(self).serialize()

    def _repr_json_(self):
        if False:
            return 10
        return CheckFailureJsonSerializer(self).serialize()

    def _repr_mimebundle_(self, **kwargs):
        if False:
            print('Hello World!')
        return {'text/html': self._repr_html_(), 'application/json': self._repr_json_()}

    def print_traceback(self):
        if False:
            i = 10
            return i + 15
        'Print the traceback of the failure.'
        print(''.join(traceback.format_exception(type(self.exception), value=self.exception, tb=self.exception.__traceback__)))

def detalize_additional_output(show_additional_outputs: bool) -> List[CheckResultSection]:
    if False:
        print('Hello World!')
    return ['condition-table', 'additional-output'] if show_additional_outputs else ['condition-table']