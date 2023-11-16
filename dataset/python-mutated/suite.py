"""Module containing the Suite object, used for running a set of checks together."""
import abc
import io
import json
import pathlib
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Type, Union, cast
import jsonpickle
from bs4 import BeautifulSoup
from ipywidgets import Widget
from typing_extensions import Self, TypedDict
from deepchecks import __version__
from deepchecks.core import check_result as check_types
from deepchecks.core.checks import BaseCheck, CheckConfig
from deepchecks.core.display import DisplayableResult, save_as_html
from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError
from deepchecks.core.serialization.abc import HTMLFormatter
from deepchecks.core.serialization.suite_result.html import SuiteResultSerializer as SuiteResultHtmlSerializer
from deepchecks.core.serialization.suite_result.ipython import SuiteResultSerializer as SuiteResultIPythonSerializer
from deepchecks.core.serialization.suite_result.json import SuiteResultSerializer as SuiteResultJsonSerializer
from deepchecks.core.serialization.suite_result.widget import SuiteResultSerializer as SuiteResultWidgetSerializer
from deepchecks.utils.strings import get_random_string, widget_to_html_string
from deepchecks.utils.wandb_utils import wandb_run
from . import common
__all__ = ['BaseSuite', 'SuiteResult']

class SuiteConfig(TypedDict):
    module_name: str
    class_name: str
    version: str
    name: str
    checks: List['CheckConfig']

class SuiteResult(DisplayableResult):
    """Contain the results of a suite run.

    Parameters
    ----------
    name: str
    results: List[BaseCheckResult]
    extra_info: Optional[List[str]]
    """
    name: str
    extra_info: List[str]
    results: List['check_types.BaseCheckResult']

    def __init__(self, name: str, results: List['check_types.BaseCheckResult'], extra_info: Optional[List[str]]=None):
        if False:
            while True:
                i = 10
        'Initialize suite result.'
        self.name = name
        self.results = sort_check_results(results)
        self.extra_info = extra_info or []
        self.results_with_conditions: Set[int] = set()
        self.results_without_conditions: Set[int] = set()
        self.results_with_display: Set[int] = set()
        self.results_without_display: Set[int] = set()
        self.failures: Set[int] = set()
        for (index, result) in enumerate(self.results):
            if isinstance(result, check_types.CheckFailure):
                self.failures.add(index)
            elif isinstance(result, check_types.CheckResult):
                has_conditions = result.have_conditions()
                has_display = result.have_display()
                if has_conditions:
                    self.results_with_conditions.add(index)
                else:
                    self.results_without_conditions.add(index)
                if has_display:
                    self.results_with_display.add(index)
                else:
                    self.results_without_display.add(index)
            else:
                raise TypeError(f'Unknown type of result - {type(result).__name__}')

    def select_results(self, idx: Set[int]=None, names: Set[str]=None) -> List[Union['check_types.CheckResult', 'check_types.CheckFailure']]:
        if False:
            while True:
                i = 10
        "Select results either by indexes or result header names.\n\n        Parameters\n        ----------\n        idx : Set[int], default None\n            The list of indexes to filter the check results from the results list. If\n            names is None, then this parameter is required.\n        names : Set[str], default None\n            The list of names denoting the header of the check results. If idx is None,\n            this parameter is required. Both idx and names cannot be passed.\n\n        Returns\n        -------\n        List[Union['check_types.CheckResult', 'check_types.CheckFailure']] :\n            A list of check results filtered either by the indexes or by their names.\n        "
        if idx is None and names is None:
            raise DeepchecksNotSupportedError('Either idx or names should be passed')
        if idx and names:
            raise DeepchecksNotSupportedError('Only one of idx or names should be passed')
        if names:
            names = [name.lower().replace('_', ' ').strip() for name in names]
            output = [result for name in names for result in self.results if result.get_header().lower() == name]
        else:
            output = [result for (index, result) in enumerate(self.results) if index in idx]
        return output

    def __repr__(self):
        if False:
            print('Hello World!')
        'Return default __repr__ function uses value.'
        return self.name

    def _repr_html_(self, unique_id: Optional[str]=None, requirejs: bool=False) -> str:
        if False:
            print('Hello World!')
        'Return html representation of check result.'
        return widget_to_html_string(self.to_widget(unique_id=unique_id or get_random_string(n=25)), title=self.name, requirejs=requirejs)

    def _repr_json_(self):
        if False:
            return 10
        return SuiteResultJsonSerializer(self).serialize()

    def _repr_mimebundle_(self, **kwargs):
        if False:
            print('Hello World!')
        return {'text/html': self._repr_html_(), 'application/json': self._repr_json_()}

    @property
    def widget_serializer(self) -> SuiteResultWidgetSerializer:
        if False:
            i = 10
            return i + 15
        'Return WidgetSerializer instance.'
        return SuiteResultWidgetSerializer(self)

    @property
    def ipython_serializer(self) -> SuiteResultIPythonSerializer:
        if False:
            print('Hello World!')
        'Return IPythonSerializer instance.'
        return SuiteResultIPythonSerializer(self)

    @property
    def html_serializer(self) -> SuiteResultHtmlSerializer:
        if False:
            for i in range(10):
                print('nop')
        'Return HtmlSerializer instance.'
        return SuiteResultHtmlSerializer(self)

    def show(self, as_widget: bool=True, unique_id: Optional[str]=None, **kwargs) -> Optional[HTMLFormatter]:
        if False:
            while True:
                i = 10
        'Display result.\n\n        Parameters\n        ----------\n        as_widget : bool\n            whether to display result with help of ipywidgets or not\n        unique_id : Optional[str], default None\n            unique identifier of the result output\n        **kwrgs :\n            other key-value arguments will be passed to the `Serializer.serialize`\n            method\n\n        Returns\n        -------\n        Optional[HTMLFormatter] :\n            when used by sphinx-gallery\n        '
        return super().show(as_widget, unique_id or get_random_string(n=25), **kwargs)

    def show_not_interactive(self, unique_id: Optional[str]=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Display the not interactive version of result output.\n\n        In this case, ipywidgets will not be used and plotly\n        figures will be transformed into png images.\n\n        Parameters\n        ----------\n        unique_id : Optional[str], default None\n            unique identifier of the result output\n        **kwrgs :\n            other key-value arguments will be passed to the `Serializer.serialize`\n            method\n        '
        return super().show_not_interactive(unique_id or get_random_string(n=25), **kwargs)

    def save_as_html(self, file: Union[str, io.TextIOWrapper, None]=None, as_widget: bool=True, requirejs: bool=True, unique_id: Optional[str]=None, connected: bool=False, **kwargs):
        if False:
            while True:
                i = 10
        "Save output as html file.\n\n        Parameters\n        ----------\n        file : filename or file-like object\n            The file to write the HTML output to. If None writes to output.html\n        as_widget : bool, default True\n            whether to use ipywidgets or not\n        requirejs: bool , default: True\n            whether to include requirejs library into output HTML or not\n        unique_id : Optional[str], default None\n            unique identifier of the result output\n        connected: bool , default False\n            indicates whether internet connection is available or not,\n            if 'True' then CDN urls will be used to load javascript otherwise\n            javascript libraries will be injected directly into HTML output.\n            Set to 'False' to make results viewing possible when the internet\n            connection is not available.\n\n        Returns\n        -------\n        Optional[str] :\n            name of newly create file\n        "
        return save_as_html(file=file, serializer=self.widget_serializer if as_widget else self.html_serializer, connected=connected, requirejs=requirejs, output_id=unique_id or get_random_string(n=25))

    def save_as_cml_markdown(self, file: str=None, platform: str='github', attach_html_report: bool=True):
        if False:
            return 10
        "Save a result to a markdown file to use with [CML](https://cml.dev).\n\n        The rendered markdown will include only the conditions summary,\n        with the full html results attached.\n\n        Parameters\n        ----------\n        file : filename or file-like object\n            The file to write the HTML output to. If None writes to report.md\n        platform: str , default: 'github'\n            Target Git platform to ensure pretty formatting and nothing funky.\n            Options currently include 'github' or 'gitlab'.\n        attach_html_report: bool , default True\n            Whether to attach the full html report with plots, making it available\n            for download. This will save a [suite_name].html file\n            in the same directory as the markdown report.\n\n        Returns\n        -------\n        Optional[str] :\n            name of newly create file.\n        "
        if file is None:
            file = './report.md'
        elif isinstance(file, str):
            pass
        elif isinstance(file, io.TextIOWrapper):
            raise NotImplementedError('io.TextIOWrapper is not yet supported for save_as_cml_markdown.')

        def format_conditions_table():
            if False:
                print('Hello World!')
            conditions_table = SuiteResultHtmlSerializer(self).prepare_conditions_table()
            soup = BeautifulSoup(conditions_table, features='html.parser')
            soup.h2.extract()
            soup.style.extract()
            summary = soup.new_tag('summary')
            summary.string = self.name
            soup.table.insert_before(summary)
            soup = BeautifulSoup(f'\n<details>{str(soup)}</details>\n', features='html.parser')
            return soup
        soup = format_conditions_table()
        if not attach_html_report:
            with open(file, 'w', encoding='utf-8') as handle:
                handle.write(soup.prettify())
        else:
            path = pathlib.Path(file)
            html_file = str(pathlib.Path(file).parent.resolve().joinpath(path.stem + '.html'))
            self.save_as_html(html_file)
            if platform == 'gitlab':
                soup.summary.string = f'![{soup.summary.string}]({html_file})'
                soup = soup.prettify()
            elif platform == 'github':
                soup = soup.prettify() + f'\n> 📎 ![Full {self.name} Report]({html_file})\n'
            else:
                error_message = "Only 'github' and 'gitlab' are supported right now."
                error_message += '\nThough one of these formats '
                error_message += 'might work for your target Git platform!'
                raise ValueError(error_message)
            with open(file, 'w', encoding='utf-8') as file_handle:
                file_handle.write(soup)

    def to_widget(self, unique_id: Optional[str]=None, **kwargs) -> Widget:
        if False:
            return 10
        'Return SuiteResult as a ipywidgets.Widget instance.\n\n        Parameters\n        ----------\n        unique_id : Optional[str], default None\n            unique identifier of the result output\n\n        Returns\n        -------\n        Widget\n        '
        output_id = unique_id or get_random_string(n=25)
        return SuiteResultWidgetSerializer(self).serialize(output_id=output_id)

    def to_json(self, with_display: bool=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Return check result as json.\n\n        Parameters\n        ----------\n        with_display : bool, default True\n            whether to include serialized `SuiteResult.display` items into\n            the output or not\n\n        Returns\n        -------\n        str\n        '
        return jsonpickle.dumps(SuiteResultJsonSerializer(self).serialize(with_display=with_display), unpicklable=False)

    def to_wandb(self, dedicated_run: Optional[bool]=None, **kwargs):
        if False:
            while True:
                i = 10
        'Send suite result to wandb.\n\n        Parameters\n        ----------\n        dedicated_run : bool\n            whether to create a separate wandb run or not\n            (deprecated parameter, does not have any effect anymore)\n        kwargs: Keyword arguments to pass to wandb.init.\n                Default project name is deepchecks.\n                Default config is the suite name.\n        '
        from deepchecks.core.serialization.suite_result.wandb import SuiteResultSerializer as WandbSerializer
        if dedicated_run is not None:
            warnings.warn('"dedicated_run" parameter is deprecated and does not have effect anymore. It will be remove in next versions.')
        wandb_kwargs = {'config': {'name': self.name}}
        wandb_kwargs.update(**kwargs)
        with wandb_run(**wandb_kwargs) as run:
            run.log(WandbSerializer(self).serialize())

    def get_not_ran_checks(self) -> List['check_types.CheckFailure']:
        if False:
            return 10
        'Get all the check results which did not run (unable to run due to missing parameters, exception, etc).\n\n        Returns\n        -------\n        List[CheckFailure]\n            All the check failures in the suite.\n        '
        return cast(List[check_types.CheckFailure], self.select_results(self.failures))

    def get_not_passed_checks(self, fail_if_warning=True) -> List['check_types.CheckResult']:
        if False:
            while True:
                i = 10
        'Get all the check results that have not passing condition.\n\n        Parameters\n        ----------\n        fail_if_warning: bool, Default: True\n            Whether conditions should fail on status of warning\n\n        Returns\n        -------\n        List[CheckResult]\n            All the check results in the suite that have failing conditions.\n        '
        results = cast(List[check_types.CheckResult], self.select_results(self.results_with_conditions))
        return [r for r in results if not r.passed_conditions(fail_if_warning)]

    def get_passed_checks(self, fail_if_warning=True) -> List['check_types.CheckResult']:
        if False:
            while True:
                i = 10
        'Get all the check results that have passing condition.\n\n        Parameters\n        ----------\n        fail_if_warning: bool, Default: True\n            Whether conditions should fail on status of warning\n\n        Returns\n        -------\n        List[CheckResult]\n            All the check results in the suite that have failing conditions.\n        '
        results = cast(List[check_types.CheckResult], self.select_results(self.results_with_conditions))
        return [r for r in results if r.passed_conditions(fail_if_warning)]

    def passed(self, fail_if_warning: bool=True, fail_if_check_not_run: bool=False) -> bool:
        if False:
            i = 10
            return i + 15
        "Return whether this suite result has passed. Pass value is derived from condition results of all individual         checks, and may consider checks that didn't run.\n\n        Parameters\n        ----------\n        fail_if_warning: bool, Default: True\n            Whether conditions should fail on status of warning\n        fail_if_check_not_run: bool, Default: False\n            Whether checks that didn't run (missing parameters, exception, etc) should fail the suite result.\n\n        Returns\n        -------\n        bool\n        "
        not_run_pass = len(self.get_not_ran_checks()) == 0 if fail_if_check_not_run else True
        conditions_pass = len(self.get_not_passed_checks(fail_if_warning)) == 0
        return conditions_pass and not_run_pass

    @classmethod
    def from_json(cls, json_res: str):
        if False:
            print('Hello World!')
        'Convert a json object that was returned from SuiteResult.to_json.\n\n        Parameters\n        ----------\n        json_data: Union[str, Dict]\n            Json data\n\n        Returns\n        -------\n        SuiteResult\n            A suite result object.\n        '
        json_dict = jsonpickle.loads(json_res)
        name = json_dict['name']
        results = []
        for res in json_dict['results']:
            results.append(check_types.BaseCheckResult.from_json(res))
        return SuiteResult(name, results)

class BaseSuite:
    """Class for running a set of checks together, and returning a unified pass / no-pass.

    Parameters
    ----------
    checks: OrderedDict
        A list of checks to run.
    name: str
        Name of the suite
    """

    @classmethod
    @abc.abstractmethod
    def supported_checks(cls) -> Tuple:
        if False:
            i = 10
            return i + 15
        'Return list of of supported check types.'
        pass
    checks: 'OrderedDict[int, BaseCheck]'
    name: str
    _check_index: int

    def __init__(self, name: str, *checks: Union[BaseCheck, 'BaseSuite']):
        if False:
            print('Hello World!')
        self.name = name
        self.checks = OrderedDict()
        self._check_index = 0
        for check in checks:
            self.add(check)

    def __repr__(self, tabs=0):
        if False:
            print('Hello World!')
        'Representation of suite as string.'
        tabs_str = '\t' * tabs
        checks_str = ''.join([f"\n{c.__repr__(tabs + 1, str(n) + ': ')}" for (n, c) in self.checks.items()])
        return f'{tabs_str}{self.name}: [{checks_str}\n{tabs_str}]'

    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        'Access check inside the suite by name.'
        if index not in self.checks:
            raise DeepchecksValueError(f'No index {index} in suite')
        return self.checks[index]

    def add(self, check: Union['BaseCheck', 'BaseSuite']):
        if False:
            while True:
                i = 10
        'Add a check or a suite to current suite.\n\n        Parameters\n        ----------\n        check : BaseCheck\n            A check or suite to add.\n        '
        if isinstance(check, BaseSuite):
            if check is self:
                return self
            for c in check.checks.values():
                self.add(c)
        elif not isinstance(check, self.supported_checks()):
            raise DeepchecksValueError(f'Suite received unsupported object type: {check.__class__.__name__}')
        else:
            self.checks[self._check_index] = check
            self._check_index += 1
        return self

    def remove(self, index: int):
        if False:
            print('Hello World!')
        'Remove a check by given index.\n\n        Parameters\n        ----------\n        index : int\n            Index of check to remove.\n        '
        if index not in self.checks:
            raise DeepchecksValueError(f'No index {index} in suite')
        self.checks.pop(index)
        return self

    def to_json(self, indent: int=3) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Serialize suite instance to JSON string.'
        conf = self.config()
        return json.dumps(conf, indent=indent)

    def from_json(self, conf: str, version_unmatch: 'common.VersionUnmatchAction'='warn') -> Self:
        if False:
            return 10
        'Deserialize suite instance from JSON string.'
        suite_conf = json.loads(conf)
        return self.from_config(suite_conf, version_unmatch=version_unmatch)

    def config(self) -> SuiteConfig:
        if False:
            while True:
                i = 10
        "Return suite configuration (checks' conditions' configuration not yet supported).\n\n        Returns\n        -------\n        SuiteConfig\n            includes the suite name, and list of check configs.\n        "
        checks = [it.config(include_version=False) for it in self.checks.values()]
        (module_name, class_name) = common.importable_name(self)
        return SuiteConfig(module_name=module_name, class_name=class_name, name=self.name, version=__version__, checks=checks)

    @classmethod
    def from_config(cls: Type[Self], conf: SuiteConfig, version_unmatch: 'common.VersionUnmatchAction'='warn') -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Return suite object from a CheckConfig object.\n\n        Parameters\n        ----------\n        conf : SuiteConfig\n            the SuiteConfig object\n\n        Returns\n        -------\n        BaseSuite\n            the suite class object from given config\n        '
        suite_conf = cast(Dict[str, Any], conf)
        suite_conf = common.validate_config(suite_conf, version_unmatch)
        if 'checks' not in suite_conf or not isinstance(suite_conf['checks'], list):
            raise ValueError('Configuration must contain "checks" key of type list')
        if 'name' not in suite_conf or not isinstance(suite_conf['name'], str):
            raise ValueError('Configuration must contain "name" key of type string')
        suite_type = common.import_type(module_name=suite_conf['module_name'], type_name=suite_conf['class_name'], base=cls)
        checks = [BaseCheck.from_config(check_conf, version_unmatch=None) for check_conf in suite_conf['checks']]
        return suite_type(suite_conf['name'], *checks)

    @classmethod
    def _get_unsupported_failure(cls, check, msg):
        if False:
            i = 10
            return i + 15
        return check_types.CheckFailure(check, DeepchecksNotSupportedError(msg))

def sort_check_results(check_results: Sequence['check_types.BaseCheckResult']) -> List['check_types.BaseCheckResult']:
    if False:
        for i in range(10):
            print('nop')
    "Sort sequence of 'CheckResult' instances.\n\n    Returns\n    -------\n    List[check_types.CheckResult]\n    "
    order = []
    check_results_index = {}
    for (index, it) in enumerate(check_results):
        check_results_index[index] = it
        if isinstance(it, check_types.CheckResult):
            order.append((it.priority, index))
        elif isinstance(it, check_types.CheckFailure):
            order.append((998, index))
        else:
            order.append((999, index))
    order = sorted(order)
    return [check_results_index[index] for (_, index) in order]