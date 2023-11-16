from __future__ import annotations
import inspect
import json
from typing import TYPE_CHECKING, Any, Callable, Sequence
from rich.box import ASCII_DOUBLE_HEAD
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from tabulate import tabulate
from airflow.plugins_manager import PluginsDirectorySource
from airflow.utils import yaml
from airflow.utils.platform import is_tty
if TYPE_CHECKING:
    from airflow.typing_compat import TypeGuard

def is_data_sequence(data: Sequence[dict | Any]) -> TypeGuard[Sequence[dict]]:
    if False:
        return 10
    return all((isinstance(d, dict) for d in data))

class AirflowConsole(Console):
    """Airflow rich console."""

    def __init__(self, show_header: bool=True, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self._width = 200 if not is_tty() else self._width
        self.show_header = show_header

    def print_as_json(self, data: dict):
        if False:
            print('Hello World!')
        'Render dict as json text representation.'
        json_content = json.dumps(data)
        self.print(Syntax(json_content, 'json', theme='ansi_dark'), soft_wrap=True)

    def print_as_yaml(self, data: dict):
        if False:
            for i in range(10):
                print('nop')
        'Render dict as yaml text representation.'
        yaml_content = yaml.dump(data)
        self.print(Syntax(yaml_content, 'yaml', theme='ansi_dark'), soft_wrap=True)

    def print_as_table(self, data: list[dict]):
        if False:
            return 10
        'Render list of dictionaries as table.'
        if not data:
            self.print('No data found')
            return
        table = SimpleTable(show_header=self.show_header)
        for col in data[0]:
            table.add_column(col)
        for row in data:
            table.add_row(*(str(d) for d in row.values()))
        self.print(table)

    def print_as_plain_table(self, data: list[dict]):
        if False:
            print('Hello World!')
        'Render list of dictionaries as a simple table than can be easily piped.'
        if not data:
            self.print('No data found')
            return
        rows = [d.values() for d in data]
        output = tabulate(rows, tablefmt='plain', headers=list(data[0]))
        print(output)

    def _normalize_data(self, value: Any, output: str) -> list | str | dict | None:
        if False:
            print('Hello World!')
        if isinstance(value, (tuple, list)):
            if output == 'table':
                return ','.join((str(self._normalize_data(x, output)) for x in value))
            return [self._normalize_data(x, output) for x in value]
        if isinstance(value, dict) and output != 'table':
            return {k: self._normalize_data(v, output) for (k, v) in value.items()}
        if inspect.isclass(value) and (not isinstance(value, PluginsDirectorySource)):
            return value.__name__
        if value is None:
            return None
        return str(value)

    def print_as(self, data: Sequence[dict | Any], output: str, mapper: Callable[[Any], dict] | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Print provided using format specified by output argument.'
        output_to_renderer: dict[str, Callable[[Any], None]] = {'json': self.print_as_json, 'yaml': self.print_as_yaml, 'table': self.print_as_table, 'plain': self.print_as_plain_table}
        renderer = output_to_renderer.get(output)
        if not renderer:
            raise ValueError(f'Unknown formatter: {output}. Allowed options: {list(output_to_renderer)}')
        if mapper:
            dict_data: Sequence[dict] = [mapper(d) for d in data]
        elif is_data_sequence(data):
            dict_data = data
        else:
            raise ValueError('To tabulate non-dictionary data you need to provide `mapper` function')
        dict_data = [{k: self._normalize_data(v, output) for (k, v) in d.items()} for d in dict_data]
        renderer(dict_data)

class SimpleTable(Table):
    """A rich Table with some default hardcoded for consistency."""

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self.show_edge = kwargs.get('show_edge', False)
        self.pad_edge = kwargs.get('pad_edge', False)
        self.box = kwargs.get('box', ASCII_DOUBLE_HEAD)
        self.show_header = kwargs.get('show_header', False)
        self.title_style = kwargs.get('title_style', 'bold green')
        self.title_justify = kwargs.get('title_justify', 'left')
        self.caption = kwargs.get('caption', ' ')

    def add_column(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        'Add a column to the table. We use different default.'
        kwargs['overflow'] = kwargs.get('overflow')
        super().add_column(*args, **kwargs)