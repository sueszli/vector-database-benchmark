"""
Utilities for table pretty printing using click
"""
import shutil
import textwrap
from functools import wraps
from itertools import count, zip_longest
from typing import Sized
import click
MIN_OFFSET = 20

def pprint_column_names(format_string, format_kwargs, margin=None, table_header=None, color='yellow', display_sleep=False):
    if False:
        for i in range(10):
            print('nop')
    "\n\n    :param format_string: format string to be used that has the strings, minimum width to be replaced\n    :param format_kwargs: dictionary that is supplied to the format_string to format the string\n    :param margin: margin that is to be reduced from column width for columnar text.\n    :param table_header: Supplied table header\n    :param color: color supplied for table headers and column names.\n    :param display_sleep: flag to format table_header to include deployer's client_sleep\n    :return: boilerplate table string\n    "
    min_width = 100
    min_margin = 2

    def pprint_wrap(func):
        if False:
            for i in range(10):
                print('nop')
        (width, _) = shutil.get_terminal_size()
        width = max(width, min_width)
        total_args = len(format_kwargs)
        if not total_args:
            raise ValueError('Number of arguments supplied should be > 0 , format_kwargs: {}'.format(format_kwargs))
        width = width - width % total_args
        usable_width_no_margin = int(width) - 1
        usable_width = int(usable_width_no_margin - (margin if margin else min_margin))
        if total_args > int(usable_width / 2):
            raise ValueError('Total number of columns exceed available width')
        width_per_column = int(usable_width / total_args)
        final_arg_width = width_per_column - 1
        format_args = [width_per_column for _ in range(total_args - 1)]
        format_args.extend([final_arg_width])

        @wraps(func)
        def wrap(*args, **kwargs):
            if False:
                while True:
                    i = 10
            if table_header:
                click.secho('\n' + table_header.format(args[0].client_sleep) if display_sleep else table_header, bold=True)
            click.secho('-' * usable_width, fg=color)
            click.secho(format_string.format(*format_args, **format_kwargs), fg=color)
            click.secho('-' * usable_width, fg=color)
            kwargs['format_args'] = format_args
            kwargs['width'] = width_per_column
            kwargs['margin'] = margin if margin else min_margin
            result = func(*args, **kwargs)
            click.secho('-' * usable_width + '\n', fg=color)
            return result
        return wrap
    return pprint_wrap

def wrapped_text_generator(texts, width, margin, **textwrap_kwargs):
    if False:
        while True:
            i = 10
    '\n\n    Return a generator where the contents are wrapped text to a specified width.\n\n    :param texts: list of text that needs to be wrapped at specified width\n    :param width: width of the text to be wrapped\n    :param margin: margin to be reduced from width for cleaner UX\n    :param textwrap_kwargs: keyword arguments that are passed to textwrap.wrap\n    :return: generator of wrapped text\n    :rtype: Iterator[str]\n    '
    for text in texts:
        yield textwrap.wrap(text, width=width - margin, **textwrap_kwargs)

def pprint_columns(columns, width, margin, format_string, format_args, columns_dict, color='yellow', **textwrap_kwargs):
    if False:
        return 10
    '\n\n    Print columns based on list of columnar text, associated formatting string and associated format arguments.\n\n    :param columns: list of columnnar text that go into columns as specified by the format_string\n    :param width: width of the text to be wrapped\n    :param margin: margin to be reduced from width for cleaner UX\n    :param format_string: A format string that has both width and text specifiers set.\n    :param format_args: list of offset specifiers\n    :param columns_dict: arguments dictionary that have dummy values per column\n    :param color: color supplied for rows within the table.\n    :param textwrap_kwargs: keyword arguments that are passed to textwrap.wrap\n    '
    for columns_text in zip_longest(*wrapped_text_generator(columns, width, margin, **textwrap_kwargs), fillvalue=''):
        counter = count()
        for (k, _) in columns_dict.items():
            columns_dict[k] = columns_text[next(counter)]
        click.secho(format_string.format(*format_args, **columns_dict), fg=color)

def newline_per_item(iterable: Sized, counter: int) -> None:
    if False:
        while True:
            i = 10
    '\n    Adds a new line based on the index of a given iterable\n    Parameters\n    ----------\n    iterable: Any iterable that implements __len__\n    counter: Current index within the iterable\n\n    Returns\n    -------\n\n    '
    if counter < len(iterable) - 1:
        click.echo(message='', nl=True)