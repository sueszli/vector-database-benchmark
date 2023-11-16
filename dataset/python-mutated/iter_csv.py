from __future__ import annotations
import csv
import datetime as dt
import random
from .. import base
from . import utils
__all__ = ['iter_csv']

class DictReader(csv.DictReader):
    """Overlay on top of `csv.DictReader` which allows sampling."""

    def __init__(self, fraction, rng, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self.fraction = fraction
        self.rng = rng

    def __next__(self):
        if False:
            i = 10
            return i + 15
        if self.line_num == 0:
            self.fieldnames
        row = next(self.reader)
        if self.fraction < 1:
            while self.rng.random() > self.fraction:
                row = next(self.reader)
        return dict(zip(self.fieldnames, row))

def iter_csv(filepath_or_buffer, target: str | list[str] | None=None, converters: dict | None=None, parse_dates: dict | None=None, drop: list[str] | None=None, drop_nones=False, fraction=1.0, compression='infer', seed: int | None=None, field_size_limit: int | None=None, **kwargs) -> base.typing.Stream:
    if False:
        print('Hello World!')
    "Iterates over rows from a CSV file.\n\n    Reading CSV files can be quite slow. If, for whatever reason, you're going to loop through\n    the same file multiple times, then we recommend that you to use the `stream.Cache` utility.\n\n    Parameters\n    ----------\n    filepath_or_buffer\n        Either a string indicating the location of a file, or a buffer object that has a\n        `read` method.\n    target\n        A single target column is assumed if a string is passed. A multiple output scenario\n        is assumed if a list of strings is passed. A `None` value will be assigned to each `y`\n        if this parameter is omitted.\n    converters\n        All values in the CSV are interpreted as strings by default. You can use this parameter to\n        cast values to the desired type. This should be a `dict` mapping feature names to callables\n        used to parse their associated values. Note that a callable may be a type, such as `float`\n        and `int`.\n    parse_dates\n        A `dict` mapping feature names to a format passed to the `datetime.datetime.strptime`\n        method.\n    drop\n        Fields to ignore.\n    drop_nones\n        Whether or not to drop fields where the value is a `None`.\n    fraction\n        Sampling fraction.\n    compression\n        For on-the-fly decompression of on-disk data. If this is set to 'infer' and\n        `filepath_or_buffer` is a path, then the decompression method is inferred for the\n        following extensions: '.gz', '.zip'.\n    seed\n        If specified, the sampling will be deterministic.\n    field_size_limit\n        If not `None`, this will be passed to the `csv.field_size_limit` function.\n    kwargs\n        All other keyword arguments are passed to the underlying `csv.DictReader`.\n\n    Returns\n    -------\n\n    By default each feature value will be of type `str`. You can use the `converters` and\n    `parse_dates` parameters to convert them as you see fit.\n\n    Examples\n    --------\n\n    Although this function is designed to handle different kinds of inputs, the most common\n    use case is to read a file on the disk. We'll first create a little CSV file to illustrate.\n\n    >>> tv_shows = '''name,year,rating\n    ... Planet Earth II,2016,9.5\n    ... Planet Earth,2006,9.4\n    ... Band of Brothers,2001,9.4\n    ... Breaking Bad,2008,9.4\n    ... Chernobyl,2019,9.4\n    ... '''\n    >>> with open('tv_shows.csv', mode='w') as f:\n    ...     _ = f.write(tv_shows)\n\n    We can now go through the rows one by one. We can use the `converters` parameter to cast\n    the `rating` field value as a `float`. We can also convert the `year` to a `datetime` via\n    the `parse_dates` parameter.\n\n    >>> from river import stream\n\n    >>> params = {\n    ...     'converters': {'rating': float},\n    ...     'parse_dates': {'year': '%Y'}\n    ... }\n    >>> for x, y in stream.iter_csv('tv_shows.csv', **params):\n    ...     print(x, y)\n    {'name': 'Planet Earth II', 'year': datetime.datetime(2016, 1, 1, 0, 0), 'rating': 9.5} None\n    {'name': 'Planet Earth', 'year': datetime.datetime(2006, 1, 1, 0, 0), 'rating': 9.4} None\n    {'name': 'Band of Brothers', 'year': datetime.datetime(2001, 1, 1, 0, 0), 'rating': 9.4} None\n    {'name': 'Breaking Bad', 'year': datetime.datetime(2008, 1, 1, 0, 0), 'rating': 9.4} None\n    {'name': 'Chernobyl', 'year': datetime.datetime(2019, 1, 1, 0, 0), 'rating': 9.4} None\n\n    The value of `y` is always `None` because we haven't provided a value for the `target`\n    parameter. Here is an example where a `target` is provided:\n\n    >>> dataset = stream.iter_csv('tv_shows.csv', target='rating', **params)\n    >>> for x, y in dataset:\n    ...     print(x, y)\n    {'name': 'Planet Earth II', 'year': datetime.datetime(2016, 1, 1, 0, 0)} 9.5\n    {'name': 'Planet Earth', 'year': datetime.datetime(2006, 1, 1, 0, 0)} 9.4\n    {'name': 'Band of Brothers', 'year': datetime.datetime(2001, 1, 1, 0, 0)} 9.4\n    {'name': 'Breaking Bad', 'year': datetime.datetime(2008, 1, 1, 0, 0)} 9.4\n    {'name': 'Chernobyl', 'year': datetime.datetime(2019, 1, 1, 0, 0)} 9.4\n\n    Finally, let's delete the example file.\n\n    >>> import os; os.remove('tv_shows.csv')\n\n    "
    limit = csv.field_size_limit()
    if field_size_limit is not None:
        csv.field_size_limit(field_size_limit)
    buffer = filepath_or_buffer
    if not hasattr(buffer, 'read'):
        buffer = utils.open_filepath(buffer, compression)
    for x in DictReader(fraction=fraction, rng=random.Random(seed), f=buffer, **kwargs):
        if drop:
            for i in drop:
                del x[i]
        if converters is not None:
            for (i, t) in converters.items():
                x[i] = t(x[i])
        if drop_nones:
            for i in list(x):
                if x[i] is None:
                    del x[i]
        if parse_dates is not None:
            for (i, fmt) in parse_dates.items():
                x[i] = dt.datetime.strptime(x[i], fmt)
        y = None
        if isinstance(target, list):
            y = {name: x.pop(name) for name in target}
        elif target is not None:
            y = x.pop(target)
        yield (x, y)
    if buffer is not filepath_or_buffer:
        buffer.close()
    csv.field_size_limit(limit)