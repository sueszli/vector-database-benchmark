from __future__ import annotations
import contextlib
from typing import TYPE_CHECKING
from polars.utils.deprecation import issue_deprecation_warning
with contextlib.suppress(ImportError):
    import polars.polars as plr
    from polars.polars import PyStringCacheHolder
if TYPE_CHECKING:
    from types import TracebackType

class StringCache(contextlib.ContextDecorator):
    """
    Context manager for enabling and disabling the global string cache.

    :class:`Categorical` columns created under the same global string cache have
    the same underlying physical value when string values are equal. This allows the
    columns to be concatenated or used in a join operation, for example.

    Notes
    -----
    Enabling the global string cache introduces some overhead.
    The amount of overhead depends on the number of categories in your data.
    It is advised to enable the global string cache only when strictly necessary.

    If `StringCache` calls are nested, the global string cache will only be disabled
    and cleared when the outermost context exits.

    Examples
    --------
    Construct two Series using the same global string cache.

    >>> with pl.StringCache():
    ...     s1 = pl.Series("color", ["red", "green", "red"], dtype=pl.Categorical)
    ...     s2 = pl.Series("color", ["blue", "red", "green"], dtype=pl.Categorical)
    ...

    As both Series are constructed under the same global string cache,
    they can be concatenated.

    >>> pl.concat([s1, s2])
    shape: (6,)
    Series: 'color' [cat]
    [
            "red"
            "green"
            "red"
            "blue"
            "red"
            "green"
    ]

    The class can also be used as a function decorator, in which case the string cache
    is enabled during function execution, and disabled afterwards.

    >>> @pl.StringCache()
    ... def construct_categoricals() -> pl.Series:
    ...     s1 = pl.Series("color", ["red", "green", "red"], dtype=pl.Categorical)
    ...     s2 = pl.Series("color", ["blue", "red", "green"], dtype=pl.Categorical)
    ...     return pl.concat([s1, s2])
    ...

    """

    def __enter__(self) -> StringCache:
        if False:
            i = 10
            return i + 15
        self._string_cache = PyStringCacheHolder()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None:
        if False:
            return 10
        del self._string_cache

def enable_string_cache(enable: bool | None=None) -> None:
    if False:
        while True:
            i = 10
    '\n    Enable the global string cache.\n\n    :class:`Categorical` columns created under the same global string cache have\n    the same underlying physical value when string values are equal. This allows the\n    columns to be concatenated or used in a join operation, for example.\n\n    Parameters\n    ----------\n    enable\n        Enable or disable the global string cache.\n\n        .. deprecated:: 0.19.3\n            `enable_string_cache` no longer accepts an argument.\n             Call `enable_string_cache()` to enable the string cache\n             and `disable_string_cache()` to disable the string cache.\n\n    See Also\n    --------\n    StringCache : Context manager for enabling and disabling the string cache.\n    disable_string_cache : Function to disable the string cache.\n\n    Notes\n    -----\n    Enabling the global string cache introduces some overhead.\n    The amount of overhead depends on the number of categories in your data.\n    It is advised to enable the global string cache only when strictly necessary.\n\n    Consider using the :class:`StringCache` context manager for a more reliable way of\n    enabling and disabling the string cache.\n\n    Examples\n    --------\n    Construct two Series using the same global string cache.\n\n    >>> pl.enable_string_cache()\n    >>> s1 = pl.Series("color", ["red", "green", "red"], dtype=pl.Categorical)\n    >>> s2 = pl.Series("color", ["blue", "red", "green"], dtype=pl.Categorical)\n    >>> pl.disable_string_cache()\n\n    As both Series are constructed under the same global string cache,\n    they can be concatenated.\n\n    >>> pl.concat([s1, s2])\n    shape: (6,)\n    Series: \'color\' [cat]\n    [\n            "red"\n            "green"\n            "red"\n            "blue"\n            "red"\n            "green"\n    ]\n\n    '
    if enable is not None:
        issue_deprecation_warning('`enable_string_cache` no longer accepts an argument. Call `enable_string_cache()` to enable the string cache and `disable_string_cache()` to disable the string cache.', version='0.19.3')
        if enable is False:
            plr.disable_string_cache()
            return
    plr.enable_string_cache()

def disable_string_cache() -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Disable and clear the global string cache.\n\n    See Also\n    --------\n    enable_string_cache : Function to enable the string cache.\n    StringCache : Context manager for enabling and disabling the string cache.\n\n    Notes\n    -----\n    Consider using the :class:`StringCache` context manager for a more reliable way of\n    enabling and disabling the string cache.\n\n    When used in conjunction with the :class:`StringCache` context manager, the string\n    cache will not be disabled until the context manager exits.\n\n    Examples\n    --------\n    Construct two Series using the same global string cache.\n\n    >>> pl.enable_string_cache()\n    >>> s1 = pl.Series("color", ["red", "green", "red"], dtype=pl.Categorical)\n    >>> s2 = pl.Series("color", ["blue", "red", "green"], dtype=pl.Categorical)\n    >>> pl.disable_string_cache()\n\n    As both Series are constructed under the same global string cache,\n    they can be concatenated.\n\n    >>> pl.concat([s1, s2])\n    shape: (6,)\n    Series: \'color\' [cat]\n    [\n            "red"\n            "green"\n            "red"\n            "blue"\n            "red"\n            "green"\n    ]\n\n    '
    return plr.disable_string_cache()

def using_string_cache() -> bool:
    if False:
        return 10
    'Check whether the global string cache is enabled.'
    return plr.using_string_cache()