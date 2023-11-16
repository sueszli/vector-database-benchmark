from typing import Optional
from pip._internal.models.format_control import FormatControl

class SelectionPreferences:
    """
    Encapsulates the candidate selection preferences for downloading
    and installing files.
    """
    __slots__ = ['allow_yanked', 'allow_all_prereleases', 'format_control', 'prefer_binary', 'ignore_requires_python']

    def __init__(self, allow_yanked: bool, allow_all_prereleases: bool=False, format_control: Optional[FormatControl]=None, prefer_binary: bool=False, ignore_requires_python: Optional[bool]=None) -> None:
        if False:
            return 10
        'Create a SelectionPreferences object.\n\n        :param allow_yanked: Whether files marked as yanked (in the sense\n            of PEP 592) are permitted to be candidates for install.\n        :param format_control: A FormatControl object or None. Used to control\n            the selection of source packages / binary packages when consulting\n            the index and links.\n        :param prefer_binary: Whether to prefer an old, but valid, binary\n            dist over a new source dist.\n        :param ignore_requires_python: Whether to ignore incompatible\n            "Requires-Python" values in links. Defaults to False.\n        '
        if ignore_requires_python is None:
            ignore_requires_python = False
        self.allow_yanked = allow_yanked
        self.allow_all_prereleases = allow_all_prereleases
        self.format_control = format_control
        self.prefer_binary = prefer_binary
        self.ignore_requires_python = ignore_requires_python