"""Split namespace for argparse to allow separating options by prefix.

We use this to direct some options to an Options object and some to a
regular namespace.
"""
from __future__ import annotations
import argparse
from typing import Any

class SplitNamespace(argparse.Namespace):

    def __init__(self, standard_namespace: object, alt_namespace: object, alt_prefix: str) -> None:
        if False:
            return 10
        self.__dict__['_standard_namespace'] = standard_namespace
        self.__dict__['_alt_namespace'] = alt_namespace
        self.__dict__['_alt_prefix'] = alt_prefix

    def _get(self) -> tuple[Any, Any]:
        if False:
            while True:
                i = 10
        return (self._standard_namespace, self._alt_namespace)

    def __setattr__(self, name: str, value: Any) -> None:
        if False:
            return 10
        if name.startswith(self._alt_prefix):
            setattr(self._alt_namespace, name[len(self._alt_prefix):], value)
        else:
            setattr(self._standard_namespace, name, value)

    def __getattr__(self, name: str) -> Any:
        if False:
            while True:
                i = 10
        if name.startswith(self._alt_prefix):
            return getattr(self._alt_namespace, name[len(self._alt_prefix):])
        else:
            return getattr(self._standard_namespace, name)