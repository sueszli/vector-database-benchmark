from __future__ import annotations
from pathlib import Path
from typing import Optional
from pydantic import BaseSettings, Extra, root_validator

class TriblerConfigSection(BaseSettings):
    """Base Class that defines Tribler Config Section

    We are waiting https://github.com/samuelcolvin/pydantic/pull/2625
    for proper and native manipulations with relative and absolute paths.
    """

    class Config:
        extra = Extra.ignore

    def put_path_as_relative(self, property_name: str, value: Path=None, state_dir: str=None):
        if False:
            i = 10
            return i + 15
        "Save a relative path if 'value' is relative to state_dir.\n        Save an absolute path otherwise.\n        "
        if value is not None:
            try:
                value = Path(value).relative_to(state_dir)
            except ValueError:
                pass
            value = str(value)
        self.__setattr__(property_name, value)

    def get_path_as_absolute(self, property_name: str, state_dir: Path=None) -> Optional[Path]:
        if False:
            return 10
        ' Get path as absolute. If stored value already in absolute form, then it will be returned in "as is".\n           `state_dir / path` will be returned otherwise.\n        '
        value = self.__getattribute__(property_name)
        if value is None:
            return None
        return state_dir / value

    @root_validator(pre=True)
    def convert_from_none_string_to_none_type(cls, values):
        if False:
            while True:
                i = 10
        'After a convert operation from "ini" to "pydantic", None values\n        becomes \'None\' string values.\n\n        So, we have to convert them from `None` to None which is what happens\n        in this function\n        '
        for (key, value) in values.items():
            if value == 'None':
                values[key] = None
        return values