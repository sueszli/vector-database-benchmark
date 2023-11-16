from pathlib import Path
from typing import List, Union
from django.template.loaders import app_directories
from typing_extensions import override

class TwoFactorLoader(app_directories.Loader):

    @override
    def get_dirs(self) -> List[Union[str, Path]]:
        if False:
            print('Hello World!')
        dirs = super().get_dirs()
        two_factor_dirs: List[Union[str, Path]] = []
        for d in dirs:
            assert isinstance(d, Path)
            if d.match('two_factor/*'):
                two_factor_dirs.append(d)
        return two_factor_dirs