"""Helper functions to get location of header files."""
import pathlib
from typing import Union

def _boost_dir(ret_path: bool=False) -> Union[pathlib.Path, str]:
    if False:
        for i in range(10):
            print('nop')
    'Directory where root Boost/ directory lives.'
    p = pathlib.Path(__file__).parent / 'boost_math/include'
    return p if ret_path else str(p)