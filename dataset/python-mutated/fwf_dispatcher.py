"""Module houses `FWFDispatcher` class, that is used for reading of tables with fixed-width formatted lines."""
from typing import Optional, Sequence, Tuple, Union
from modin.core.io.text.text_file_dispatcher import TextFileDispatcher

class FWFDispatcher(TextFileDispatcher):
    """Class handles utils for reading of tables with fixed-width formatted lines."""

    @classmethod
    def check_parameters_support(cls, filepath_or_buffer, read_kwargs: dict, skiprows_md: Union[Sequence, callable, int], header_size: int) -> Tuple[bool, Optional[str]]:
        if False:
            print('Hello World!')
        '\n        Check support of parameters of `read_fwf` function.\n\n        Parameters\n        ----------\n        filepath_or_buffer : str, path object or file-like object\n            `filepath_or_buffer` parameter of `read_fwf` function.\n        read_kwargs : dict\n            Parameters of `read_fwf` function.\n        skiprows_md : int, array or callable\n            `skiprows` parameter modified for easier handling by Modin.\n        header_size : int\n            Number of rows that are used by header.\n\n        Returns\n        -------\n        bool\n            Whether passed parameters are supported or not.\n        Optional[str]\n            `None` if parameters are supported, otherwise an error\n            message describing why parameters are not supported.\n        '
        if read_kwargs['infer_nrows'] > 100:
            return (False, '`infer_nrows` is a significant portion of the number of rows, so Pandas may be faster')
        return super().check_parameters_support(filepath_or_buffer, read_kwargs, skiprows_md, header_size)