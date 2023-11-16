"""This file add the console interface to the package."""
import argparse
from pathlib import Path
from typing import Any, List, Optional
from ydata_profiling.__init__ import ProfileReport, __version__
from ydata_profiling.utils.dataframe import read_pandas

def parse_args(args: Optional[List[Any]]=None) -> argparse.Namespace:
    if False:
        i = 10
        return i + 15
    'Parse the command line arguments for the `ydata_profiling` binary.\n\n    Args:\n      args: List of input arguments. (Default value=None).\n\n    Returns:\n      Namespace with parsed arguments.\n\n    '
    parser = argparse.ArgumentParser(description='Profile the variables in a CSV file and generate a HTML report.')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('-s', '--silent', help='Only generate but do not open report', action='store_true')
    parser.add_argument('-m', '--minimal', help='Minimal configuration for big data sets', action='store_true')
    parser.add_argument('-e', '--explorative', help='Explorative configuration featuring unicode, file and image analysis', action='store_true')
    parser.add_argument('--pool_size', type=int, default=0, help='Number of CPU cores to use')
    parser.add_argument('--title', type=str, default='Pandas Profiling Report', help='Title for the report')
    parser.add_argument('--infer_dtypes', default=False, action='store_true', help='To infer dtypes of the dataframe')
    parser.add_argument('--no-infer_dtypes', dest='infer_dtypes', action='store_false', help='To read dtypes as read by pandas')
    parser.add_argument('--config_file', type=str, default=None, help="Specify a yaml config file. Have a look at the 'config_default.yaml' as a starting point.")
    parser.add_argument('input_file', type=str, help='CSV file (or other file type supported by pandas) to profile')
    parser.add_argument('output_file', type=str, nargs='?', help="Output report file. If empty, replaces the input_file's extension with .html and uses that.", default=None)
    return parser.parse_args(args)

def main(args: Optional[List[Any]]=None) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Run the `ydata_profiling` package.\n\n    Args:\n      args: Arguments for the programme (Default value=None).\n    '
    parsed_args = parse_args(args)
    kwargs = vars(parsed_args)
    input_file = Path(kwargs.pop('input_file'))
    output_file = kwargs.pop('output_file')
    if output_file is None:
        output_file = str(input_file.with_suffix('.html'))
    silent = kwargs.pop('silent')
    df = read_pandas(input_file)
    p = ProfileReport(df, **kwargs)
    p.to_file(Path(output_file), silent=silent)