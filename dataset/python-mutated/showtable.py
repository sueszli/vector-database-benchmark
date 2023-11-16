"""
``showtable`` is a command-line script based on ``astropy.io`` and
``astropy.table`` for printing ASCII, FITS, HDF5 or VOTable files(s) to the
standard output.

Example usage of ``showtable``:

1. FITS::

    $ showtable astropy/io/fits/tests/data/table.fits

     target V_mag
    ------- -----
    NGC1001  11.1
    NGC1002  12.3
    NGC1003  15.2

2. ASCII::

    $ showtable astropy/io/ascii/tests/t/simple_csv.csv

     a   b   c
    --- --- ---
      1   2   3
      4   5   6

3. XML::

    $ showtable astropy/io/votable/tests/data/names.xml --max-width 70

               col1             col2     col3  ... col15 col16 col17
               ---              deg      deg   ...  mag   mag   ---
    ------------------------- -------- ------- ... ----- ----- -----
    SSTGLMC G000.0000+00.1611   0.0000  0.1611 ...    --    --    AA



4. Print all the FITS tables in the current directory::

    $ showtable *.fits

"""
import argparse
import textwrap
import warnings
from astropy import log
from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning

def showtable(filename, args):
    if False:
        while True:
            i = 10
    '\n    Read a table and print to the standard output.\n\n    Parameters\n    ----------\n    filename : str\n        The path to a FITS file.\n\n    '
    if args.info and args.stats:
        warnings.warn('--info and --stats cannot be used together', AstropyUserWarning)
    if any((args.max_lines, args.max_width, args.hide_unit, args.show_dtype)) and (args.info or args.stats):
        warnings.warn('print parameters are ignored if --info or --stats is used', AstropyUserWarning)
    read_kwargs = ('hdu', 'format', 'table_id', 'delimiter')
    kwargs = {k: v for (k, v) in vars(args).items() if k in read_kwargs and v is not None}
    try:
        table = Table.read(filename, **kwargs)
        if args.info:
            table.info('attributes')
        elif args.stats:
            table.info('stats')
        else:
            formatter = table.more if args.more else table.pprint
            formatter(max_lines=args.max_lines, max_width=args.max_width, show_unit=False if args.hide_unit else None, show_dtype=True if args.show_dtype else None)
    except OSError as e:
        log.error(str(e))

def main(args=None):
    if False:
        print('Hello World!')
    'The main function called by the `showtable` script.'
    parser = argparse.ArgumentParser(description=textwrap.dedent("\n            Print tables from ASCII, FITS, HDF5, VOTable file(s).  The tables\n            are read with 'astropy.table.Table.read' and are printed with\n            'astropy.table.Table.pprint'. The default behavior is to make the\n            table output fit onto a single screen page.  For a long and wide\n            table this will mean cutting out inner rows and columns.  To print\n            **all** the rows or columns use ``--max-lines=-1`` or\n            ``max-width=-1``, respectively. The complete list of supported\n            formats can be found at\n            http://astropy.readthedocs.io/en/latest/io/unified.html#built-in-table-readers-writers\n        "))
    addarg = parser.add_argument
    addarg('filename', nargs='+', help='path to one or more files')
    addarg('--format', help='input table format, should be specified if it cannot be automatically detected')
    addarg('--more', action='store_true', help='use the pager mode from Table.more')
    addarg('--info', action='store_true', help='show information about the table columns')
    addarg('--stats', action='store_true', help='show statistics about the table columns')
    pprint_args = parser.add_argument_group('pprint arguments')
    addarg = pprint_args.add_argument
    addarg('--max-lines', type=int, help='maximum number of lines in table output (default=screen length, -1 for no limit)')
    addarg('--max-width', type=int, help='maximum width in table output (default=screen width, -1 for no limit)')
    addarg('--hide-unit', action='store_true', help='hide the header row for unit (which is shown only if one or more columns has a unit)')
    addarg('--show-dtype', action='store_true', help='always include a header row for column dtypes (otherwise shown only if any column is multidimensional)')
    ascii_args = parser.add_argument_group('ASCII arguments')
    addarg = ascii_args.add_argument
    addarg('--delimiter', help='column delimiter string')
    fits_args = parser.add_argument_group('FITS arguments')
    addarg = fits_args.add_argument
    addarg('--hdu', help='name of the HDU to show')
    hdf5_args = parser.add_argument_group('HDF5 arguments')
    addarg = hdf5_args.add_argument
    addarg('--path', help='the path from which to read the table')
    votable_args = parser.add_argument_group('VOTable arguments')
    addarg = votable_args.add_argument
    addarg('--table-id', help='the table to read in')
    args = parser.parse_args(args)
    for (idx, filename) in enumerate(args.filename):
        if idx > 0:
            print()
        showtable(filename, args)