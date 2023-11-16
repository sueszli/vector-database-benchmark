"""|Cosmology| <-> ECSV I/O, using |Cosmology.read| and |Cosmology.write|.

This module provides functions to write/read a |Cosmology| object to/from an ECSV file.
The functions are registered with ``readwrite_registry`` under the format name
"ascii.ecsv".

We assume the following setup:

    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> temp_dir = TemporaryDirectory()

To see reading a Cosmology from an ECSV file, we first write a Cosmology to an ECSV
file:

    >>> from astropy.cosmology import Cosmology, Planck18
    >>> file = Path(temp_dir.name) / "file.ecsv"
    >>> Planck18.write(file)

    >>> with open(file) as f: print(f.read())
    # %ECSV 1.0
    # ---
    # datatype:
    # - {name: name, datatype: string}
    ...
    # meta: !!omap
    # - {Oc0: 0.2607}
    ...
    # schema: astropy-2.0
    name H0 Om0 Tcmb0 Neff m_nu Ob0
    Planck18 67.66 0.30966 2.7255 3.046 [0.0,0.0,0.06] 0.04897
    <BLANKLINE>

Now we can read the Cosmology from the ECSV file, constructing a new cosmological
instance identical to the ``Planck18`` cosmology from which it was generated.

    >>> cosmo = Cosmology.read(file)
    >>> print(cosmo)
    FlatLambdaCDM(name="Planck18", H0=67.66 km / (Mpc s), Om0=0.30966,
                Tcmb0=2.7255 K, Neff=3.046, m_nu=[0. 0. 0.06] eV, Ob0=0.04897)
    >>> cosmo == Planck18
    True

If a file already exists, attempting to write will raise an error unless
``overwrite=True``.

    >>> Planck18.write(file, overwrite=True)

By default the cosmology class is written to the Table metadata. This can be changed to
a column of the table using the ``cosmology_in_meta`` keyword argument.

    >>> file = Path(temp_dir.name) / "file2.ecsv"
    >>> Planck18.write(file, cosmology_in_meta=False)
    >>> with open(file) as f: print(f.read())
    # %ECSV 1.0
    # ---
    # datatype:
    # - {name: cosmology, datatype: string}
    # - {name: name, datatype: string}
    ...
    # meta: !!omap
    # - {Oc0: 0.2607}
    ...
    # schema: astropy-2.0
    cosmology name H0 Om0 Tcmb0 Neff m_nu Ob0
    FlatLambdaCDM Planck18 67.66 0.30966 2.7255 3.046 [0.0,0.0,0.06] 0.04897
    <BLANKLINE>

The ``cosmology`` information (column or metadata) may be omitted if the cosmology class
(or its string name) is passed as the ``cosmology`` keyword argument to
|Cosmology.read|. Alternatively, specific cosmology classes can be used to parse the
data.

    >>> from astropy.cosmology import FlatLambdaCDM
    >>> print(FlatLambdaCDM.read(file))
    FlatLambdaCDM(name="Planck18", H0=67.66 km / (Mpc s), Om0=0.30966,
                    Tcmb0=2.7255 K, Neff=3.046, m_nu=[0. 0. 0.06] eV, Ob0=0.04897)

When using a specific cosmology class, the class' default parameter values are used to
fill in any missing information.

For files with multiple rows of cosmological parameters, the ``index`` argument is
needed to select the correct row. The index can be an integer for the row number or, if
the table is indexed by a column, the value of that column. If the table is not indexed
and ``index`` is a string, the "name" column is used as the indexing column.

Here is an example where ``index`` is needed and can be either an integer (for the row
number) or the name of one of the cosmologies, e.g. 'Planck15'.

    >>> from astropy.cosmology import Planck13, Planck15, Planck18
    >>> from astropy.table import vstack
    >>> cts = vstack([c.to_format("astropy.table")
    ...               for c in (Planck13, Planck15, Planck18)],
    ...              metadata_conflicts='silent')
    >>> file = Path(temp_dir.name) / "file3.ecsv"
    >>> cts.write(file)
    >>> with open(file) as f: print(f.read())
    # %ECSV 1.0
    # ---
    # datatype:
    # - {name: name, datatype: string}
    ...
    # meta: !!omap
    # - {Oc0: 0.2607}
    ...
    # schema: astropy-2.0
    name H0 Om0 Tcmb0 Neff m_nu Ob0
    Planck13 67.77 0.30712 2.7255 3.046 [0.0,0.0,0.06] 0.048252
    Planck15 67.74 0.3075 2.7255 3.046 [0.0,0.0,0.06] 0.0486
    Planck18 67.66 0.30966 2.7255 3.046 [0.0,0.0,0.06] 0.04897

    >>> cosmo = Cosmology.read(file, index="Planck15", format="ascii.ecsv")
    >>> cosmo == Planck15
    True

Fields of the table in the file can be renamed to match the
`~astropy.cosmology.Cosmology` class' signature using the ``rename`` argument. This is
useful when the files's column names do not match the class' parameter names.

    >>> file = Path(temp_dir.name) / "file4.ecsv"
    >>> Planck18.write(file, rename={"H0": "Hubble"})
    >>> with open(file) as f: print(f.read())
     # %ECSV 1.0
    # ---
    # datatype:
    # - {name: name, datatype: string}
    ...
    # meta: !!omap
    # - {Oc0: 0.2607}
    ...
    # schema: astropy-2.0
    name Hubble Om0 Tcmb0 Neff m_nu Ob0
    ...

    >>> cosmo = Cosmology.read(file, rename={"Hubble": "H0"})
    >>> cosmo == Planck18
    True

By default :class:`~astropy.cosmology.Cosmology` instances are written using
`~astropy.table.QTable` as an intermediate representation (for details see
|Cosmology.to_format|, with ``format="astropy.table"``). The `~astropy.table.Table` type
can be changed using the ``cls`` keyword argument.

    >>> from astropy.table import Table
    >>> file = Path(temp_dir.name) / "file5.ecsv"
    >>> Planck18.write(file, cls=Table)

For most use cases, the default ``cls`` of :class:`~astropy.table.QTable` is recommended
and will be largely indistinguishable from other table types, as the ECSV format is
agnostic to the table type. An example of a difference that might necessitate using a
different table type is if a different ECSV schema is desired.

Additional keyword arguments are passed to ``QTable.read`` and ``QTable.write``.

.. testcleanup::

    >>> temp_dir.cleanup()
"""
import astropy.cosmology.units as cu
import astropy.units as u
from astropy.cosmology.connect import readwrite_registry
from astropy.cosmology.core import Cosmology
from astropy.table import QTable
from .table import from_table, to_table

def read_ecsv(filename, index=None, *, move_to_meta=False, cosmology=None, rename=None, **kwargs):
    if False:
        i = 10
        return i + 15
    'Read a `~astropy.cosmology.Cosmology` from an ECSV file.\n\n    Parameters\n    ----------\n    filename : path-like or file-like\n        From where to read the Cosmology.\n    index : int, str, or None, optional\n        Needed to select the row in tables with multiple rows. ``index`` can be an\n        integer for the row number or, if the table is indexed by a column, the value of\n        that column. If the table is not indexed and ``index`` is a string, the "name"\n        column is used as the indexing column.\n\n    move_to_meta : bool (optional, keyword-only)\n        Whether to move keyword arguments that are not in the Cosmology class\' signature\n        to the Cosmology\'s metadata. This will only be applied if the Cosmology does NOT\n        have a keyword-only argument (e.g. ``**kwargs``). Arguments moved to the\n        metadata will be merged with existing metadata, preferring specified metadata in\n        the case of a merge conflict (e.g. for ``Cosmology(meta={\'key\':10}, key=42)``,\n        the ``Cosmology.meta`` will be ``{\'key\': 10}``).\n\n    rename : dict or None (optional keyword-only)\n        A dictionary mapping column names to fields of the\n        `~astropy.cosmology.Cosmology`.\n\n    **kwargs\n        Passed to ``QTable.read``\n\n    Returns\n    -------\n    `~astropy.cosmology.Cosmology` subclass instance\n\n    Examples\n    --------\n    We assume the following setup:\n\n        >>> from pathlib import Path\n        >>> from tempfile import TemporaryDirectory\n        >>> temp_dir = TemporaryDirectory()\n\n    To see reading a Cosmology from an ECSV file, we first write a Cosmology to an ECSV\n    file:\n\n        >>> from astropy.cosmology import Cosmology, Planck18\n        >>> file = Path(temp_dir.name) / "file.ecsv"\n        >>> Planck18.write(file)\n\n        >>> with open(file) as f: print(f.read())\n        # %ECSV 1.0\n        # ---\n        # datatype:\n        # - {name: name, datatype: string}\n        ...\n        # meta: !!omap\n        # - {Oc0: 0.2607}\n        ...\n        # schema: astropy-2.0\n        name H0 Om0 Tcmb0 Neff m_nu Ob0\n        Planck18 67.66 0.30966 2.7255 3.046 [0.0,0.0,0.06] 0.04897\n        <BLANKLINE>\n\n    Now we can read the Cosmology from the ECSV file, constructing a new cosmological\n    instance identical to the ``Planck18`` cosmology from which it was generated.\n\n        >>> cosmo = Cosmology.read(file)\n        >>> print(cosmo)\n        FlatLambdaCDM(name="Planck18", H0=67.66 km / (Mpc s), Om0=0.30966,\n                    Tcmb0=2.7255 K, Neff=3.046, m_nu=[0. 0. 0.06] eV, Ob0=0.04897)\n        >>> cosmo == Planck18\n        True\n\n    The ``cosmology`` information (column or metadata) may be omitted if the cosmology\n    class (or its string name) is passed as the ``cosmology`` keyword argument to\n    |Cosmology.read|. Alternatively, specific cosmology classes can be used to parse the\n    data.\n\n        >>> from astropy.cosmology import FlatLambdaCDM\n        >>> print(FlatLambdaCDM.read(file))\n        FlatLambdaCDM(name="Planck18", H0=67.66 km / (Mpc s), Om0=0.30966,\n                      Tcmb0=2.7255 K, Neff=3.046, m_nu=[0. 0. 0.06] eV, Ob0=0.04897)\n\n    When using a specific cosmology class, the class\' default parameter values are used\n    to fill in any missing information.\n\n    For files with multiple rows of cosmological parameters, the ``index`` argument is\n    needed to select the correct row. The index can be an integer for the row number or,\n    if the table is indexed by a column, the value of that column. If the table is not\n    indexed and ``index`` is a string, the "name" column is used as the indexing column.\n\n    Here is an example where ``index`` is needed and can be either an integer (for the\n    row number) or the name of one of the cosmologies, e.g. \'Planck15\'.\n\n        >>> from astropy.cosmology import Planck13, Planck15, Planck18\n        >>> from astropy.table import vstack\n        >>> cts = vstack([c.to_format("astropy.table")\n        ...               for c in (Planck13, Planck15, Planck18)],\n        ...              metadata_conflicts=\'silent\')\n        >>> file = Path(temp_dir.name) / "file2.ecsv"\n        >>> cts.write(file)\n        >>> with open(file) as f: print(f.read())\n        # %ECSV 1.0\n        # ---\n        # datatype:\n        # - {name: name, datatype: string}\n        ...\n        # meta: !!omap\n        # - {Oc0: 0.2607}\n        ...\n        # schema: astropy-2.0\n        name H0 Om0 Tcmb0 Neff m_nu Ob0\n        Planck13 67.77 0.30712 2.7255 3.046 [0.0,0.0,0.06] 0.048252\n        Planck15 67.74 0.3075 2.7255 3.046 [0.0,0.0,0.06] 0.0486\n        Planck18 67.66 0.30966 2.7255 3.046 [0.0,0.0,0.06] 0.04897\n\n        >>> cosmo = Cosmology.read(file, index="Planck15", format="ascii.ecsv")\n        >>> cosmo == Planck15\n        True\n\n    Fields of the table in the file can be renamed to match the\n    `~astropy.cosmology.Cosmology` class\' signature using the ``rename`` argument. This\n    is useful when the files\'s column names do not match the class\' parameter names.\n    For this example we need to make a new file with renamed columns:\n\n        >>> file = Path(temp_dir.name) / "file3.ecsv"\n        >>> renamed_table = Planck18.to_format("astropy.table", rename={"H0": "Hubble"})\n        >>> renamed_table.write(file)\n        >>> with open(file) as f: print(f.read())\n        # %ECSV 1.0\n        # ---\n        # datatype:\n        # - {name: name, datatype: string}\n        ...\n        # meta: !!omap\n        # - {Oc0: 0.2607}\n        ...\n        # schema: astropy-2.0\n        name Hubble Om0 Tcmb0 Neff m_nu Ob0\n        ...\n\n    Now we can read the Cosmology from the ECSV file, with the required renaming:\n\n        >>> cosmo = Cosmology.read(file, rename={"Hubble": "H0"})\n        >>> cosmo == Planck18\n        True\n\n    Additional keyword arguments are passed to ``QTable.read``.\n\n    .. testcleanup::\n\n        >>> temp_dir.cleanup()\n    '
    kwargs['format'] = 'ascii.ecsv'
    with u.add_enabled_units(cu):
        table = QTable.read(filename, **kwargs)
    return from_table(table, index=index, move_to_meta=move_to_meta, cosmology=cosmology, rename=rename)

def write_ecsv(cosmology, file, *, overwrite=False, cls=QTable, cosmology_in_meta=True, rename=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Serialize the cosmology into a ECSV.\n\n    Parameters\n    ----------\n    cosmology : `~astropy.cosmology.Cosmology`\n        The cosmology instance to convert to a mapping.\n    file : path-like or file-like\n        Location to save the serialized cosmology.\n\n    overwrite : bool\n        Whether to overwrite the file, if it exists.\n    cls : type (optional, keyword-only)\n        Astropy :class:`~astropy.table.Table` (sub)class to use when writing. Default is\n        :class:`~astropy.table.QTable`.\n    cosmology_in_meta : bool (optional, keyword-only)\n        Whether to put the cosmology class in the Table metadata (if `True`, default) or\n        as the first column (if `False`).\n    rename : dict or None (optional keyword-only)\n        A dictionary mapping fields of the `~astropy.cosmology.Cosmology` to columns of\n        the table.\n\n    **kwargs\n        Passed to ``cls.write``\n\n    Raises\n    ------\n    TypeError\n        If kwarg (optional) \'cls\' is not a subclass of `astropy.table.Table`\n\n    Examples\n    --------\n    We assume the following setup:\n\n        >>> from pathlib import Path\n        >>> from tempfile import TemporaryDirectory\n        >>> temp_dir = TemporaryDirectory()\n\n    A Cosmology can be written to an ECSV file:\n\n        >>> from astropy.cosmology import Cosmology, Planck18\n        >>> file = Path(temp_dir.name) / "file.ecsv"\n        >>> Planck18.write(file)\n\n        >>> with open(file) as f: print(f.read())\n        # %ECSV 1.0\n        # ---\n        # datatype:\n        # - {name: name, datatype: string}\n        ...\n        # meta: !!omap\n        # - {Oc0: 0.2607}\n        ...\n        # schema: astropy-2.0\n        name H0 Om0 Tcmb0 Neff m_nu Ob0\n        Planck18 67.66 0.30966 2.7255 3.046 [0.0,0.0,0.06] 0.04897\n        <BLANKLINE>\n\n    If a file already exists, attempting to write will raise an error unless\n    ``overwrite=True``.\n\n        >>> Planck18.write(file, overwrite=True)\n\n    By default :class:`~astropy.cosmology.Cosmology` instances are written using\n    `~astropy.table.QTable` as an intermediate representation (for details see\n    |Cosmology.to_format|, with ``format="astropy.table"``). The `~astropy.table.Table`\n    type can be changed using the ``cls`` keyword argument.\n\n        >>> from astropy.table import Table\n        >>> file = Path(temp_dir.name) / "file2.ecsv"\n        >>> Planck18.write(file, cls=Table)\n\n    For most use cases, the default ``cls`` of :class:`~astropy.table.QTable` is\n    recommended and will be largely indistinguishable from other table types, as the\n    ECSV format is agnostic to the table type. An example of a difference that might\n    necessitate using a different table type is if a different ECSV schema is desired.\n\n    By default the cosmology class is written to the Table metadata. This can be changed\n    to a column of the table using the ``cosmology_in_meta`` keyword argument.\n\n        >>> file = Path(temp_dir.name) / "file3.ecsv"\n        >>> Planck18.write(file, cosmology_in_meta=False)\n        >>> with open(file) as f: print(f.read())\n        # %ECSV 1.0\n        # ---\n        # datatype:\n        # - {name: cosmology, datatype: string}\n        # - {name: name, datatype: string}\n        ...\n        # meta: !!omap\n        # - {Oc0: 0.2607}\n        ...\n        # schema: astropy-2.0\n        cosmology name H0 Om0 Tcmb0 Neff m_nu Ob0\n        FlatLambdaCDM Planck18 67.66 0.30966 2.7255 3.046 [0.0,0.0,0.06] 0.04897\n        <BLANKLINE>\n\n    Fields of the Cosmology can be renamed to when writing to an ECSV file using the\n    ``rename`` argument.\n\n        >>> file = Path(temp_dir.name) / "file4.ecsv"\n        >>> Planck18.write(file, rename={"H0": "Hubble"})\n        >>> with open(file) as f: print(f.read())\n        # %ECSV 1.0\n        # ---\n        # datatype:\n        # - {name: name, datatype: string}\n        ...\n        # meta: !!omap\n        # - {Oc0: 0.2607}\n        ...\n        # schema: astropy-2.0\n        name Hubble Om0 Tcmb0 Neff m_nu Ob0\n        ...\n\n    Additional keyword arguments are passed to :attr:`astropy.table.QTable.write`.\n\n    .. testcleanup::\n\n        >>> temp_dir.cleanup()\n    '
    table = to_table(cosmology, cls=cls, cosmology_in_meta=cosmology_in_meta, rename=rename)
    kwargs['format'] = 'ascii.ecsv'
    table.write(file, overwrite=overwrite, **kwargs)

def ecsv_identify(origin, filepath, fileobj, *args, **kwargs):
    if False:
        return 10
    'Identify if object uses the Table format.\n\n    Returns\n    -------\n    bool\n    '
    return filepath is not None and filepath.endswith('.ecsv')
readwrite_registry.register_reader('ascii.ecsv', Cosmology, read_ecsv)
readwrite_registry.register_writer('ascii.ecsv', Cosmology, write_ecsv)
readwrite_registry.register_identifier('ascii.ecsv', Cosmology, ecsv_identify)