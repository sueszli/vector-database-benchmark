from functools import partial
from visidata import VisiData, vd, Sheet, date, anytype, Path, options, Column, asyncthread, Progress, undoAttrCopyFunc, run

@VisiData.api
def open_pandas(vd, p):
    if False:
        while True:
            i = 10
    return PandasSheet(p.name, source=p)

@VisiData.api
def open_dta(vd, p):
    if False:
        for i in range(10):
            print('nop')
    return PandasSheet(p.name, source=p, filetype='stata')
VisiData.open_stata = VisiData.open_pandas
for ft in 'feather gbq orc pickle sas stata'.split():
    funcname = 'open_' + ft
    if not getattr(VisiData, funcname, None):
        setattr(VisiData, funcname, lambda vd, p, ft=ft: PandasSheet(p.name, source=p, filetype=ft))

@VisiData.api
@asyncthread
def save_dta(vd, p, *sheets):
    if False:
        print('Hello World!')
    import pandas as pd
    import numpy as np
    vs = sheets[0]
    columns = [col.name for col in vs.visibleCols]
    types = list()
    dispvals = next(vs.iterdispvals(format=True))
    for (col, _) in dispvals.items():
        if col.type in [bool, int, float]:
            types.append(col.type)
        elif vd.isNumeric(col):
            types.append(float)
        else:
            types.append(str)
    data = np.empty((vs.nRows, len(columns)), dtype=object)
    for (r_i, dispvals) in enumerate(vs.iterdispvals(format=True)):
        for (c_i, v) in enumerate(dispvals.values()):
            data[r_i, c_i] = v
    dtype = {col: t for (col, t) in zip(columns, types)}
    df = pd.DataFrame(data, columns=columns)
    df = df.astype(dtype)
    df.to_stata(p, version=118, write_index=False)

class DataFrameAdapter:

    def __init__(self, df):
        if False:
            print('Hello World!')
        pd = vd.importExternal('pandas')
        if not isinstance(df, pd.DataFrame):
            vd.fail('%s is not a dataframe' % type(df).__name__)
        self.df = df

    def __len__(self):
        if False:
            i = 10
            return i + 15
        if 'df' not in self.__dict__:
            return 0
        return len(self.df)

    def __getitem__(self, k):
        if False:
            return 10
        if isinstance(k, slice):
            return DataFrameAdapter(self.df.iloc[k])
        return self.df.iloc[k]

    def __getattr__(self, k):
        if False:
            while True:
                i = 10
        if 'df' not in self.__dict__:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{k}'")
        return getattr(self.df, k)

class PandasSheet(Sheet):
    """Sheet sourced from a pandas.DataFrame

    Warning:
        The index of the pandas.DataFrame input must be unique.
        Otherwise the selection functionality, which relies on
        looking up selected rows via the index, will break.
        This can be done by calling reset_index().

    Note:
        Columns starting with "__vd_" are reserved for internal usage
        by the VisiData loader.
    """

    def dtype_to_type(self, dtype):
        if False:
            i = 10
            return i + 15
        np = vd.importExternal('numpy')
        dtype = getattr(dtype, 'numpy_dtype', dtype)
        try:
            if np.issubdtype(dtype, np.integer):
                return int
            if np.issubdtype(dtype, np.floating):
                return float
            if np.issubdtype(dtype, np.datetime64):
                return date
        except TypeError:
            pass
        return anytype

    def read_tsv(self, path, **kwargs):
        if False:
            return 10
        'Partial function for reading TSV files using pd.read_csv'
        pd = vd.importExternal('pandas')
        return pd.read_csv(path, sep='\t', **kwargs)

    @property
    def df(self):
        if False:
            print('Hello World!')
        if isinstance(getattr(self, 'rows', None), DataFrameAdapter):
            return self.rows.df

    @df.setter
    def df(self, val):
        if False:
            i = 10
            return i + 15
        if isinstance(getattr(self, 'rows', None), DataFrameAdapter):
            self.rows.df = val
        else:
            self.rows = DataFrameAdapter(val)

    def getValue(self, col, row):
        if False:
            i = 10
            return i + 15
        'Look up column values in the underlying DataFrame.'
        return col.sheet.df.loc[row.name, col.expr]

    def setValue(self, col, row, val):
        if False:
            for i in range(10):
                print('nop')
        "\n        Update a column's value in the underlying DataFrame, loosening the\n        column's type as needed. Take care to avoid assigning to a view or\n        a copy as noted here:\n\n        https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#why-does-assignment-fail-when-using-chained-indexing\n        "
        try:
            col.sheet.df.loc[row.name, col.expr] = val
        except ValueError as err:
            vd.warning(f'Type of {val} does not match column {col.name}. Changing type.')
            col.type = anytype
            col.sheet.df.loc[row.name, col.expr] = val
        self.setModified()

    @asyncthread
    def reload(self):
        if False:
            return 10
        pd = vd.importExternal('pandas')
        if isinstance(self.source, pd.DataFrame):
            df = self.source
        elif isinstance(self.source, Path):
            filetype = getattr(self, 'filetype', self.source.ext)
            if filetype == 'tsv':
                readfunc = self.read_tsv
            elif filetype == 'jsonl':
                readfunc = partial(pd.read_json, lines=True)
            else:
                readfunc = getattr(pd, 'read_' + filetype) or vd.error('no pandas.read_' + filetype)
            df = readfunc(self.source, **options.getall('pandas_' + filetype + '_'))
            if isinstance(df, list):
                for (idx, inner_df) in enumerate(df[1:], start=1):
                    vd.push(PandasSheet(f'{self.name}[{idx}]', source=inner_df))
                df = df[0]
                self.name += '[0]'
            if filetype == 'pickle' and (not isinstance(df, pd.DataFrame)):
                vd.fail('pandas loader can only unpickle dataframes')
        else:
            try:
                df = pd.DataFrame(self.source)
            except ValueError as err:
                vd.fail('error building pandas DataFrame from source data: %s' % err)
        if type(df.index) is not pd.RangeIndex:
            df = df.reset_index(drop=True)
        df.columns = df.columns.astype(str)
        self.columns = []
        for col in (c for c in df.columns if not c.startswith('__vd_')):
            self.addColumn(Column(col, type=self.dtype_to_type(df[col]), getter=self.getValue, setter=self.setValue, expr=col))
        if self.columns[0].name == 'index':
            self.column('index').hide()
        self.rows = DataFrameAdapter(df)
        self._selectedMask = pd.Series(False, index=df.index)
        if df.index.nunique() != df.shape[0]:
            vd.warning('Non-unique index, row selection API may not work or may be incorrect')

    @asyncthread
    def sort(self):
        if False:
            while True:
                i = 10
        'Sort rows according to the current self._ordering.'
        by_cols = []
        ascending = []
        for (col, reverse) in self._ordering[::-1]:
            by_cols.append(col.expr)
            ascending.append(not reverse)
        self.rows.sort_values(by=by_cols, ascending=ascending, inplace=True)

    def _checkSelectedIndex(self):
        if False:
            i = 10
            return i + 15
        pd = vd.importExternal('pandas')
        if self._selectedMask.index is not self.df.index:
            vd.status('pd.DataFrame.index updated, clearing {} selected rows'.format(self._selectedMask.sum()))
            self._selectedMask = pd.Series(False, index=self.df.index)

    def rowid(self, row):
        if False:
            return 10
        return getattr(row, 'name', None) or ''

    def isSelected(self, row):
        if False:
            print('Hello World!')
        if row is None:
            return False
        self._checkSelectedIndex()
        return self._selectedMask.loc[row.name]

    def selectRow(self, row):
        if False:
            for i in range(10):
                print('nop')
        'Select given row'
        self._checkSelectedIndex()
        self._selectedMask.loc[row.name] = True

    def unselectRow(self, row):
        if False:
            return 10
        self._checkSelectedIndex()
        is_selected = self._selectedMask.loc[row.name]
        self._selectedMask.loc[row.name] = False
        return is_selected

    @property
    def nSelectedRows(self):
        if False:
            for i in range(10):
                print('nop')
        self._checkSelectedIndex()
        return self._selectedMask.sum()

    @property
    def selectedRows(self):
        if False:
            while True:
                i = 10
        self._checkSelectedIndex()
        return DataFrameAdapter(self.df.loc[self._selectedMask])

    @asyncthread
    def select(self, rows, status=True, progress=True):
        if False:
            i = 10
            return i + 15
        self.addUndoSelection()
        for row in Progress(rows, 'selecting') if progress else rows:
            self.selectRow(row)

    @asyncthread
    def unselect(self, rows, status=True, progress=True):
        if False:
            i = 10
            return i + 15
        self.addUndoSelection()
        for row in Progress(rows, 'unselecting') if progress else rows:
            self.unselectRow(row)

    def clearSelected(self):
        if False:
            while True:
                i = 10
        pd = vd.importExternal('pandas')
        self._selectedMask = pd.Series(False, index=self.df.index)

    def selectByIndex(self, start=None, end=None):
        if False:
            i = 10
            return i + 15
        self._checkSelectedIndex()
        self._selectedMask.iloc[start:end] = True

    def unselectByIndex(self, start=None, end=None):
        if False:
            print('Hello World!')
        self._checkSelectedIndex()
        self._selectedMask.iloc[start:end] = False

    def toggleByIndex(self, start=None, end=None):
        if False:
            for i in range(10):
                print('nop')
        self._checkSelectedIndex()
        self.addUndoSelection()
        self._selectedMask.iloc[start:end] = ~self._selectedMask.iloc[start:end]

    def _selectByILoc(self, mask, selected=True):
        if False:
            i = 10
            return i + 15
        self._checkSelectedIndex()
        self._selectedMask.iloc[mask] = selected

    @asyncthread
    def selectByRegex(self, regex, columns, unselect=False):
        if False:
            i = 10
            return i + 15
        '\n        Find rows matching regex in the provided columns. By default, add\n        matching rows to the selection. If unselect is True, remove from the\n        active selection instead.\n        '
        pd = vd.importExternal('pandas')
        case_sensitive = 'I' not in vd.options.regex_flags
        masks = pd.DataFrame([self.df[col.expr].astype(str).str.contains(pat=regex, case=case_sensitive, regex=True) for col in columns])
        if unselect:
            self._selectedMask = self._selectedMask & ~masks.any()
        else:
            self._selectedMask = self._selectedMask | masks.any()

    def addUndoSelection(self):
        if False:
            return 10
        vd.addUndo(undoAttrCopyFunc([self], '_selectedMask'))

    @property
    def nRows(self):
        if False:
            print('Hello World!')
        if self.df is None:
            return 0
        return len(self.df)

    def newRows(self, n):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return n rows of empty data. Let pandas decide on the most\n        appropriate missing value (NaN, NA, etc) based on the underlying\n        DataFrame's dtypes.\n        "
        pd = vd.importExternal('pandas')
        return pd.DataFrame({col: [None] * n for col in self.df.columns}).astype(self.df.dtypes.to_dict(), errors='ignore')

    def addRows(self, rows, index=None, undo=True):
        if False:
            while True:
                i = 10
        pd = vd.importExternal('pandas')
        if index is None:
            self.df = self.df.append(pd.DataFrame(rows))
        else:
            self.df = pd.concat((self.df.iloc[0:index], pd.DataFrame(rows), self.df.iloc[index:]))
        self.df.index = pd.RangeIndex(self.nRows)
        self._checkSelectedIndex()
        if undo:
            self.setModified()
            vd.addUndo(self._deleteRows, range(index, index + len(rows)))

    def _deleteRows(self, which):
        if False:
            return 10
        pd = vd.importExternal('pandas')
        self.df.drop(which, inplace=True)
        self.df.index = pd.RangeIndex(self.nRows)
        self._checkSelectedIndex()

    def addRow(self, row, index=None):
        if False:
            i = 10
            return i + 15
        self.addRows([row], index)
        vd.addUndo(self._deleteRows, index or self.nRows - 1)

    def delete_row(self, rowidx):
        if False:
            i = 10
            return i + 15
        pd = vd.importExternal('pandas')
        oldrow = self.df.iloc[rowidx:rowidx + 1]
        vd.addUndo(self.addRows, oldrow.to_dict(), rowidx, undo=False)
        self._deleteRows(rowidx)
        vd.memory.cliprows = [oldrow]
        self.setModified()

    def deleteBy(self, by):
        if False:
            return 10
        'Delete rows for which func(row) is true.  Returns number of deleted rows.'
        pd = vd.importExternal('pandas')
        nRows = self.nRows
        vd.addUndo(setattr, self, 'df', self.df.copy())
        self.df = self.df[~by]
        self.df.index = pd.RangeIndex(self.nRows)
        ndeleted = nRows - self.nRows
        self.setModified()
        vd.status('deleted %s %s' % (ndeleted, self.rowtype))
        return ndeleted

    def deleteSelected(self):
        if False:
            while True:
                i = 10
        'Delete all selected rows.'
        self.deleteBy(self._selectedMask)

@VisiData.global_api
def view_pandas(vd, df):
    if False:
        i = 10
        return i + 15
    run(PandasSheet('', source=df))
PandasSheet.addCommand(None, 'stoggle-rows', 'toggleByIndex()', 'toggle selection of all rows')
PandasSheet.addCommand(None, 'select-rows', 'selectByIndex()', 'select all rows')
PandasSheet.addCommand(None, 'unselect-rows', 'unselectByIndex()', 'unselect all rows')
PandasSheet.addCommand(None, 'stoggle-before', 'toggleByIndex(end=cursorRowIndex)', 'toggle selection of rows from top to cursor')
PandasSheet.addCommand(None, 'select-before', 'selectByIndex(end=cursorRowIndex)', 'select all rows from top to cursor')
PandasSheet.addCommand(None, 'unselect-before', 'unselectByIndex(end=cursorRowIndex)', 'unselect all rows from top to cursor')
PandasSheet.addCommand(None, 'stoggle-after', 'toggleByIndex(start=cursorRowIndex)', 'toggle selection of rows from cursor to bottom')
PandasSheet.addCommand(None, 'select-after', 'selectByIndex(start=cursorRowIndex)', 'select all rows from cursor to bottom')
PandasSheet.addCommand(None, 'unselect-after', 'unselectByIndex(start=cursorRowIndex)', 'unselect all rows from cursor to bottom')
PandasSheet.addCommand(None, 'random-rows', 'nrows=int(input("random number to select: ", value=nRows)); vs=copy(sheet); vs.name=name+"_sample"; vs.rows=DataFrameAdapter(sheet.df.sample(nrows or nRows)); vd.push(vs)', 'open duplicate sheet with a random population subset of N rows')
PandasSheet.addCommand('|', 'select-col-regex', 'selectByRegex(regex=input("select regex: ", type="regex", defaultLast=True), columns=[cursorCol])', 'select rows matching regex in current column')
PandasSheet.addCommand('\\', 'unselect-col-regex', 'selectByRegex(regex=input("select regex: ", type="regex", defaultLast=True), columns=[cursorCol], unselect=True)', 'unselect rows matching regex in current column')
PandasSheet.addCommand('g|', 'select-cols-regex', 'selectByRegex(regex=input("select regex: ", type="regex", defaultLast=True), columns=visibleCols)', 'select rows matching regex in any visible column')
PandasSheet.addCommand('g\\', 'unselect-cols-regex', 'selectByRegex(regex=input("select regex: ", type="regex", defaultLast=True), columns=visibleCols, unselect=True)', 'unselect rows matching regex in any visible column')
PandasSheet.addCommand('"', 'dup-selected', 'vs=PandasSheet(sheet.name, "selectedref", source=selectedRows.df); vd.push(vs)', 'open duplicate sheet with only selected rows')
vd.addGlobals({'PandasSheet': PandasSheet})