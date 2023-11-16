from visidata import vd, Sheet, options, Column, asyncthread, Progress, PivotGroupRow, ENTER, HistogramColumn
from visidata.loaders._pandas import PandasSheet
from visidata.pivot import PivotSheet

class DataFrameRowSliceAdapter:
    """Tracks original dataframe and a boolean row mask

    This is a workaround to (1) save memory (2) keep id(row)
    consistent when iterating, as id() is used significantly
    by visidata's selectRow implementation.
    """

    def __init__(self, df, mask):
        if False:
            return 10
        pd = vd.importExternal('pandas')
        np = vd.importExternal('numpy')
        if not isinstance(df, pd.DataFrame):
            vd.fail('%s is not a dataframe' % type(df).__name__)
        if not isinstance(mask, pd.Series):
            vd.fail('mask %s is not a Series' % type(mask).__name__)
        if df.shape[0] != mask.shape[0]:
            vd.fail('dataframe and mask have different shapes (%s vs %s)' % (df.shape[0], mask.shape[0]))
        self.df = df
        self.mask_bool = mask
        self.mask_iloc = np.where(mask.values)[0]
        self.mask_count = mask.sum()

    def __len__(self):
        if False:
            while True:
                i = 10
        return self.mask_count

    def __getitem__(self, k):
        if False:
            i = 10
            return i + 15
        if isinstance(k, slice):
            import pandas as pd
            new_mask = pd.Series(False, index=self.df.index)
            new_mask.iloc[self.mask_iloc[k]] = True
            return DataFrameRowSliceAdapter(self.df, new_mask)
        return self.df.iloc[self.mask_iloc[k]]

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return DataFrameRowSliceIter(self.df, self.mask_iloc)

    def __getattr__(self, k):
        if False:
            i = 10
            return i + 15
        return getattr(self.df[self.mask_bool], k)

class DataFrameRowSliceIter:

    def __init__(self, df, mask_iloc, index=0):
        if False:
            i = 10
            return i + 15
        self.df = df
        self.mask_iloc = mask_iloc
        self.index = index

    def __next__(self):
        if False:
            i = 10
            return i + 15
        if self.index >= self.mask_iloc.shape[0]:
            raise StopIteration()
        row = self.df.iloc[self.mask_iloc[self.index]]
        self.index += 1
        return row

def makePandasFreqTable(sheet, *groupByCols):
    if False:
        for i in range(10):
            print('nop')
    fqcolname = '%s_freq' % '-'.join((col.name for col in groupByCols))
    return PandasFreqTableSheet(sheet.name, fqcolname, groupByCols=groupByCols, source=sheet)

class PandasFreqTableSheet(PivotSheet):
    """Generate frequency-table sheet on currently selected column."""
    rowtype = 'bins'

    def selectRow(self, row):
        if False:
            while True:
                i = 10
        self.source._selectByILoc(row.sourcerows.mask_iloc, selected=True)
        return super().selectRow(row)

    def unselectRow(self, row):
        if False:
            print('Hello World!')
        self.source._selectByILoc(row.sourcerows.mask_iloc, selected=False)
        return super().unselectRow(row)

    def updateLargest(self, grouprow):
        if False:
            for i in range(10):
                print('nop')
        self.largest = max(self.largest, len(grouprow.sourcerows))

    def loader(self):
        if False:
            print('Hello World!')
        'Generate frequency table then reverse-sort by length.'
        import pandas as pd
        df = self.source.df.copy()
        if len(self.groupByCols) >= 1:
            _pivot_count_column = '__vd_pivot_count'
            if _pivot_count_column not in df.columns:
                df[_pivot_count_column] = 1
            value_counts = df.pivot_table(index=[c.name for c in self.groupByCols], values=_pivot_count_column, aggfunc='count')[_pivot_count_column].sort_values(ascending=False, kind='mergesort')
        else:
            vd.fail('Unable to do FrequencyTable, no columns to group on provided')
        for c in [Column('count', type=int, getter=lambda col, row: len(row.sourcerows)), Column('percent', type=float, getter=lambda col, row: len(row.sourcerows) * 100 / df.shape[0]), HistogramColumn('histogram', type=str, width=self.options.default_width * 2)]:
            self.addColumn(c)
        for element in Progress(value_counts.index):
            if len(self.groupByCols) == 1:
                element = (element,)
            elif len(element) != len(self.groupByCols):
                vd.fail('different number of index cols and groupby cols (%s vs %s)' % (len(element), len(self.groupByCols)))
            mask = df[self.groupByCols[0].name] == element[0]
            for i in range(1, len(self.groupByCols)):
                mask = mask & (df[self.groupByCols[i].name] == element[i])
            self.addRow(PivotGroupRow(element, (0, 0), DataFrameRowSliceAdapter(df, mask), {}))

    def openRow(self, row):
        if False:
            return 10
        return self.source.expand_source_rows(row)

@Sheet.api
def expand_source_rows(sheet, row):
    if False:
        while True:
            i = 10
    'Support for expanding a row of frequency table to underlying rows'
    if row.sourcerows is None:
        vd.fail('no source rows')
    return PandasSheet(sheet.name, vd.valueNames(row.discrete_keys, row.numeric_key), source=row.sourcerows)
PandasSheet.addCommand('F', 'freq-col', 'vd.push(makePandasFreqTable(sheet, cursorCol))', 'open Frequency Table grouped on current column, with aggregations of other columns')
PandasSheet.addCommand('gF', 'freq-keys', 'vd.push(makePandasFreqTable(sheet, *keyCols))', 'open Frequency Table grouped by all key columns on source sheet, with aggregations of other columns')
PandasFreqTableSheet.init('largest', lambda : 1)
PandasFreqTableSheet.options.numeric_binning = False
vd.addGlobals(makePandasFreqTable=makePandasFreqTable)