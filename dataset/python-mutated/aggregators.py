import sys
import math
import functools
import collections
import statistics
from visidata import Progress, Sheet, Column, ColumnsSheet, VisiData
from visidata import vd, anytype, vlen, asyncthread, wrapply, AttrDict
vd.help_aggregators = '# Aggregators Help\nHELPTODO'
vd.option('null_value', None, 'a value to be counted as null', replay=True)

@Column.api
def getValueRows(self, rows):
    if False:
        print('Hello World!')
    'Generate (value, row) for each row in *rows* at this column, excluding null and error values.'
    f = self.sheet.isNullFunc()
    for r in Progress(rows, 'calculating'):
        try:
            v = self.getTypedValue(r)
            if not f(v):
                yield (v, r)
        except Exception:
            pass

@Column.api
def getValues(self, rows):
    if False:
        print('Hello World!')
    'Generate value for each row in *rows* at this column, excluding null and error values.'
    for (v, r) in self.getValueRows(rows):
        yield v
vd.aggregators = collections.OrderedDict()
Column.init('aggstr', str, copy=True)

def aggregators_get(col):
    if False:
        print('Hello World!')
    'A space-separated names of aggregators on this column.'
    return list((vd.aggregators[k] for k in (col.aggstr or '').split()))

def aggregators_set(col, aggs):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(aggs, str):
        newaggs = []
        for agg in aggs.split():
            if agg not in vd.aggregators:
                vd.fail(f'unknown aggregator {agg}')
            newaggs.append(agg)
    elif aggs is None:
        newaggs = ''
    else:
        newaggs = [agg.name for agg in aggs]
    col.aggstr = ' '.join(newaggs)
Column.aggregators = property(aggregators_get, aggregators_set)

class Aggregator:

    def __init__(self, name, type, funcRows, funcValues=None, helpstr='foo'):
        if False:
            return 10
        'Define aggregator `name` that calls func(col, rows)'
        self.type = type
        self.func = funcRows
        self.funcValues = funcValues
        self.helpstr = helpstr
        self.name = name

    def __call__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return self.func(*args, **kwargs)
_defaggr = Aggregator

@VisiData.api
def aggregator(vd, name, funcValues, helpstr='', *args, type=None):
    if False:
        for i in range(10):
            print('nop')
    'Define simple aggregator *name* that calls ``funcValues(values, *args)`` to aggregate *values*.  Use *type* to force the default type of the aggregated column.'

    def _funcRows(col, rows):
        if False:
            return 10
        vals = list(col.getValues(rows))
        try:
            return funcValues(vals, *args)
        except Exception as e:
            if len(vals) == 0:
                return None
            return e
    vd.aggregators[name] = _defaggr(name, type, _funcRows, funcValues=funcValues, helpstr=helpstr)

def mean(vals):
    if False:
        for i in range(10):
            print('nop')
    vals = list(vals)
    if vals:
        return float(sum(vals)) / len(vals)

def _vsum(vals):
    if False:
        while True:
            i = 10
    return sum(vals, start=type(vals[0] if len(vals) else 0)())
vsum = _vsum if sys.version_info[:2] >= (3, 8) else sum

def _percentile(N, percent, key=lambda x: x):
    if False:
        return 10
    '\n    Find the percentile of a list of values.\n\n    @parameter N - is a list of values. Note N MUST BE already sorted.\n    @parameter percent - a float value from 0.0 to 1.0.\n    @parameter key - optional key function to compute value from each element of N.\n\n    @return - the percentile of the values\n    '
    if not N:
        return None
    k = (len(N) - 1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return key(N[int(k)])
    d0 = key(N[int(f)]) * (c - k)
    d1 = key(N[int(c)]) * (k - f)
    return d0 + d1

@functools.lru_cache(100)
def percentile(pct, helpstr=''):
    if False:
        return 10
    return _defaggr('p%s' % pct, None, lambda col, rows, pct=pct: _percentile(sorted(col.getValues(rows)), pct / 100), helpstr=helpstr)

def quantiles(q, helpstr):
    if False:
        i = 10
        return i + 15
    return [percentile(round(100 * i / q), helpstr) for i in range(1, q)]
vd.aggregator('min', min, 'minimum value')
vd.aggregator('max', max, 'maximum value')
vd.aggregator('avg', mean, 'arithmetic mean of values', type=float)
vd.aggregator('mean', mean, 'arithmetic mean of values', type=float)
vd.aggregator('median', statistics.median, 'median of values')
vd.aggregator('mode', statistics.mode, 'mode of values')
vd.aggregator('sum', vsum, 'sum of values')
vd.aggregator('distinct', set, 'distinct values', type=vlen)
vd.aggregator('count', lambda values: sum((1 for v in values)), 'number of values', type=int)
vd.aggregator('list', list, 'list of values')
vd.aggregator('stdev', statistics.stdev, 'standard deviation of values', type=float)
vd.aggregators['q3'] = quantiles(3, 'tertiles (33/66th pctile)')
vd.aggregators['q4'] = quantiles(4, 'quartiles (25/50/75th pctile)')
vd.aggregators['q5'] = quantiles(5, 'quintiles (20/40/60/80th pctiles)')
vd.aggregators['q10'] = quantiles(10, 'deciles (10/20/30/40/50/60/70/80/90th pctiles)')
for pct in (10, 20, 25, 30, 33, 40, 50, 60, 67, 70, 75, 80, 90, 95, 99):
    vd.aggregators[f'p{pct}'] = percentile(pct, f'{pct}th percentile')
vd.aggregators['keymax'] = _defaggr('keymax', anytype, lambda col, rows: col.sheet.rowkey(max(col.getValueRows(rows))[1]), helpstr='key of the maximum value')
ColumnsSheet.columns += [Column('aggregators', getter=lambda c, r: r.aggstr, setter=lambda c, r, v: setattr(r, 'aggregators', v), help='change the metrics calculated in every Frequency or Pivot derived from the source sheet')]

@Sheet.api
def addAggregators(sheet, cols, aggrnames):
    if False:
        for i in range(10):
            print('nop')
    'Add each aggregator in list of *aggrnames* to each of *cols*.'
    for aggrname in aggrnames:
        aggrs = vd.aggregators.get(aggrname)
        aggrs = aggrs if isinstance(aggrs, list) else [aggrs]
        for aggr in aggrs:
            for c in cols:
                if not hasattr(c, 'aggregators'):
                    c.aggregators = []
                if aggr and aggr not in c.aggregators:
                    c.aggregators += [aggr]

@Column.api
def aggname(col, agg):
    if False:
        i = 10
        return i + 15
    'Consistent formatting of the name of given aggregator for this column.  e.g. "col1_sum"'
    return '%s_%s' % (col.name, agg.name)

@Column.api
@asyncthread
def memo_aggregate(col, agg, rows):
    if False:
        while True:
            i = 10
    'Show aggregated value in status, and add to memory.'
    aggval = agg(col, rows)
    typedval = wrapply(agg.type or col.type, aggval)
    dispval = col.format(typedval)
    k = col.name + '_' + agg.name
    vd.status(f'{k}={dispval}')
    vd.memory[k] = typedval

@VisiData.property
def aggregator_choices(vd):
    if False:
        print('Hello World!')
    return [AttrDict(key=agg, desc=v[0].helpstr if isinstance(v, list) else v.helpstr) for (agg, v) in vd.aggregators.items() if not agg.startswith('p')]

@VisiData.api
def chooseAggregators(vd):
    if False:
        print('Hello World!')
    prompt = 'choose aggregators: '

    def _fmt_aggr_summary(match, row, trigger_key):
        if False:
            return 10
        formatted_aggrname = match.formatted.get('key', row.key) if match else row.key
        r = ' ' * (len(prompt) - 3)
        r += f'[:keystrokes]{trigger_key}[/]  '
        r += formatted_aggrname
        if row.desc:
            r += ' - '
            r += match.formatted.get('desc', row.desc) if match else row.desc
        return r
    r = vd.activeSheet.inputPalette(prompt, vd.aggregator_choices, value_key='key', formatter=_fmt_aggr_summary, type='aggregators', help=vd.help_aggregators, multiple=True)
    aggrs = r.split()
    for aggr in aggrs:
        vd.usedInputs[aggr] += 1
    return aggrs
Sheet.addCommand('+', 'aggregate-col', 'addAggregators([cursorCol], chooseAggregators())', 'Add aggregator to current column')
Sheet.addCommand('z+', 'memo-aggregate', 'for agg in chooseAggregators(): cursorCol.memo_aggregate(aggregators[agg], selectedRows or rows)', 'memo result of aggregator over values in selected rows for current column')
ColumnsSheet.addCommand('g+', 'aggregate-cols', 'addAggregators(selectedRows or source[0].nonKeyVisibleCols, chooseAggregators())', 'add aggregators to selected source columns')
vd.addMenuItems('\n    Column > Add aggregator > aggregate-col\n')