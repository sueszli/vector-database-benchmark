from collections import namedtuple
from datetime import datetime as dt
import pandas as pd
from arctic.date._mktz import mktz
from arctic.multi_index import groupby_asof
BitemporalItem = namedtuple('BitemporalItem', 'symbol, library, data, metadata, last_updated')

class BitemporalStore(object):
    """ A versioned pandas DataFrame store.

    As the name hinted, this holds versions of DataFrame by maintaining an extra 'insert time' index internally.
    """

    def __init__(self, version_store, observe_column='observed_dt'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        version_store : `VersionStore`\n            The version store that keeps the underlying data frames\n        observe_column : `str`\n            Column name for the datetime index that represents the insertion time of a row of data. Unless you intend to\n            read raw data out, this column is internal to this store.\n        '
        self._store = version_store
        self.observe_column = observe_column

    def read(self, symbol, as_of=None, raw=False, **kwargs):
        if False:
            i = 10
            return i + 15
        'Read data for the named symbol. Returns a BitemporalItem object with\n        a data and metdata element (as passed into write).\n\n        Parameters\n        ----------\n        symbol : `str`\n            symbol name for the item\n        as_of : `datetime.datetime`\n            Return the data as it was as_of the point in time.\n        raw : `bool`\n            If True, will return the full bitemporal dataframe (i.e. all versions of the data). This also means as_of is\n            ignored.\n\n        Returns\n        -------\n        BitemporalItem namedtuple which contains a .data and .metadata element\n        '
        item = self._store.read(symbol, **kwargs)
        last_updated = max(item.data.index.get_level_values(self.observe_column))
        if raw:
            return BitemporalItem(symbol=symbol, library=self._store._arctic_lib.get_name(), data=item.data, metadata=item.metadata, last_updated=last_updated)
        else:
            index_names = list(item.data.index.names)
            index_names.remove(self.observe_column)
            return BitemporalItem(symbol=symbol, library=self._store._arctic_lib.get_name(), data=groupby_asof(item.data, as_of=as_of, dt_col=index_names, asof_col=self.observe_column), metadata=item.metadata, last_updated=last_updated)

    def update(self, symbol, data, metadata=None, upsert=True, as_of=None, **kwargs):
        if False:
            return 10
        ' Append \'data\' under the specified \'symbol\' name to this library.\n\n        Parameters\n        ----------\n        symbol : `str`\n            symbol name for the item\n        data : `pd.DataFrame`\n            to be persisted\n        metadata : `dict`\n            An optional dictionary of metadata to persist along with the symbol. If None and there are existing\n            metadata, current metadata will be maintained\n        upsert : `bool`\n            Write \'data\' if no previous version exists.\n        as_of : `datetime.datetime`\n            The "insert time". Default to datetime.now()\n        '
        local_tz = mktz()
        if not as_of:
            as_of = dt.now()
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=local_tz)
        data = self._add_observe_dt_index(data, as_of)
        if upsert and (not self._store.has_symbol(symbol)):
            df = data
        else:
            existing_item = self._store.read(symbol, **kwargs)
            if metadata is None:
                metadata = existing_item.metadata
            df = existing_item.data.append(data).sort_index(kind='mergesort')
        self._store.write(symbol, df, metadata=metadata, prune_previous_version=True)

    def write(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('Direct write for BitemporalStore is not supported. Use append insteadto add / modify timeseries.')

    def _add_observe_dt_index(self, df, as_of):
        if False:
            print('Hello World!')
        index_names = list(df.index.names)
        index_names.append(self.observe_column)
        index = [x + (as_of,) if df.index.nlevels > 1 else (x, as_of) for x in df.index.tolist()]
        df = df.set_index(pd.MultiIndex.from_tuples(index, names=index_names), inplace=False)
        return df