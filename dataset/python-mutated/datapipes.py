import random
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
__all__ = ['ConcatDataFramesPipe', 'DataFramesAsTuplesPipe', 'ExampleAggregateAsDataFrames', 'FilterDataFramesPipe', 'PerRowDataFramesPipe', 'ShuffleDataFramesPipe']

@functional_datapipe('_dataframes_as_tuples')
class DataFramesAsTuplesPipe(IterDataPipe):

    def __init__(self, source_datapipe):
        if False:
            while True:
                i = 10
        self.source_datapipe = source_datapipe

    def __iter__(self):
        if False:
            print('Hello World!')
        for df in self.source_datapipe:
            yield from df_wrapper.iterate(df)

@functional_datapipe('_dataframes_per_row', enable_df_api_tracing=True)
class PerRowDataFramesPipe(DFIterDataPipe):

    def __init__(self, source_datapipe):
        if False:
            for i in range(10):
                print('nop')
        self.source_datapipe = source_datapipe

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        for df in self.source_datapipe:
            for i in range(len(df)):
                yield df[i:i + 1]

@functional_datapipe('_dataframes_concat', enable_df_api_tracing=True)
class ConcatDataFramesPipe(DFIterDataPipe):

    def __init__(self, source_datapipe, batch=3):
        if False:
            return 10
        self.source_datapipe = source_datapipe
        self.n_batch = batch

    def __iter__(self):
        if False:
            while True:
                i = 10
        buffer = []
        for df in self.source_datapipe:
            buffer.append(df)
            if len(buffer) == self.n_batch:
                yield df_wrapper.concat(buffer)
                buffer = []
        if len(buffer):
            yield df_wrapper.concat(buffer)

@functional_datapipe('_dataframes_shuffle', enable_df_api_tracing=True)
class ShuffleDataFramesPipe(DFIterDataPipe):

    def __init__(self, source_datapipe):
        if False:
            return 10
        self.source_datapipe = source_datapipe

    def __iter__(self):
        if False:
            while True:
                i = 10
        size = None
        all_buffer = []
        for df in self.source_datapipe:
            if size is None:
                size = df_wrapper.get_len(df)
            for i in range(df_wrapper.get_len(df)):
                all_buffer.append(df_wrapper.get_item(df, i))
        random.shuffle(all_buffer)
        buffer = []
        for df in all_buffer:
            buffer.append(df)
            if len(buffer) == size:
                yield df_wrapper.concat(buffer)
                buffer = []
        if len(buffer):
            yield df_wrapper.concat(buffer)

@functional_datapipe('_dataframes_filter', enable_df_api_tracing=True)
class FilterDataFramesPipe(DFIterDataPipe):

    def __init__(self, source_datapipe, filter_fn):
        if False:
            i = 10
            return i + 15
        self.source_datapipe = source_datapipe
        self.filter_fn = filter_fn

    def __iter__(self):
        if False:
            return 10
        size = None
        all_buffer = []
        filter_res = []
        for df in self.source_datapipe:
            if size is None:
                size = len(df.index)
            for i in range(len(df.index)):
                all_buffer.append(df[i:i + 1])
                filter_res.append(self.filter_fn(df.iloc[i]))
        buffer = []
        for (df, res) in zip(all_buffer, filter_res):
            if res:
                buffer.append(df)
                if len(buffer) == size:
                    yield df_wrapper.concat(buffer)
                    buffer = []
        if len(buffer):
            yield df_wrapper.concat(buffer)

@functional_datapipe('_to_dataframes_pipe', enable_df_api_tracing=True)
class ExampleAggregateAsDataFrames(DFIterDataPipe):

    def __init__(self, source_datapipe, dataframe_size=10, columns=None):
        if False:
            i = 10
            return i + 15
        self.source_datapipe = source_datapipe
        self.columns = columns
        self.dataframe_size = dataframe_size

    def _as_list(self, item):
        if False:
            i = 10
            return i + 15
        try:
            return list(item)
        except Exception:
            return [item]

    def __iter__(self):
        if False:
            return 10
        aggregate = []
        for item in self.source_datapipe:
            aggregate.append(self._as_list(item))
            if len(aggregate) == self.dataframe_size:
                yield df_wrapper.create_dataframe(aggregate, columns=self.columns)
                aggregate = []
        if len(aggregate) > 0:
            yield df_wrapper.create_dataframe(aggregate, columns=self.columns)