from pandas import DataFrame, Series
from ._chunker import Chunker

class PassthroughChunker(Chunker):
    TYPE = 'passthru'

    def to_chunks(self, df, **kwargs):
        if False:
            print('Hello World!')
        "\n        pass thru chunker of the dataframe/series\n\n        returns\n        -------\n        ('NA', 'NA', 'NA', dataframe/series)\n        "
        if len(df) > 0:
            yield (b'NA', b'NA', b'NA', df)

    def to_range(self, start, end):
        if False:
            i = 10
            return i + 15
        '\n        returns a RangeObject from start/end sentinels.\n\n        returns\n        -------\n        string\n        '
        return b'NA'

    def chunk_to_str(self, chunk_id):
        if False:
            for i in range(10):
                print('nop')
        '\n        Converts parts of a chunk range (start or end) to a string\n\n        returns\n        -------\n        string\n        '
        return b'NA'

    def to_mongo(self, range_obj):
        if False:
            for i in range(10):
                print('nop')
        '\n        returns mongo query against range object.\n        since range object is not valid, returns empty dict\n\n        returns\n        -------\n        string\n        '
        return {}

    def filter(self, data, range_obj):
        if False:
            print('Hello World!')
        '\n        ensures data is properly subset to the range in range_obj.\n        since range object is not valid, returns data\n\n        returns\n        -------\n        data\n        '
        return data

    def exclude(self, data, range_obj):
        if False:
            i = 10
            return i + 15
        '\n        Removes data within the bounds of the range object.\n        Since range object is not valid for this chunk type,\n        returns nothing\n\n        returns\n        -------\n        empty dataframe or series\n        '
        if isinstance(data, DataFrame):
            return DataFrame()
        else:
            return Series()