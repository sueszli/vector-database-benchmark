START = 's'
END = 'e'

class Chunker(object):

    def to_chunks(self, data, **kwargs):
        if False:
            print('Hello World!')
        '\n        Chunks data. keyword args passed in from write API\n\n        returns\n        -------\n        generator that produces 4-tuples\n            (chunk start index/marker/key,\n            chunk end index/marker/key,\n            chunk_size,\n            chunked data)\n        '
        raise NotImplementedError

    def to_range(self, start, end):
        if False:
            for i in range(10):
                print('nop')
        '\n        takes start, end from to_chunks and returns a "range" that can be used\n        as the argument to methods require a chunk_range\n\n        returns\n        -------\n        A range object (dependent on type of chunker)\n        '
        raise NotImplementedError

    def to_mongo(self, range_obj):
        if False:
            return 10
        '\n        takes the range object used for this chunker type\n        and converts it into a string that can be use for a\n        mongo query that filters by the range\n\n        returns\n        -------\n        dict\n        '
        raise NotImplementedError

    def filter(self, data, range_obj):
        if False:
            print('Hello World!')
        '\n        ensures data is properly subset to the range in range_obj.\n        (Depending on how the chunking is implemented, it might be possible\n        to specify a chunk range that reads out more than the actual range\n        eg: date range, chunked monthly. read out 2016-01-01 to 2016-01-02.\n        This will read ALL of January 2016 but it should be subset to just\n        the first two days)\n\n        returns\n        -------\n        data, filtered by range_obj\n        '
        raise NotImplementedError

    def exclude(self, data, range_obj):
        if False:
            for i in range(10):
                print('nop')
        '\n        Removes data within the bounds of the range object (inclusive)\n\n        returns\n        -------\n        data, filtered by range_obj\n        '
        raise NotImplementedError

    def chunk_to_str(self, chunk_id):
        if False:
            print('Hello World!')
        '\n        Converts parts of a chunk range (start or end) to a string. These\n        chunk ids/indexes/markers are produced by to_chunks.\n        (See to_chunks)\n\n        returns\n        -------\n        string\n        '
        raise NotImplementedError