"""AlignIO support module (not for general use).

Unless you are writing a new parser or writer for Bio.AlignIO, you should not
use this module.  It provides base classes to try and simplify things.
"""

class AlignmentIterator:
    """Base class for building MultipleSeqAlignment iterators.

    You should write a next() method to return Alignment
    objects.  You may wish to redefine the __init__
    method as well.
    """

    def __init__(self, handle, seq_count=None):
        if False:
            while True:
                i = 10
        'Create an AlignmentIterator object.\n\n        Arguments:\n         - handle   - input file\n         - count    - optional, expected number of records per alignment\n           Recommend for fasta file format.\n\n        Note when subclassing:\n         - there should be a single non-optional argument, the handle,\n           and optional count IN THAT ORDER.\n         - you can add additional optional arguments.\n\n        '
        self.handle = handle
        self.records_per_alignment = seq_count

    def __next__(self):
        if False:
            print('Hello World!')
        'Return the next alignment in the file.\n\n        This method should be replaced by any derived class to do something\n        useful.\n        '
        raise NotImplementedError('This object should be subclassed')

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        'Iterate over the entries as MultipleSeqAlignment objects.\n\n        Example usage for (concatenated) PHYLIP files::\n\n            with open("many.phy","r") as myFile:\n                for alignment in PhylipIterator(myFile):\n                    print("New alignment:")\n                    for record in alignment:\n                        print(record.id)\n                        print(record.seq)\n\n        '
        return iter(self.__next__, None)

class AlignmentWriter:
    """Base class for building MultipleSeqAlignment writers.

    You should write a write_alignment() method.
    You may wish to redefine the __init__ method as well.
    """

    def __init__(self, handle):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the class.'
        self.handle = handle

    def write_file(self, alignments):
        if False:
            i = 10
            return i + 15
        'Use this to write an entire file containing the given alignments.\n\n        Arguments:\n         - alignments - A list or iterator returning MultipleSeqAlignment objects\n\n        In general, this method can only be called once per file.\n\n        This method should be replaced by any derived class to do something\n        useful.  It should return the number of alignments..\n        '
        raise NotImplementedError('This object should be subclassed')

    def clean(self, text):
        if False:
            while True:
                i = 10
        'Use this to avoid getting newlines in the output.'
        return text.replace('\n', ' ').replace('\r', ' ')

class SequentialAlignmentWriter(AlignmentWriter):
    """Base class for building MultipleSeqAlignment writers.

    This assumes each alignment can be simply appended to the file.
    You should write a write_alignment() method.
    You may wish to redefine the __init__ method as well.
    """

    def __init__(self, handle):
        if False:
            return 10
        'Initialize the class.'
        self.handle = handle

    def write_file(self, alignments):
        if False:
            return 10
        'Use this to write an entire file containing the given alignments.\n\n        Arguments:\n         - alignments - A list or iterator returning MultipleSeqAlignment objects\n\n        In general, this method can only be called once per file.\n        '
        self.write_header()
        count = 0
        for alignment in alignments:
            self.write_alignment(alignment)
            count += 1
        self.write_footer()
        return count

    def write_header(self):
        if False:
            return 10
        'Use this to write any header.\n\n        This method should be replaced by any derived class to do something\n        useful.\n        '

    def write_footer(self):
        if False:
            for i in range(10):
                print('nop')
        'Use this to write any footer.\n\n        This method should be replaced by any derived class to do something\n        useful.\n        '

    def write_alignment(self, alignment):
        if False:
            while True:
                i = 10
        'Use this to write a single alignment.\n\n        This method should be replaced by any derived class to do something\n        useful.\n        '
        raise NotImplementedError('This object should be subclassed')