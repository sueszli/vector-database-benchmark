"""Corpus in CSV format."""
from __future__ import with_statement
import logging
import csv
import itertools
from gensim import interfaces, utils
logger = logging.getLogger(__name__)

class CsvCorpus(interfaces.CorpusABC):
    """Corpus in CSV format.

    Notes
    -----
    The CSV delimiter, headers etc. are guessed automatically based on the file content.
    All row values are expected to be ints/floats.

    """

    def __init__(self, fname, labels):
        if False:
            print('Hello World!')
        '\n\n        Parameters\n        ----------\n        fname : str\n            Path to corpus.\n        labels : bool\n            If True - ignore first column (class labels).\n\n        '
        logger.info('loading corpus from %s', fname)
        self.fname = fname
        self.length = None
        self.labels = labels
        with utils.open(self.fname, 'rb') as f:
            head = ''.join(itertools.islice(f, 5))
        self.headers = csv.Sniffer().has_header(head)
        self.dialect = csv.Sniffer().sniff(head)
        logger.info('sniffed CSV delimiter=%r, headers=%s', self.dialect.delimiter, self.headers)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        'Iterate over the corpus, returning one BoW vector at a time.\n\n        Yields\n        ------\n        list of (int, float)\n            Document in BoW format.\n\n        '
        with utils.open(self.fname, 'rb') as f:
            reader = csv.reader(f, self.dialect)
            if self.headers:
                next(reader)
            line_no = -1
            for (line_no, line) in enumerate(reader):
                if self.labels:
                    line.pop(0)
                yield list(enumerate((float(x) for x in line)))
            self.length = line_no + 1