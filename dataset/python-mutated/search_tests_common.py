"""Common code for SearchIO tests."""
import os
import gzip
import unittest
try:
    import sqlite3
except ImportError:
    sqlite3 = None
from Bio import SearchIO
from Bio.SeqRecord import SeqRecord

class SearchTestBaseClass(unittest.TestCase):

    def compare_attrs(self, obj_a, obj_b, attrs):
        if False:
            i = 10
            return i + 15
        'Compare attribute values of two objects.'
        for attr in attrs:
            if attr.startswith('_items'):
                continue
            val_a = getattr(obj_a, attr)
            val_b = getattr(obj_b, attr)
            if attr in ('_hit', '_query') and (val_a is not None and val_b is not None):
                if isinstance(val_a, SeqRecord) and isinstance(val_b, SeqRecord):
                    msg = f'Comparing attribute {attr}'
                    self.assertEqual(str(val_a.seq), str(val_b.seq), msg=msg)
                elif isinstance(val_a, list) and isinstance(val_b, list):
                    for (seq_a, seq_b) in zip(val_a, val_b):
                        msg = f'Comparing attribute {attr}'
                        self.assertEqual(str(seq_a.seq), str(seq_b.seq), msg=msg)
            else:
                self.assertIsInstance(val_b, type(val_a))
                msg = f'Comparing attribute {attr}'
                self.assertEqual(val_a, val_b)

    def compare_search_obj(self, obj_a, obj_b):
        if False:
            i = 10
            return i + 15
        'Compare attribute values of two QueryResult objects.'
        self.assertEqual(_num_difference(obj_a, obj_b), 0)
        self.compare_attrs(obj_a, obj_b, list(obj_a.__dict__))
        if not isinstance(obj_a, SearchIO.HSPFragment):
            msg = f'comparing {obj_a!r} vs {obj_b!r}'
            self.assertEqual(len(obj_a), len(obj_b), msg=msg)
            for (item_a, item_b) in zip(obj_a, obj_b):
                self.compare_search_obj(item_a, item_b)

class CheckRaw(unittest.TestCase):
    """Base class for testing index's get_raw method."""
    fmt = None

    def check_raw(self, filename, id, raw, **kwargs):
        if False:
            return 10
        'Index filename using keyword arguments, check get_raw(id)==raw.'
        idx = SearchIO.index(filename, self.fmt, **kwargs)
        raw = raw.encode()
        new = idx.get_raw(id)
        self.assertIsInstance(new, bytes, f"Didn't get bytes from {self.fmt} get_raw")
        self.assertEqual(raw.replace(b'\r\n', b'\n'), new.replace(b'\r\n', b'\n'))
        idx.close()
        if sqlite3:
            idx = SearchIO.index_db(':memory:', filename, self.fmt, **kwargs)
            new = idx.get_raw(id)
            self.assertIsInstance(new, bytes, f"Didn't get bytes from {self.fmt} get_raw")
            self.assertEqual(raw.replace(b'\r\n', b'\n'), new.replace(b'\r\n', b'\n'))
            idx.close()
        if os.path.isfile(filename + '.bgz'):
            print(f'[BONUS {filename}.bgz]')
            self.check_raw(filename + '.bgz', id, raw, **kwargs)

class CheckIndex(SearchTestBaseClass):
    """Base class for testing indexing."""

    def check_index(self, filename, format, **kwargs):
        if False:
            print('Hello World!')
        if filename.endswith('.bgz'):
            with gzip.open(filename) as handle:
                parsed = list(SearchIO.parse(handle, format, **kwargs))
        else:
            parsed = list(SearchIO.parse(filename, format, **kwargs))
        indexed = SearchIO.index(filename, format, **kwargs)
        self.assertEqual(len(parsed), len(indexed), 'Should be %i records in %s, index says %i' % (len(parsed), filename, len(indexed)))
        if sqlite3 is not None:
            db_indexed = SearchIO.index_db(':memory:', [filename], format, **kwargs)
            self.assertEqual(len(parsed), len(db_indexed), 'Should be %i records in %s, index_db says %i' % (len(parsed), filename, len(db_indexed)))
        for qres in parsed:
            idx_qres = indexed[qres.id]
            self.assertNotEqual(id(qres), id(idx_qres))
            self.compare_search_obj(qres, idx_qres)
            if sqlite3 is not None:
                dbidx_qres = db_indexed[qres.id]
                self.assertNotEqual(id(qres), id(dbidx_qres))
                self.compare_search_obj(qres, dbidx_qres)
        indexed.close()
        if sqlite3 is not None:
            db_indexed.close()
            db_indexed._con.close()
        if os.path.isfile(filename + '.bgz'):
            print(f'[BONUS {filename}.bgz]')
            self.check_index(filename + '.bgz', format, **kwargs)

def _num_difference(obj_a, obj_b):
    if False:
        print('Hello World!')
    'Return the number of instance attributes present only in one object.'
    attrs_a = set(obj_a.__dict__)
    attrs_b = set(obj_b.__dict__)
    diff = attrs_a.symmetric_difference(attrs_b)
    privates = len([x for x in diff if x.startswith('_')])
    return len(diff) - privates