from perspective.table import Table
import psutil
import os

class TestDelete(object):

    def test_table_delete(self):
        if False:
            for i in range(10):
                print('nop')
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss
        for x in range(10000):
            data = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
            tbl = Table(data)
            tbl.delete()
        mem2 = process.memory_info().rss
        assert mem2 - mem < 2000000

    def test_table_delete_with_view(self, sentinel):
        if False:
            return 10
        data = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
        tbl = Table(data)
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss
        for x in range(10000):
            view = tbl.view()
            view.delete()
        tbl.delete()
        mem2 = process.memory_info().rss
        assert mem2 - mem < 2000000