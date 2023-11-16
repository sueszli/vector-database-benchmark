from collections import OrderedDict
import numpy as np
import pyqtgraph as pg
app = pg.mkQApp()
listOfTuples = [('text_%d' % i, i, i / 9.0) for i in range(12)]
listOfLists = [list(row) for row in listOfTuples]
plainArray = np.array(listOfLists, dtype=object)
recordArray = np.array(listOfTuples, dtype=[('string', object), ('integer', int), ('floating', float)])
dictOfLists = OrderedDict([(name, list(recordArray[name])) for name in recordArray.dtype.names])
listOfDicts = [OrderedDict([(name, rec[name]) for name in recordArray.dtype.names]) for rec in recordArray]
transposed = [[row[col] for row in listOfTuples] for col in range(len(listOfTuples[0]))]

def assertTableData(table, data):
    if False:
        print('Hello World!')
    assert len(data) == table.rowCount()
    rows = list(range(table.rowCount()))
    columns = list(range(table.columnCount()))
    for r in rows:
        assert len(data[r]) == table.columnCount()
        row = []
        for c in columns:
            item = table.item(r, c)
            if item is not None:
                row.append(item.value)
            else:
                row.append(None)
        assert row == list(data[r])

def test_TableWidget():
    if False:
        for i in range(10):
            print('nop')
    w = pg.TableWidget(sortable=False)
    w.setData(listOfTuples)
    assertTableData(w, listOfTuples)
    w.setData(listOfLists)
    assertTableData(w, listOfTuples)
    w.setData(plainArray)
    assertTableData(w, listOfTuples)
    w.setData(recordArray)
    assertTableData(w, listOfTuples)
    w.setData(dictOfLists)
    assertTableData(w, transposed)
    w.appendData(dictOfLists)
    assertTableData(w, transposed * 2)
    w.setData(listOfDicts)
    assertTableData(w, listOfTuples)
    w.appendData(listOfDicts)
    assertTableData(w, listOfTuples * 2)
    w.setData(listOfTuples)
    w.sortByColumn(0, pg.QtCore.Qt.SortOrder.AscendingOrder)
    assertTableData(w, sorted(listOfTuples, key=lambda a: a[0]))
    w.sortByColumn(1, pg.QtCore.Qt.SortOrder.AscendingOrder)
    assertTableData(w, sorted(listOfTuples, key=lambda a: a[1]))
    w.sortByColumn(2, pg.QtCore.Qt.SortOrder.AscendingOrder)
    assertTableData(w, sorted(listOfTuples, key=lambda a: a[2]))
    w.setSortMode(1, 'text')
    w.sortByColumn(1, pg.QtCore.Qt.SortOrder.AscendingOrder)
    assertTableData(w, sorted(listOfTuples, key=lambda a: str(a[1])))
    w.setSortMode(1, 'index')
    w.sortByColumn(1, pg.QtCore.Qt.SortOrder.AscendingOrder)
    assertTableData(w, listOfTuples)
    item = w.item(0, 2)
    assert item.text() == '%0.3g' % item.value
    w.setFormat('%0.6f')
    assert item.text() == '%0.6f' % item.value
    w.setFormat('X%0.7f', column=2)
    assert isinstance(item.value, float)
    assert item.text() == 'X%0.7f' % item.value
    w.setFormat('X%0.7f', column=3)
    w.appendRow(('x', 10, 7.3))
    item = w.item(w.rowCount() - 1, 2)
    assert isinstance(item.value, float)
    assert item.text() == 'X%0.7f' % item.value
    w.setFormat(None, column=2)
    assert isinstance(item.value, float)
    assert item.text() == '%0.6f' % item.value
    w.setFormat(None)
    assert isinstance(item.value, float)
    assert item.text() == '%0.3g' % item.value

    def fmt(item):
        if False:
            return 10
        if isinstance(item.value, float):
            return '%d %f' % (item.index, item.value)
        else:
            return str(item.value)
    w.setFormat(fmt)
    assert isinstance(item.value, float)
    assert isinstance(item.index, int)
    assert item.text() == '%d %f' % (item.index, item.value)