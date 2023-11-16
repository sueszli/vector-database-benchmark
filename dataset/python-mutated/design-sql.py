import itertools

class SQL(object):

    def __init__(self, names, columns):
        if False:
            return 10
        '\n        :type names: List[str]\n        :type columns: List[int]\n        '
        self.__table = {name: [column] for (name, column) in itertools.izip(names, columns)}

    def insertRow(self, name, row):
        if False:
            return 10
        '\n        :type name: str\n        :type row: List[str]\n        :rtype: None\n        '
        row.append('')
        self.__table[name].append(row)

    def deleteRow(self, name, rowId):
        if False:
            print('Hello World!')
        '\n        :type name: str\n        :type rowId: int\n        :rtype: None\n        '
        self.__table[name][rowId][-1] = 'deleted'

    def selectCell(self, name, rowId, columnId):
        if False:
            print('Hello World!')
        '\n        :type name: str\n        :type rowId: int\n        :type columnId: int\n        :rtype: str\n        '
        return self.__table[name][rowId][columnId - 1] if self.__table[name][rowId][-1] == '' else ''