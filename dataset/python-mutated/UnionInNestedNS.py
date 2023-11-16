class UnionInNestedNS(object):
    NONE = 0
    TableInNestedNS = 1

def UnionInNestedNSCreator(unionType, table):
    if False:
        while True:
            i = 10
    from flatbuffers.table import Table
    if not isinstance(table, Table):
        return None
    if unionType == UnionInNestedNS().TableInNestedNS:
        return TableInNestedNST.InitFromBuf(table.Bytes, table.Pos)
    return None