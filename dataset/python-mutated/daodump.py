import win32com.client

def DumpDB(db, bDeep=1):
    if False:
        for i in range(10):
            print('nop')
    DumpTables(db, bDeep)
    DumpRelations(db, bDeep)
    DumpAllContainers(db, bDeep)

def DumpTables(db, bDeep=1):
    if False:
        print('Hello World!')
    for tab in db.TableDefs:
        tab = db.TableDefs(tab.Name)
        print('Table %s - Fields: %d, Attributes:%d' % (tab.Name, len(tab.Fields), tab.Attributes))
        if bDeep:
            DumpFields(tab.Fields)

def DumpFields(fields):
    if False:
        return 10
    for field in fields:
        print('  %s, size=%d, reqd=%d, type=%d, defVal=%s' % (field.Name, field.Size, field.Required, field.Type, str(field.DefaultValue)))

def DumpRelations(db, bDeep=1):
    if False:
        for i in range(10):
            print('nop')
    for relation in db.Relations:
        print(f'Relation {relation.Name} - {relation.Table}->{relation.ForeignTable}')

def DumpAllContainers(db, bDeep=1):
    if False:
        print('Hello World!')
    for cont in db.Containers:
        print('Container %s - %d documents' % (cont.Name, len(cont.Documents)))
        if bDeep:
            DumpContainerDocuments(cont)

def DumpContainerDocuments(container):
    if False:
        return 10
    for doc in container.Documents:
        import time
        timeStr = time.ctime(int(doc.LastUpdated))
        print(f'  {doc.Name} - updated {timeStr} (', end=' ')
        print(doc.LastUpdated, ')')

def TestEngine(engine):
    if False:
        return 10
    import sys
    if len(sys.argv) > 1:
        dbName = sys.argv[1]
    else:
        dbName = 'e:\\temp\\TestPython.mdb'
    db = engine.OpenDatabase(dbName)
    DumpDB(db)

def test():
    if False:
        for i in range(10):
            print('nop')
    for progid in ('DAO.DBEngine.36', 'DAO.DBEngine.35', 'DAO.DBEngine.30'):
        try:
            ob = win32com.client.gencache.EnsureDispatch(progid)
        except pythoncom.com_error:
            print(progid, 'does not seem to be installed')
        else:
            TestEngine(ob)
            break
if __name__ == '__main__':
    test()