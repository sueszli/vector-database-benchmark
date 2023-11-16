import os
import sys
import pythoncom
import win32api
from win32com.client import Dispatch, constants, gencache

def CreateTestAccessDatabase(dbname=None):
    if False:
        while True:
            i = 10
    if dbname is None:
        dbname = os.path.join(win32api.GetTempPath(), 'COMTestSuiteTempDatabase.mdb')
    access = Dispatch('Access.Application')
    dbEngine = access.DBEngine
    workspace = dbEngine.Workspaces(0)
    try:
        os.unlink(dbname)
    except OSError:
        print('WARNING - Unable to delete old test database - expect a COM exception RSN!')
    newdb = workspace.CreateDatabase(dbname, constants.dbLangGeneral, constants.dbEncrypt)
    table = newdb.CreateTableDef('Test Table 1')
    table.Fields.Append(table.CreateField('First Name', constants.dbText))
    table.Fields.Append(table.CreateField('Last Name', constants.dbText))
    index = table.CreateIndex('UniqueIndex')
    index.Fields.Append(index.CreateField('First Name'))
    index.Fields.Append(index.CreateField('Last Name'))
    index.Unique = -1
    table.Indexes.Append(index)
    newdb.TableDefs.Append(table)
    table = newdb.CreateTableDef('Test Table 2')
    table.Fields.Append(table.CreateField('First Name', constants.dbText))
    table.Fields.Append(table.CreateField('Last Name', constants.dbText))
    newdb.TableDefs.Append(table)
    relation = newdb.CreateRelation('TestRelationship')
    relation.Table = 'Test Table 1'
    relation.ForeignTable = 'Test Table 2'
    field = relation.CreateField('First Name')
    field.ForeignName = 'First Name'
    relation.Fields.Append(field)
    field = relation.CreateField('Last Name')
    field.ForeignName = 'Last Name'
    relation.Fields.Append(field)
    relation.Attributes = constants.dbRelationDeleteCascade + constants.dbRelationUpdateCascade
    newdb.Relations.Append(relation)
    tab1 = newdb.OpenRecordset('Test Table 1')
    tab1.AddNew()
    tab1.Fields('First Name').Value = 'Mark'
    tab1.Fields('Last Name').Value = 'Hammond'
    tab1.Update()
    tab1.MoveFirst()
    bk = tab1.Bookmark
    tab1.AddNew()
    tab1.Fields('First Name').Value = 'Second'
    tab1.Fields('Last Name').Value = 'Person'
    tab1.Update()
    tab1.MoveLast()
    if tab1.Fields('First Name').Value != 'Second':
        raise RuntimeError('Unexpected record is last - makes bookmark test pointless!')
    tab1.Bookmark = bk
    if tab1.Bookmark != bk:
        raise RuntimeError('The bookmark data is not the same')
    if tab1.Fields('First Name').Value != 'Mark':
        raise RuntimeError('The bookmark did not reset the record pointer correctly')
    return dbname

def DoDumpAccessInfo(dbname):
    if False:
        while True:
            i = 10
    from . import daodump
    a = forms = None
    try:
        sys.stderr.write('Creating Access Application...\n')
        a = Dispatch('Access.Application')
        print('Opening database %s' % dbname)
        a.OpenCurrentDatabase(dbname)
        db = a.CurrentDb()
        daodump.DumpDB(db, 1)
        forms = a.Forms
        print('There are %d forms open.' % len(forms))
        reports = a.Reports
        print('There are %d reports open' % len(reports))
    finally:
        if not a is None:
            sys.stderr.write('Closing database\n')
            try:
                a.CloseCurrentDatabase()
            except pythoncom.com_error:
                pass

def GenerateSupport():
    if False:
        for i in range(10):
            print('nop')
    gencache.EnsureModule('{00025E01-0000-0000-C000-000000000046}', 0, 4, 0)
    gencache.EnsureDispatch('Access.Application')

def DumpAccessInfo(dbname):
    if False:
        i = 10
        return i + 15
    amod = gencache.GetModuleForProgID('Access.Application')
    dmod = gencache.GetModuleForProgID('DAO.DBEngine.35')
    if amod is None and dmod is None:
        DoDumpAccessInfo(dbname)
        GenerateSupport()
    else:
        sys.stderr.write('testAccess not doing dynamic test, as generated code already exists\n')
    DoDumpAccessInfo(dbname)

def test(dbname=None):
    if False:
        print('Hello World!')
    if dbname is None:
        try:
            GenerateSupport()
        except pythoncom.com_error:
            print('*** Can not import the MSAccess type libraries - tests skipped')
            return
        dbname = CreateTestAccessDatabase()
        print("A test database at '%s' was created" % dbname)
    DumpAccessInfo(dbname)
if __name__ == '__main__':
    from .util import CheckClean
    dbname = None
    if len(sys.argv) > 1:
        dbname = sys.argv[1]
    test(dbname)
    CheckClean()