from twisted.internet import defer
from buildbot.db.test_result_sets import TestResultSetAlreadyCompleted
from buildbot.test.fakedb.base import FakeDBComponent
from buildbot.test.fakedb.row import Row

class TestResultSet(Row):
    table = 'test_result_sets'
    id_column = 'id'
    foreignKeys = ('builderid', 'buildid', 'stepid')
    required_columns = ('builderid', 'buildid', 'stepid', 'category', 'value_unit', 'complete')

    def __init__(self, id=None, builderid=None, buildid=None, stepid=None, description=None, category=None, value_unit=None, tests_passed=None, tests_failed=None, complete=None):
        if False:
            while True:
                i = 10
        super().__init__(id=id, builderid=builderid, buildid=buildid, stepid=stepid, description=description, category=category, value_unit=value_unit, tests_passed=tests_passed, tests_failed=tests_failed, complete=complete)

class FakeTestResultSetsComponent(FakeDBComponent):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.result_sets = {}

    def insert_test_data(self, rows):
        if False:
            return 10
        for row in rows:
            if isinstance(row, TestResultSet):
                self.result_sets[row.id] = row.values.copy()

    def addTestResultSet(self, builderid, buildid, stepid, description, category, value_unit):
        if False:
            i = 10
            return i + 15
        id = Row.nextId()
        self.result_sets[id] = {'id': id, 'builderid': builderid, 'buildid': buildid, 'stepid': stepid, 'description': description, 'category': category, 'value_unit': value_unit, 'tests_failed': None, 'tests_passed': None, 'complete': False}
        return defer.succeed(id)

    def _row2dict(self, row):
        if False:
            while True:
                i = 10
        row = row.copy()
        row['complete'] = bool(row['complete'])
        return row

    def getTestResultSet(self, test_result_setid):
        if False:
            print('Hello World!')
        if test_result_setid not in self.result_sets:
            return defer.succeed(None)
        return defer.succeed(self._row2dict(self.result_sets[test_result_setid]))

    def getTestResultSets(self, builderid, buildid=None, stepid=None, complete=None, result_spec=None):
        if False:
            while True:
                i = 10
        ret = []
        for row in self.result_sets.values():
            if row['builderid'] != builderid:
                continue
            if buildid is not None and row['buildid'] != buildid:
                continue
            if stepid is not None and row['stepid'] != stepid:
                continue
            if complete is not None and row['complete'] != complete:
                continue
            ret.append(self._row2dict(row))
        if result_spec is not None:
            ret = self.applyResultSpec(ret, result_spec)
        return defer.succeed(ret)

    def completeTestResultSet(self, test_result_setid, tests_passed=None, tests_failed=None):
        if False:
            while True:
                i = 10
        if test_result_setid not in self.result_sets:
            raise TestResultSetAlreadyCompleted(f'Test result set {test_result_setid} is already completed or does not exist')
        row = self.result_sets[test_result_setid]
        if row['complete'] != 0:
            raise TestResultSetAlreadyCompleted(f'Test result set {test_result_setid} is already completed or does not exist')
        row['complete'] = 1
        if tests_passed is not None:
            row['tests_passed'] = tests_passed
        if tests_failed is not None:
            row['tests_failed'] = tests_failed
        return defer.succeed(None)