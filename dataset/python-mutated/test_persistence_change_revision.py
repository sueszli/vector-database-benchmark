from __future__ import absolute_import
import uuid
from st2common.exceptions import db as db_exc
from st2tests import DbTestCase
from tests.unit.base import ChangeRevFakeModel, ChangeRevFakeModelDB

class TestChangeRevision(DbTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        super(TestChangeRevision, cls).setUpClass()
        cls.access = ChangeRevFakeModel()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        ChangeRevFakeModelDB.drop_collection()
        super(TestChangeRevision, self).tearDown()

    def test_crud(self):
        if False:
            while True:
                i = 10
        initial = ChangeRevFakeModelDB(name=uuid.uuid4().hex, context={'a': 1})
        created = self.access.add_or_update(initial)
        self.assertEqual(initial.rev, 1)
        doc_id = created.id
        retrieved = self.access.get_by_id(doc_id)
        self.assertDictEqual(created.context, retrieved.context)
        retrieved = self.access.update(retrieved, context={'a': 2})
        updated = self.access.get_by_id(doc_id)
        self.assertNotEqual(created.rev, updated.rev)
        self.assertEqual(retrieved.rev, updated.rev)
        self.assertDictEqual(retrieved.context, updated.context)
        retrieved.context = {'a': 1, 'b': 2}
        retrieved = self.access.add_or_update(retrieved)
        updated = self.access.get_by_id(doc_id)
        self.assertNotEqual(created.rev, updated.rev)
        self.assertEqual(retrieved.rev, updated.rev)
        self.assertDictEqual(retrieved.context, updated.context)
        created.delete()
        self.assertRaises(db_exc.StackStormDBObjectNotFoundError, self.access.get_by_id, doc_id)

    def test_write_conflict(self):
        if False:
            return 10
        initial = ChangeRevFakeModelDB(name=uuid.uuid4().hex, context={'a': 1})
        created = self.access.add_or_update(initial)
        self.assertEqual(initial.rev, 1)
        doc_id = created.id
        retrieved1 = self.access.get_by_id(doc_id)
        retrieved2 = self.access.get_by_id(doc_id)
        retrieved1 = self.access.update(retrieved1, context={'a': 2})
        updated = self.access.get_by_id(doc_id)
        self.assertNotEqual(created.rev, updated.rev)
        self.assertEqual(retrieved1.rev, updated.rev)
        self.assertDictEqual(retrieved1.context, updated.context)
        self.assertRaises(db_exc.StackStormDBObjectWriteConflictError, self.access.update, retrieved2, context={'a': 1, 'b': 2})