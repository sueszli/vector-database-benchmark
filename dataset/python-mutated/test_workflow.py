from odoo.tests import common

class test_workflows(common.TransactionCase):

    def check_activities(self, record, names):
        if False:
            print('Hello World!')
        ' Check that the record has workitems in the given activity names.\n        '
        Instance = self.env['workflow.instance']
        Workitem = self.env['workflow.workitem']
        instance = Instance.search([('res_type', '=', record._name), ('res_id', '=', record.id)])
        self.assertTrue(instance, 'A workflow instance is expected.')
        workitems = Workitem.search([('inst_id', '=', instance.id)])
        self.assertTrue(workitems, 'The workflow instance should have workitems.')
        self.assertEqual(sorted([item.act_id.name for item in workitems]), sorted(names))

    def check_value(self, record, value):
        if False:
            i = 10
            return i + 15
        ' Check that the record has the given value.\n        '
        self.assertEqual(record.value, value)

    def test_workflow(self):
        if False:
            print('Hello World!')
        model = self.env['test.workflow.model']
        trigger = self.env['test.workflow.trigger']
        record = model.create({})
        self.check_activities(record, ['a'])
        record.signal_workflow('a-b')
        self.check_activities(record, ['b'])
        record.trigger()
        self.check_activities(record, ['b'])
        trigger.browse(1).write({'value': True})
        record.trigger()
        self.check_activities(record, ['c'])
        self.assertEqual(True, True)

    def test_workflow_a(self):
        if False:
            while True:
                i = 10
        record = self.env['test.workflow.model.a'].create({})
        self.check_activities(record, ['a'])
        self.check_value(record, 0)

    def test_workflow_b(self):
        if False:
            print('Hello World!')
        record = self.env['test.workflow.model.b'].create({})
        self.check_activities(record, ['a'])
        self.check_value(record, 1)

    def test_workflow_c(self):
        if False:
            for i in range(10):
                print('nop')
        record = self.env['test.workflow.model.c'].create({})
        self.check_activities(record, ['a'])
        self.check_value(record, 0)

    def test_workflow_d(self):
        if False:
            print('Hello World!')
        record = self.env['test.workflow.model.d'].create({})
        self.check_activities(record, ['a'])
        self.check_value(record, 1)

    def test_workflow_e(self):
        if False:
            for i in range(10):
                print('nop')
        record = self.env['test.workflow.model.e'].create({})
        self.check_activities(record, ['b'])
        self.check_value(record, 2)

    def test_workflow_f(self):
        if False:
            print('Hello World!')
        record = self.env['test.workflow.model.f'].create({})
        self.check_activities(record, ['a'])
        self.check_value(record, 1)
        record.signal_workflow('a-b')
        self.check_activities(record, ['b'])
        self.check_value(record, 2)

    def test_workflow_g(self):
        if False:
            i = 10
            return i + 15
        record = self.env['test.workflow.model.g'].create({})
        self.check_activities(record, ['a'])
        self.check_value(record, 1)

    def test_workflow_h(self):
        if False:
            i = 10
            return i + 15
        record = self.env['test.workflow.model.h'].create({})
        self.check_activities(record, ['b', 'c'])
        self.check_value(record, 2)

    def test_workflow_i(self):
        if False:
            i = 10
            return i + 15
        record = self.env['test.workflow.model.i'].create({})
        self.check_activities(record, ['b'])
        self.check_value(record, 2)

    def test_workflow_j(self):
        if False:
            i = 10
            return i + 15
        record = self.env['test.workflow.model.j'].create({})
        self.check_activities(record, ['a'])
        self.check_value(record, 1)

    def test_workflow_k(self):
        if False:
            print('Hello World!')
        record = self.env['test.workflow.model.k'].create({})
        self.check_value(record, 2)

    def test_workflow_l(self):
        if False:
            for i in range(10):
                print('nop')
        record = self.env['test.workflow.model.l'].create({})
        self.check_activities(record, ['c', 'c', 'd'])
        self.check_value(record, 3)