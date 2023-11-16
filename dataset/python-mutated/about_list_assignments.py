from runner.koan import *

class AboutListAssignments(Koan):

    def test_non_parallel_assignment(self):
        if False:
            for i in range(10):
                print('nop')
        names = ['John', 'Smith']
        self.assertEqual(__, names)

    def test_parallel_assignments(self):
        if False:
            print('Hello World!')
        (first_name, last_name) = ['John', 'Smith']
        self.assertEqual(__, first_name)
        self.assertEqual(__, last_name)

    def test_parallel_assignments_with_extra_values(self):
        if False:
            i = 10
            return i + 15
        (title, *first_names, last_name) = ['Sir', 'Ricky', 'Bobby', 'Worthington']
        self.assertEqual(__, title)
        self.assertEqual(__, first_names)
        self.assertEqual(__, last_name)

    def test_parallel_assignments_with_fewer_values(self):
        if False:
            for i in range(10):
                print('nop')
        (title, *first_names, last_name) = ['Mr', 'Bond']
        self.assertEqual(__, title)
        self.assertEqual(__, first_names)
        self.assertEqual(__, last_name)

    def test_parallel_assignments_with_sublists(self):
        if False:
            i = 10
            return i + 15
        (first_name, last_name) = [['Willie', 'Rae'], 'Johnson']
        self.assertEqual(__, first_name)
        self.assertEqual(__, last_name)

    def test_swapping_with_parallel_assignment(self):
        if False:
            while True:
                i = 10
        first_name = 'Roy'
        last_name = 'Rob'
        (first_name, last_name) = (last_name, first_name)
        self.assertEqual(__, first_name)
        self.assertEqual(__, last_name)