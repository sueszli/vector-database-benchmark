from runner.koan import *

class AboutLists(Koan):

    def test_creating_lists(self):
        if False:
            print('Hello World!')
        empty_list = list()
        self.assertEqual(list, type(empty_list))
        self.assertEqual(__, len(empty_list))

    def test_list_literals(self):
        if False:
            print('Hello World!')
        nums = list()
        self.assertEqual([], nums)
        nums[0:] = [1]
        self.assertEqual([1], nums)
        nums[1:] = [2]
        self.assertListEqual([1, __], nums)
        nums.append(333)
        self.assertListEqual([1, 2, __], nums)

    def test_accessing_list_elements(self):
        if False:
            i = 10
            return i + 15
        noms = ['peanut', 'butter', 'and', 'jelly']
        self.assertEqual(__, noms[0])
        self.assertEqual(__, noms[3])
        self.assertEqual(__, noms[-1])
        self.assertEqual(__, noms[-3])

    def test_slicing_lists(self):
        if False:
            return 10
        noms = ['peanut', 'butter', 'and', 'jelly']
        self.assertEqual(__, noms[0:1])
        self.assertEqual(__, noms[0:2])
        self.assertEqual(__, noms[2:2])
        self.assertEqual(__, noms[2:20])
        self.assertEqual(__, noms[4:0])
        self.assertEqual(__, noms[4:100])
        self.assertEqual(__, noms[5:0])

    def test_slicing_to_the_edge(self):
        if False:
            i = 10
            return i + 15
        noms = ['peanut', 'butter', 'and', 'jelly']
        self.assertEqual(__, noms[2:])
        self.assertEqual(__, noms[:2])

    def test_lists_and_ranges(self):
        if False:
            print('Hello World!')
        self.assertEqual(range, type(range(5)))
        self.assertNotEqual([1, 2, 3, 4, 5], range(1, 6))
        self.assertEqual(__, list(range(5)))
        self.assertEqual(__, list(range(5, 9)))

    def test_ranges_with_steps(self):
        if False:
            return 10
        self.assertEqual(__, list(range(5, 3, -1)))
        self.assertEqual(__, list(range(0, 8, 2)))
        self.assertEqual(__, list(range(1, 8, 3)))
        self.assertEqual(__, list(range(5, -7, -4)))
        self.assertEqual(__, list(range(5, -8, -4)))

    def test_insertions(self):
        if False:
            while True:
                i = 10
        knight = ['you', 'shall', 'pass']
        knight.insert(2, 'not')
        self.assertEqual(__, knight)
        knight.insert(0, 'Arthur')
        self.assertEqual(__, knight)

    def test_popping_lists(self):
        if False:
            i = 10
            return i + 15
        stack = [10, 20, 30, 40]
        stack.append('last')
        self.assertEqual(__, stack)
        popped_value = stack.pop()
        self.assertEqual(__, popped_value)
        self.assertEqual(__, stack)
        popped_value = stack.pop(1)
        self.assertEqual(__, popped_value)
        self.assertEqual(__, stack)

    def test_making_queues(self):
        if False:
            while True:
                i = 10
        queue = [1, 2]
        queue.append('last')
        self.assertEqual(__, queue)
        popped_value = queue.pop(0)
        self.assertEqual(__, popped_value)
        self.assertEqual(__, queue)