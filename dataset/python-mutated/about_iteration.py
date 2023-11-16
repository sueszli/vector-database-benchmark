from runner.koan import *

class AboutIteration(Koan):

    def test_iterators_are_a_type(self):
        if False:
            print('Hello World!')
        it = iter(range(1, 6))
        total = 0
        for num in it:
            total += num
        self.assertEqual(__, total)

    def test_iterating_with_next(self):
        if False:
            for i in range(10):
                print('nop')
        stages = iter(['alpha', 'beta', 'gamma'])
        try:
            self.assertEqual(__, next(stages))
            next(stages)
            self.assertEqual(__, next(stages))
            next(stages)
        except StopIteration as ex:
            err_msg = 'Ran out of iterations'
        self.assertRegex(err_msg, __)

    def add_ten(self, item):
        if False:
            i = 10
            return i + 15
        return item + 10

    def test_map_transforms_elements_of_a_list(self):
        if False:
            return 10
        seq = [1, 2, 3]
        mapped_seq = list()
        mapping = map(self.add_ten, seq)
        self.assertNotEqual(list, mapping.__class__)
        self.assertEqual(__, mapping.__class__)
        for item in mapping:
            mapped_seq.append(item)
        self.assertEqual(__, mapped_seq)

    def test_filter_selects_certain_items_from_a_list(self):
        if False:
            while True:
                i = 10

        def is_even(item):
            if False:
                return 10
            return item % 2 == 0
        seq = [1, 2, 3, 4, 5, 6]
        even_numbers = list()
        for item in filter(is_even, seq):
            even_numbers.append(item)
        self.assertEqual(__, even_numbers)

    def test_filter_returns_all_items_matching_criterion(self):
        if False:
            for i in range(10):
                print('nop')

        def is_big_name(item):
            if False:
                print('Hello World!')
            return len(item) > 4
        names = ['Jim', 'Bill', 'Clarence', 'Doug', 'Eli', 'Elizabeth']
        iterator = filter(is_big_name, names)
        self.assertEqual(__, next(iterator))
        self.assertEqual(__, next(iterator))
        try:
            next(iterator)
            pass
        except StopIteration:
            msg = 'Ran out of big names'
        self.assertEquals(__, msg)

    def add(self, accum, item):
        if False:
            print('Hello World!')
        return accum + item

    def multiply(self, accum, item):
        if False:
            i = 10
            return i + 15
        return accum * item

    def test_reduce_will_blow_your_mind(self):
        if False:
            print('Hello World!')
        import functools
        result = functools.reduce(self.add, [2, 3, 4])
        self.assertEqual(__, result.__class__)
        self.assertEqual(__, result)
        result2 = functools.reduce(self.multiply, [2, 3, 4], 1)
        self.assertEqual(__, result2)

    def test_use_pass_for_iterations_with_no_body(self):
        if False:
            for i in range(10):
                print('nop')
        for num in range(1, 5):
            pass
        self.assertEqual(__, num)

    def test_all_iteration_methods_work_on_any_sequence_not_just_lists(self):
        if False:
            print('Hello World!')
        result = map(self.add_ten, range(1, 4))
        self.assertEqual(__, list(result))

    def test_lines_in_a_file_are_iterable_sequences_too(self):
        if False:
            return 10

        def make_upcase(line):
            if False:
                print('Hello World!')
            return line.strip().title()
        file = open('example_file.txt')
        upcase_lines = map(make_upcase, file.readlines())
        self.assertEqual(__, list(upcase_lines))
        file.close()