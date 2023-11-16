class TupleMatchingMixin:

    def do_test_match(self, routingKey, shouldMatch, *tuples):
        if False:
            return 10
        raise NotImplementedError

    def test_simple_tuple_match(self):
        if False:
            for i in range(10):
                print('nop')
        return self.do_test_match(('abc',), True, ('abc',))

    def test_simple_tuple_no_match(self):
        if False:
            return 10
        return self.do_test_match(('abc',), False, ('def',))

    def test_multiple_tuple_match(self):
        if False:
            i = 10
            return i + 15
        return self.do_test_match(('a', 'b', 'c'), True, ('a', 'b', 'c'))

    def test_multiple_tuple_match_tuple_prefix(self):
        if False:
            print('Hello World!')
        return self.do_test_match(('a', 'b', 'c'), False, ('a', 'b'))

    def test_multiple_tuple_match_tuple_suffix(self):
        if False:
            i = 10
            return i + 15
        return self.do_test_match(('a', 'b', 'c'), False, ('b', 'c'))

    def test_multiple_tuple_match_rk_prefix(self):
        if False:
            print('Hello World!')
        return self.do_test_match(('a', 'b'), False, ('a', 'b', 'c'))

    def test_multiple_tuple_match_rk_suffix(self):
        if False:
            i = 10
            return i + 15
        return self.do_test_match(('b', 'c'), False, ('a', 'b', 'c'))

    def test_None_match(self):
        if False:
            return 10
        return self.do_test_match(('a', 'b', 'c'), True, ('a', None, 'c'))

    def test_None_match_empty(self):
        if False:
            i = 10
            return i + 15
        return self.do_test_match(('a', '', 'c'), True, ('a', None, 'c'))

    def test_None_no_match(self):
        if False:
            print('Hello World!')
        return self.do_test_match(('a', 'b', 'c'), False, ('a', None, 'x'))