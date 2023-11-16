from metaflow_test import MetaflowTest, ExpectationFailed, steps

class NestedForeachTest(MetaflowTest):
    PRIORITY = 1

    @steps(0, ['foreach-nested-inner'], required=True)
    def inner(self):
        if False:
            return 10
        [x, y, z] = self.foreach_stack()
        assert_equals(len(self.x), x[1])
        assert_equals(len(self.y), y[1])
        assert_equals(len(self.z), z[1])
        assert_equals(x[2], self.x[x[0]])
        assert_equals(y[2], self.y[y[0]])
        assert_equals(z[2], self.z[z[0]])
        self.combo = x[2] + y[2] + z[2]

    @steps(1, ['all'])
    def step_all(self):
        if False:
            print('Hello World!')
        pass

    def check_results(self, flow, checker):
        if False:
            print('Hello World!')
        from itertools import product
        artifacts = checker.artifact_dict('foreach_inner', 'combo')
        got = sorted((val['combo'] for val in artifacts.values()))
        expected = sorted((''.join(p) for p in product('abc', 'de', 'fghijk')))
        assert_equals(expected, got)