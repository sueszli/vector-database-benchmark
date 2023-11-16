from metaflow_test import MetaflowTest, ExpectationFailed, steps

class WideForeachTest(MetaflowTest):
    PRIORITY = 3

    @steps(0, ['foreach-split-small'], required=True)
    def split(self):
        if False:
            print('Hello World!')
        self.my_index = None
        self.arr = range(1200)

    @steps(0, ['foreach-inner-small'], required=True)
    def inner(self):
        if False:
            i = 10
            return i + 15
        self.my_input = self.input

    @steps(0, ['foreach-join-small'], required=True)
    def join(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        got = sorted([inp.my_input for inp in inputs])
        assert_equals(list(range(1200)), got)

    @steps(1, ['all'])
    def step_all(self):
        if False:
            while True:
                i = 10
        pass

    def check_results(self, flow, checker):
        if False:
            return 10
        run = checker.get_run()
        if run:
            res = sorted((task.data.my_input for task in run['foreach_inner']))
            assert_equals(list(range(1200)), res)