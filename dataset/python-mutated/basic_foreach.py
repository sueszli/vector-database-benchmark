from metaflow_test import MetaflowTest, ExpectationFailed, steps

class BasicForeachTest(MetaflowTest):
    PRIORITY = 0

    @steps(0, ['foreach-split'], required=True)
    def split(self):
        if False:
            i = 10
            return i + 15
        self.my_index = None
        self.arr = range(32)

    @steps(0, ['foreach-inner'], required=True)
    def inner(self):
        if False:
            for i in range(10):
                print('nop')
        if self.my_index is None:
            self.my_index = self.index
        assert_equals(self.my_index, self.index)
        assert_equals(self.input, self.arr[self.index])
        self.my_input = self.input

    @steps(0, ['foreach-join'], required=True)
    def join(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        got = sorted([inp.my_input for inp in inputs])
        assert_equals(list(range(32)), got)

    @steps(1, ['all'])
    def step_all(self):
        if False:
            while True:
                i = 10
        pass