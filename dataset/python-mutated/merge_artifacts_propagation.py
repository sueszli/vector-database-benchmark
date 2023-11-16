from metaflow_test import MetaflowTest, ExpectationFailed, steps

class MergeArtifactsPropagationTest(MetaflowTest):
    PRIORITY = 1

    @steps(0, ['start'])
    def start(self):
        if False:
            for i in range(10):
                print('nop')
        self.non_modified_passdown = 'a'

    @steps(0, ['foreach-inner-small'], required=True)
    def modify_things(self):
        if False:
            while True:
                i = 10
        val = self.index
        setattr(self, 'var%d' % val, val)

    @steps(0, ['foreach-join-small'], required=True)
    def merge_things(self, inputs):
        if False:
            return 10
        self.merge_artifacts(inputs)
        assert_equals(self.non_modified_passdown, 'a')
        for (i, _) in enumerate(inputs):
            assert_equals(getattr(self, 'var%d' % i), i)

    @steps(1, ['all'])
    def step_all(self):
        if False:
            for i in range(10):
                print('nop')
        assert_equals(self.non_modified_passdown, 'a')