from metaflow_test import MetaflowTest, ExpectationFailed, steps

class MergeArtifactsTest(MetaflowTest):
    PRIORITY = 1

    @steps(0, ['start'])
    def start(self):
        if False:
            i = 10
            return i + 15
        self.non_modified_passdown = 'a'
        self.modified_to_same_value = 'b'
        self.manual_merge_required = 'c'
        self.ignore_me = 'd'

    @steps(2, ['linear'])
    def modify_things(self):
        if False:
            return 10
        from metaflow.current import current
        self.manual_merge_required = current.task_id
        self.ignore_me = current.task_id
        self.modified_to_same_value = 'e'
        assert_equals(self.non_modified_passdown, 'a')

    @steps(0, ['join'], required=True)
    def merge_things(self, inputs):
        if False:
            while True:
                i = 10
        from metaflow.current import current
        from metaflow.exception import UnhandledInMergeArtifactsException, MetaflowException
        assert_exception(lambda : self.merge_artifacts(inputs), UnhandledInMergeArtifactsException)
        assert not hasattr(self, 'non_modified_passdown')
        assert not hasattr(self, 'manual_merge_required')
        assert_exception(lambda : self.merge_artifacts(inputs, exclude=['ignore_me'], include=['non_modified_passdown']), MetaflowException)
        assert not hasattr(self, 'non_modified_passdown')
        assert not hasattr(self, 'manual_merge_required')
        self.manual_merge_required = current.task_id
        self.merge_artifacts(inputs, exclude=['ignore_me'])
        assert_equals(self.non_modified_passdown, 'a')
        assert_equals(self.modified_to_same_value, 'e')
        assert_equals(self.manual_merge_required, current.task_id)
        assert not hasattr(self, 'ignore_me')

    @steps(0, ['end'])
    def end(self):
        if False:
            print('Hello World!')
        from metaflow.exception import MetaflowException
        assert_exception(lambda : self.merge_artifacts([]), MetaflowException)
        assert_equals(self.non_modified_passdown, 'a')
        assert_equals(self.modified_to_same_value, 'e')
        assert hasattr(self, 'manual_merge_required')

    @steps(3, ['all'])
    def step_all(self):
        if False:
            i = 10
            return i + 15
        assert_equals(self.non_modified_passdown, 'a')