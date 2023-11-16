from metaflow_test import MetaflowTest, ExpectationFailed, steps, tag

class ProjectBranchTest(MetaflowTest):
    PRIORITY = 1
    HEADER = "\nimport os\n\nos.environ['METAFLOW_BRANCH'] = 'this_is_a_test_branch'\n@project(name='project_branch')\n"

    @steps(0, ['singleton'], required=True)
    def step_single(self):
        if False:
            while True:
                i = 10
        pass

    @steps(1, ['all'])
    def step_all(self):
        if False:
            while True:
                i = 10
        from metaflow import current
        assert_equals(current.branch_name, 'test.this_is_a_test_branch')
        assert_equals(current.project_flow_name, 'project_branch.test.this_is_a_test_branch.ProjectBranchTestFlow')