from metaflow_test import MetaflowTest, ExpectationFailed, steps, tag

class ProjectProductionTest(MetaflowTest):
    PRIORITY = 1
    HEADER = "\nimport os\n\nos.environ['METAFLOW_PRODUCTION'] = 'True'\n@project(name='project_prod')\n"

    @steps(0, ['singleton'], required=True)
    def step_single(self):
        if False:
            return 10
        pass

    @steps(1, ['all'])
    def step_all(self):
        if False:
            return 10
        from metaflow import current
        assert_equals(current.branch_name, 'prod')
        assert_equals(current.project_flow_name, 'project_prod.prod.ProjectProductionTestFlow')