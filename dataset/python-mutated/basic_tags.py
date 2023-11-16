from metaflow_test import MetaflowTest, ExpectationFailed, steps

class BasicTagTest(MetaflowTest):
    """
    Test that tags are assigned properly.
    """
    PRIORITY = 2
    HEADER = "@project(name='basic_tag')"

    @steps(0, ['all'])
    def step_all(self):
        if False:
            return 10
        from metaflow import get_namespace
        import os
        user = 'user:%s' % os.environ.get('METAFLOW_USER')
        assert_equals(user, get_namespace())

    def check_results(self, flow, checker):
        if False:
            return 10
        import os
        from metaflow import namespace
        run = checker.get_run()
        if run is None:
            return
        flow_obj = run.parent
        tags = ('project:basic_tag', 'project_branch:user.tester', 'user:%s' % os.environ.get('METAFLOW_USER'), '刺身 means sashimi', 'multiple tags should be ok')
        for tag in tags:
            namespace(tag)
            run = flow_obj[checker.run_id]
            assert_equals(frozenset(), frozenset(flow_obj.tags))
            assert_equals([True] * len(tags), [t in run.tags for t in tags])
            assert_equals([], list(flow_obj.runs('not_a_tag')))
            assert_equals([], list(flow_obj.runs('not_a_tag', tag)))
            assert_equals(frozenset((step.name for step in flow)), frozenset((step.id.split('/')[-1] for step in run.steps(tag))))
            assert_equals(frozenset((step.name for step in flow)), frozenset((step.id.split('/')[-1] for step in run.steps(*tags))))
            for step in run:
                assert_equals([True] * len(tags), [t in step.tags for t in tags])
                assert_equals([], list(step.tasks('not_a_tag')))
                assert_equals([task.id for task in step], [task.id for task in step.tasks(tag)])
                for task in step.tasks(tag):
                    assert_equals([True] * len(tags), [t in task.tags for t in tags])
                    for data in task:
                        assert_equals([True] * len(tags), [t in data.tags for t in tags])