import json
from pyflink.testing.test_case_utils import PyFlinkStreamTableTestCase
from pyflink.table.explain_detail import ExplainDetail

class StreamTableExplainTests(PyFlinkStreamTableTestCase):

    def test_explain(self):
        if False:
            print('Hello World!')
        t = self.t_env.from_elements([(1, 'Hi', 'Hello')], ['a', 'b', 'c'])
        result = t.group_by(t.c).select(t.a.sum, t.c.alias('b')).explain(ExplainDetail.CHANGELOG_MODE, ExplainDetail.PLAN_ADVICE)
        assert isinstance(result, str)
        self.assertGreaterEqual(result.find('== Optimized Physical Plan With Advice =='), 0)
        self.assertGreaterEqual(result.find("advice[1]: [ADVICE] You might want to enable local-global two-phase optimization by configuring ('table.exec.mini-batch.enabled' to 'true', 'table.exec.mini-batch.allow-latency' to a positive long value, 'table.exec.mini-batch.size' to a positive long value)."), 0)
        result = t.group_by(t.c).select(t.a.sum, t.c.alias('b')).explain(ExplainDetail.JSON_EXECUTION_PLAN)
        assert isinstance(result, str)
        try:
            json.loads(result.split('== Physical Execution Plan ==')[1])
        except:
            self.fail('The execution plan of explain detail is not in json format.')
if __name__ == '__main__':
    import unittest
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)