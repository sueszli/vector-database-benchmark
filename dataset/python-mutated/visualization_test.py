"""Tests for dragnn.python.visualization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.platform import googletest
from dragnn.protos import spec_pb2
from dragnn.protos import trace_pb2
from dragnn.python import visualization

def _get_trace_proto_string():
    if False:
        for i in range(10):
            print('nop')
    trace = trace_pb2.MasterTrace()
    trace.component_trace.add(step_trace=[trace_pb2.ComponentStepTrace(fixed_feature_trace=[])], name='零件')
    return trace.SerializeToString()

def _get_master_spec():
    if False:
        i = 10
        return i + 15
    return spec_pb2.MasterSpec(component=[spec_pb2.ComponentSpec(name='jalapeño')])

class VisualizationTest(googletest.TestCase):

    def testCanFindScript(self):
        if False:
            print('Hello World!')
        script = visualization._load_viz_script()
        self.assertIsInstance(script, str)
        self.assertTrue(10000.0 < len(script) < 10000000.0, 'Script size should be between 10k and 10M')

    def testSampleTraceSerialization(self):
        if False:
            return 10
        json = visualization.parse_trace_json(_get_trace_proto_string())
        self.assertIsInstance(json, str)
        self.assertTrue('component_trace' in json)

    def testInteractiveVisualization(self):
        if False:
            for i in range(10):
                print('nop')
        widget = visualization.InteractiveVisualization()
        widget.initial_html()
        widget.show_trace(_get_trace_proto_string())

    def testMasterSpecJson(self):
        if False:
            for i in range(10):
                print('nop')
        visualization.trace_html(_get_trace_proto_string(), master_spec=_get_master_spec())
        widget = visualization.InteractiveVisualization()
        widget.initial_html()
        widget.show_trace(_get_trace_proto_string(), master_spec=_get_master_spec())
if __name__ == '__main__':
    googletest.main()