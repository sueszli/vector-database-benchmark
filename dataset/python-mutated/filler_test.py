from caffe2.python import core, test_util, workspace

class TestFiller(test_util.TestCase):

    def test_filler(self):
        if False:
            i = 10
            return i + 15
        net = core.Net('test_filler')
        net.Concat(['X0', 'X1', 'X2'], ['concat_out', 'split_info'])
        self.assertFalse(workspace.HasBlob('X0'))
        input_dim = (30, 20)
        workspace.FillRandomNetworkInputs(net, [[input_dim, input_dim, input_dim]], [['float', 'float', 'float']])
        self.assertTrue(workspace.HasBlob('X0'))
        self.assertEqual(workspace.FetchBlob('X0').shape, input_dim)
        with self.assertRaises(RuntimeError):
            workspace.FillRandomNetworkInputs(net, [[input_dim]], [['float']])