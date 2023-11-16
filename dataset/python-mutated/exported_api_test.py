from tensorflow.python.platform import test
from tensorflow.python.tools.api.generator2.shared import exported_api
_EXPORTS = exported_api.ExportedApi(docs=[exported_api.ExportedDoc(file_name='tf/python/framework/tensor.py', line_no=0, modules=('tf',), docstring='This is a docstring')], symbols=[exported_api.ExportedSymbol(file_name='tf/python/framework/tensor.py', line_no=139, symbol_name='Tensor', v1_apis=('tf.Tensor',), v2_apis=('tf.Tensor', 'tf.experimental.numpy.ndarray')), exported_api.ExportedSymbol(file_name='tf/python/framework/tensor.py', line_no=770, symbol_name='Tensor', v1_apis=('tf.enable_tensor_equality',), v2_apis=())])

class ExportedApiTest(test.TestCase):

    def test_read_write(self):
        if False:
            for i in range(10):
                print('nop')
        filename = self.get_temp_dir() + '/test_write.json'
        _EXPORTS.write(filename)
        e = exported_api.ExportedApi()
        e.read(filename)
        self.assertEqual(e, _EXPORTS)
if __name__ == '__main__':
    test.main()