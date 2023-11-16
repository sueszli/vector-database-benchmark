import os
import tempfile
import tensorflow.compat.v1 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common_v1

def write_vocabulary_file(vocabulary):
    if False:
        while True:
            i = 10
    'Write temporary vocab file for module construction.'
    tmpdir = tempfile.mkdtemp()
    vocabulary_file = os.path.join(tmpdir, 'tokens.txt')
    with tf.io.gfile.GFile(vocabulary_file, 'w') as f:
        for entry in vocabulary:
            f.write(entry + '\n')
    return vocabulary_file

def test():
    if False:
        i = 10
        return i + 15
    vocabulary_file = write_vocabulary_file(['cat', 'is', 'on', 'the', 'mat'])
    table_initializer = tf.lookup.TextFileInitializer(vocabulary_file, tf.string, tf.lookup.TextFileIndex.WHOLE_LINE, tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER)
    table = tf.lookup.StaticVocabularyTable(table_initializer, num_oov_buckets=10)
    vocab_file_tensor = tf.convert_to_tensor(vocabulary_file, tf.string, name='asset_filepath')
    tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, vocab_file_tensor)
    x = tf.placeholder(tf.string, shape=(), name='input')
    r = table.lookup(x)
    tensor_info_x = tf.compat.v1.saved_model.utils.build_tensor_info(x)
    tensor_info_r = tf.compat.v1.saved_model.utils.build_tensor_info(r)
    return ({'key': tf.compat.v1.saved_model.signature_def_utils.build_signature_def(inputs={'x': tensor_info_x}, outputs={'r': tensor_info_r}, method_name='some_function')}, tf.tables_initializer(), tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS))
if __name__ == '__main__':
    common_v1.set_tf_options()
    common_v1.do_test(test, use_lite=True)