import tensorflow.compat.v1 as tf
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('output_dir', '/tmp/matrix_half_plus_two/1', 'The directory where to write SavedModel files.')

def _generate_saved_model_for_matrix_half_plus_two(export_dir):
    if False:
        while True:
            i = 10
    'Creates SavedModel for half plus two model that accepts batches of\n       3*3 matrices.\n       The model divides all elements in each matrix by 2 and adds 2 to them.\n       So, for one input matrix [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\n       the result will be [[2.5, 3, 3.5], [4, 4.5, 5], [5.5, 6, 6.5]].\n    Args:\n      export_dir: The directory where to write SavedModel files.\n    '
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    with tf.Session() as session:
        x = tf.placeholder(tf.float32, shape=[None, 3, 3], name='x')
        a = tf.constant(0.5)
        b = tf.constant(2.0)
        y = tf.add(tf.multiply(a, x), b, name='y')
        predict_signature_def = tf.saved_model.signature_def_utils.predict_signature_def({'x': x}, {'y': y})
        signature_def_map = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: predict_signature_def}
        session.run(tf.global_variables_initializer())
        builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING], signature_def_map=signature_def_map)
        builder.save()

def main(_):
    if False:
        for i in range(10):
            print('nop')
    _generate_saved_model_for_matrix_half_plus_two(FLAGS.output_dir)
if __name__ == '__main__':
    tf.compat.v1.app.run()