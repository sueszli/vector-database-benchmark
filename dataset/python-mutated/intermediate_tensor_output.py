"""A model whose intermediate tensor is also used as a model output."""
import tensorflow as tf

@tf.function(input_signature=[tf.TensorSpec(shape=[1, 4, 4, 4], dtype=tf.float32), tf.TensorSpec(shape=[1, 4, 4, 4], dtype=tf.float32)])
def func(a, b):
    if False:
        return 10
    c = a + b
    d = c + a
    e = d + a
    f = e + a
    return (c, f)

def main():
    if False:
        for i in range(10):
            print('nop')
    converter = tf.lite.TFLiteConverter.from_concrete_functions([func.get_concrete_function()])
    converter.target_spec = tf.lite.TargetSpec()
    tflite_model = converter.convert()
    model_path = '/tmp/intermediate_tensor_output.tflite'
    with open(model_path, 'wb') as f:
        f.write(tflite_model)
    print(f'TFLite model {model_path} is generated.\n')
if __name__ == '__main__':
    main()