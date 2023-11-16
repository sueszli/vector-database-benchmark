import logging
import unittest
import uuid
import pytest
from apache_beam.io.filesystems import FileSystems
from apache_beam.testing.test_pipeline import TestPipeline
try:
    import tfx_bsl
    import tensorflow as tf
    from apache_beam.examples.inference.tfx_bsl import tensorflow_image_classification
    from apache_beam.examples.inference.tfx_bsl.build_tensorflow_model import save_tf_model_with_signature
except ImportError as e:
    tfx_bsl = None
_EXPECTED_OUTPUTS = {'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005001.JPEG': '681', 'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005002.JPEG': '333', 'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005003.JPEG': '711', 'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005004.JPEG': '286', 'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005005.JPEG': '445', 'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005006.JPEG': '288', 'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005007.JPEG': '880', 'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005008.JPEG': '534', 'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005009.JPEG': '888', 'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005010.JPEG': '996', 'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005011.JPEG': '327', 'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005012.JPEG': '573'}

def process_outputs(filepath):
    if False:
        i = 10
        return i + 15
    with FileSystems().open(filepath) as f:
        lines = f.readlines()
    lines = [l.decode('utf-8').strip('\n') for l in lines]
    return lines

@unittest.skipIf(tfx_bsl is None, 'Missing dependencies. Test depends on tfx_bsl')
class TFXRunInferenceTests(unittest.TestCase):

    @pytest.mark.uses_tensorflow
    @pytest.mark.it_postcommit
    def test_tfx_run_inference_mobilenetv2(self):
        if False:
            return 10
        test_pipeline = TestPipeline(is_integration_test=True)
        model = tf.keras.applications.MobileNetV2(weights='imagenet')
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        path_to_save_model = 'gs://apache-beam-ml/models/tensorflow_models/mobilenet_v2'
        save_tf_model_with_signature(path_to_save_model, model, preprocess_input, training=False)
        file_of_image_names = 'gs://apache-beam-ml/testing/inputs/it_mobilenetv2_imagenet_validation_inputs.txt'
        output_file_dir = 'gs://apache-beam-ml/testing/predictions'
        output_file = '/'.join([output_file_dir, str(uuid.uuid4()), 'result.txt'])
        extra_opts = {'input': file_of_image_names, 'output': output_file, 'model_path': path_to_save_model}
        tensorflow_image_classification.run(test_pipeline.get_full_options_as_args(**extra_opts), save_main_session=False)
        self.assertEqual(FileSystems().exists(output_file), True)
        predictions = process_outputs(filepath=output_file)
        for prediction in predictions:
            (filename, prediction) = prediction.split(',')
            self.assertEqual(_EXPECTED_OUTPUTS[filename], prediction)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()