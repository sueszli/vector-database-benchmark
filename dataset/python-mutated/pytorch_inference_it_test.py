"""End-to-End test for Pytorch Inference"""
import logging
import os
import unittest
import uuid
import pytest
from apache_beam.io.filesystems import FileSystems
from apache_beam.testing.test_pipeline import TestPipeline
try:
    import torch
    from apache_beam.examples.inference import pytorch_image_classification
    from apache_beam.examples.inference import pytorch_image_segmentation
    from apache_beam.examples.inference import pytorch_model_per_key_image_segmentation
    from apache_beam.examples.inference import pytorch_language_modeling
except ImportError as e:
    torch = None
_EXPECTED_OUTPUTS = {'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005001.JPEG': '681', 'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005002.JPEG': '333', 'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005003.JPEG': '711', 'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005004.JPEG': '286', 'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005005.JPEG': '433', 'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005006.JPEG': '290', 'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005007.JPEG': '890', 'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005008.JPEG': '592', 'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005009.JPEG': '406', 'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005010.JPEG': '996', 'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005011.JPEG': '327', 'gs://apache-beam-ml/datasets/imagenet/raw-data/validation/ILSVRC2012_val_00005012.JPEG': '573'}

def process_outputs(filepath):
    if False:
        return 10
    with FileSystems().open(filepath) as f:
        lines = f.readlines()
    lines = [l.decode('utf-8').strip('\n') for l in lines]
    return lines

@unittest.skipIf(os.getenv('FORCE_TORCH_IT') is None and torch is None, 'Missing dependencies. Test depends on torch, torchvision, pillow, and transformers')
class PyTorchInference(unittest.TestCase):

    @pytest.mark.uses_pytorch
    @pytest.mark.it_postcommit
    def test_torch_run_inference_imagenet_mobilenetv2(self):
        if False:
            for i in range(10):
                print('nop')
        test_pipeline = TestPipeline(is_integration_test=True)
        file_of_image_names = 'gs://apache-beam-ml/testing/inputs/it_mobilenetv2_imagenet_validation_inputs.txt'
        output_file_dir = 'gs://apache-beam-ml/testing/predictions'
        output_file = '/'.join([output_file_dir, str(uuid.uuid4()), 'result.txt'])
        model_state_dict_path = 'gs://apache-beam-ml/models/imagenet_classification_mobilenet_v2.pt'
        extra_opts = {'input': file_of_image_names, 'output': output_file, 'model_state_dict_path': model_state_dict_path}
        pytorch_image_classification.run(test_pipeline.get_full_options_as_args(**extra_opts), save_main_session=False)
        self.assertEqual(FileSystems().exists(output_file), True)
        predictions = process_outputs(filepath=output_file)
        for prediction in predictions:
            (filename, prediction) = prediction.split(',')
            self.assertEqual(_EXPECTED_OUTPUTS[filename], prediction)

    @pytest.mark.uses_pytorch
    @pytest.mark.it_postcommit
    def test_torch_run_inference_coco_maskrcnn_resnet50_fpn(self):
        if False:
            for i in range(10):
                print('nop')
        test_pipeline = TestPipeline(is_integration_test=True)
        file_of_image_names = 'gs://apache-beam-ml/testing/inputs/it_coco_validation_inputs.txt'
        output_file_dir = 'gs://apache-beam-ml/testing/predictions'
        output_file = '/'.join([output_file_dir, str(uuid.uuid4()), 'result.txt'])
        model_state_dict_path = 'gs://apache-beam-ml/models/torchvision.models.detection.maskrcnn_resnet50_fpn.pth'
        images_dir = 'gs://apache-beam-ml/datasets/coco/raw-data/val2017'
        extra_opts = {'input': file_of_image_names, 'output': output_file, 'model_state_dict_path': model_state_dict_path, 'images_dir': images_dir}
        pytorch_image_segmentation.run(test_pipeline.get_full_options_as_args(**extra_opts), save_main_session=False)
        self.assertEqual(FileSystems().exists(output_file), True)
        predictions = process_outputs(filepath=output_file)
        actuals_file = 'gs://apache-beam-ml/testing/expected_outputs/test_torch_run_inference_coco_maskrcnn_resnet50_fpn_actuals.txt'
        actuals = process_outputs(filepath=actuals_file)
        predictions_dict = {}
        for prediction in predictions:
            (filename, prediction_labels) = prediction.split(';')
            predictions_dict[filename] = prediction_labels
        for actual in actuals:
            (filename, actual_labels) = actual.split(';')
            prediction_labels = predictions_dict[filename]
            self.assertEqual(actual_labels, prediction_labels)

    @pytest.mark.uses_pytorch
    @pytest.mark.it_postcommit
    @pytest.mark.timeout(1800)
    def test_torch_run_inference_coco_maskrcnn_resnet50_fpn_v1_and_v2(self):
        if False:
            print('Hello World!')
        test_pipeline = TestPipeline(is_integration_test=True)
        file_of_image_names = 'gs://apache-beam-ml/testing/inputs/it_coco_validation_inputs.txt'
        output_file_dir = 'gs://apache-beam-ml/testing/predictions'
        output_file = '/'.join([output_file_dir, str(uuid.uuid4()), 'result.txt'])
        model_state_dict_paths = ['gs://apache-beam-ml/models/torchvision.models.detection.maskrcnn_resnet50_fpn.pth', 'gs://apache-beam-ml/models/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.pth']
        images_dir = 'gs://apache-beam-ml/datasets/coco/raw-data/val2017'
        extra_opts = {'input': file_of_image_names, 'output': output_file, 'model_state_dict_paths': ','.join(model_state_dict_paths), 'images_dir': images_dir}
        pytorch_model_per_key_image_segmentation.run(test_pipeline.get_full_options_as_args(**extra_opts), save_main_session=False)
        self.assertEqual(FileSystems().exists(output_file), True)
        predictions = process_outputs(filepath=output_file)
        actuals_file = 'gs://apache-beam-ml/testing/expected_outputs/test_torch_run_inference_coco_maskrcnn_resnet50_fpn_v1_and_v2_actuals.txt'
        actuals = process_outputs(filepath=actuals_file)
        predictions_dict = {}
        for prediction in predictions:
            p = prediction.split('---')
            filename = p[0]
            v1predictions = p[1]
            v2predictions = p[2]
            predictions_dict[filename] = (v1predictions, v2predictions)
        for actual in actuals:
            a = actual.split('---')
            filename = a[0]
            v1actuals = a[1]
            v2actuals = a[2]
            (v1prediction_labels, v2prediction_labels) = predictions_dict[filename]
            self.assertEqual(v1actuals, v1prediction_labels)
            self.assertEqual(v2actuals, v2prediction_labels)

    @pytest.mark.uses_pytorch
    @pytest.mark.it_postcommit
    def test_torch_run_inference_bert_for_masked_lm(self):
        if False:
            i = 10
            return i + 15
        test_pipeline = TestPipeline(is_integration_test=True)
        file_of_sentences = 'gs://apache-beam-ml/datasets/custom/sentences.txt'
        output_file_dir = 'gs://apache-beam-ml/testing/predictions'
        output_file = '/'.join([output_file_dir, str(uuid.uuid4()), 'result.txt'])
        model_state_dict_path = 'gs://apache-beam-ml/models/huggingface.BertForMaskedLM.bert-base-uncased.pth'
        extra_opts = {'input': file_of_sentences, 'output': output_file, 'model_state_dict_path': model_state_dict_path}
        pytorch_language_modeling.run(test_pipeline.get_full_options_as_args(**extra_opts), save_main_session=False)
        self.assertEqual(FileSystems().exists(output_file), True)
        predictions = process_outputs(filepath=output_file)
        actuals_file = 'gs://apache-beam-ml/testing/expected_outputs/test_torch_run_inference_bert_for_masked_lm_actuals.txt'
        actuals = process_outputs(filepath=actuals_file)
        predictions_dict = {}
        for prediction in predictions:
            (text, predicted_text) = prediction.split(';')
            predictions_dict[text] = predicted_text
        for actual in actuals:
            (text, actual_predicted_text) = actual.split(';')
            predicted_predicted_text = predictions_dict[text]
            self.assertEqual(actual_predicted_text, predicted_predicted_text)

    @pytest.mark.uses_pytorch
    @pytest.mark.it_postcommit
    def test_torch_run_inference_bert_for_masked_lm_large_model(self):
        if False:
            return 10
        test_pipeline = TestPipeline(is_integration_test=True)
        file_of_sentences = 'gs://apache-beam-ml/datasets/custom/sentences.txt'
        output_file_dir = 'gs://apache-beam-ml/testing/predictions'
        output_file = '/'.join([output_file_dir, str(uuid.uuid4()), 'result.txt'])
        model_state_dict_path = 'gs://apache-beam-ml/models/huggingface.BertForMaskedLM.bert-base-uncased.pth'
        extra_opts = {'input': file_of_sentences, 'output': output_file, 'model_state_dict_path': model_state_dict_path, 'large_model': True}
        pytorch_language_modeling.run(test_pipeline.get_full_options_as_args(**extra_opts), save_main_session=False)
        self.assertEqual(FileSystems().exists(output_file), True)
        predictions = process_outputs(filepath=output_file)
        actuals_file = 'gs://apache-beam-ml/testing/expected_outputs/test_torch_run_inference_bert_for_masked_lm_actuals.txt'
        actuals = process_outputs(filepath=actuals_file)
        predictions_dict = {}
        for prediction in predictions:
            (text, predicted_text) = prediction.split(';')
            predictions_dict[text] = predicted_text
        for actual in actuals:
            (text, actual_predicted_text) = actual.split(';')
            predicted_predicted_text = predictions_dict[text]
            self.assertEqual(actual_predicted_text, predicted_predicted_text)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()