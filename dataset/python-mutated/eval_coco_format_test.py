"""Tests for eval_coco_format script."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import flags
from absl.testing import absltest
import evaluation as panopticapi_eval
from deeplab.evaluation import eval_coco_format
_TEST_DIR = 'deeplab/evaluation/testdata'
FLAGS = flags.FLAGS

class EvalCocoFormatTest(absltest.TestCase):

    def test_compare_pq_with_reference_eval(self):
        if False:
            return 10
        sample_data_dir = os.path.join(_TEST_DIR)
        gt_json_file = os.path.join(sample_data_dir, 'coco_gt.json')
        gt_folder = os.path.join(sample_data_dir, 'coco_gt')
        pred_json_file = os.path.join(sample_data_dir, 'coco_pred.json')
        pred_folder = os.path.join(sample_data_dir, 'coco_pred')
        panopticapi_results = panopticapi_eval.pq_compute(gt_json_file, pred_json_file, gt_folder, pred_folder)
        deeplab_results = eval_coco_format.eval_coco_format(gt_json_file, pred_json_file, gt_folder, pred_folder, metric='pq', num_categories=7, ignored_label=0, max_instances_per_category=256, intersection_offset=256 * 256)
        self.assertCountEqual(list(deeplab_results.keys()), ['All', 'Things', 'Stuff'])
        for cat_group in ['All', 'Things', 'Stuff']:
            self.assertCountEqual(deeplab_results[cat_group], ['pq', 'sq', 'rq', 'n'])
            for metric in ['pq', 'sq', 'rq', 'n']:
                self.assertAlmostEqual(deeplab_results[cat_group][metric], panopticapi_results[cat_group][metric])

    def test_compare_pc_with_golden_value(self):
        if False:
            return 10
        sample_data_dir = os.path.join(_TEST_DIR)
        gt_json_file = os.path.join(sample_data_dir, 'coco_gt.json')
        gt_folder = os.path.join(sample_data_dir, 'coco_gt')
        pred_json_file = os.path.join(sample_data_dir, 'coco_pred.json')
        pred_folder = os.path.join(sample_data_dir, 'coco_pred')
        deeplab_results = eval_coco_format.eval_coco_format(gt_json_file, pred_json_file, gt_folder, pred_folder, metric='pc', num_categories=7, ignored_label=0, max_instances_per_category=256, intersection_offset=256 * 256, normalize_by_image_size=False)
        self.assertCountEqual(list(deeplab_results.keys()), ['All', 'Things', 'Stuff'])
        for cat_group in ['All', 'Things', 'Stuff']:
            self.assertCountEqual(deeplab_results[cat_group], ['pc', 'n'])
        self.assertAlmostEqual(deeplab_results['All']['pc'], 0.68210561)
        self.assertEqual(deeplab_results['All']['n'], 6)
        self.assertAlmostEqual(deeplab_results['Things']['pc'], 0.5890529)
        self.assertEqual(deeplab_results['Things']['n'], 4)
        self.assertAlmostEqual(deeplab_results['Stuff']['pc'], 0.86821097)
        self.assertEqual(deeplab_results['Stuff']['n'], 2)

    def test_compare_pc_with_golden_value_normalize_by_size(self):
        if False:
            return 10
        sample_data_dir = os.path.join(_TEST_DIR)
        gt_json_file = os.path.join(sample_data_dir, 'coco_gt.json')
        gt_folder = os.path.join(sample_data_dir, 'coco_gt')
        pred_json_file = os.path.join(sample_data_dir, 'coco_pred.json')
        pred_folder = os.path.join(sample_data_dir, 'coco_pred')
        deeplab_results = eval_coco_format.eval_coco_format(gt_json_file, pred_json_file, gt_folder, pred_folder, metric='pc', num_categories=7, ignored_label=0, max_instances_per_category=256, intersection_offset=256 * 256, normalize_by_image_size=True)
        self.assertCountEqual(list(deeplab_results.keys()), ['All', 'Things', 'Stuff'])
        self.assertAlmostEqual(deeplab_results['All']['pc'], 0.6821490884)

    def test_pc_with_multiple_workers(self):
        if False:
            for i in range(10):
                print('nop')
        sample_data_dir = os.path.join(_TEST_DIR)
        gt_json_file = os.path.join(sample_data_dir, 'coco_gt.json')
        gt_folder = os.path.join(sample_data_dir, 'coco_gt')
        pred_json_file = os.path.join(sample_data_dir, 'coco_pred.json')
        pred_folder = os.path.join(sample_data_dir, 'coco_pred')
        deeplab_results = eval_coco_format.eval_coco_format(gt_json_file, pred_json_file, gt_folder, pred_folder, metric='pc', num_categories=7, ignored_label=0, max_instances_per_category=256, intersection_offset=256 * 256, num_workers=3, normalize_by_image_size=False)
        self.assertCountEqual(list(deeplab_results.keys()), ['All', 'Things', 'Stuff'])
        self.assertAlmostEqual(deeplab_results['All']['pc'], 0.68210561668)
if __name__ == '__main__':
    absltest.main()