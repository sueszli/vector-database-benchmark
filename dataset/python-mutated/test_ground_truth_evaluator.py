from __future__ import absolute_import, division, print_function, unicode_literals
import json
import logging
import pprint
import unittest
from art.defences.detector.poison import GroundTruthEvaluator
from tests.utils import master_seed
logger = logging.getLogger(__name__)

class TestGroundTruth(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        master_seed(seed=1234)

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        cls.evaluator = GroundTruthEvaluator()
        cls.n_classes = 3
        cls.n_dp = 10
        cls.n_dp_mix = 5
        cls.is_clean_all_clean = [[] for _ in range(cls.n_classes)]
        cls.is_clean_all_poison = [[] for _ in range(cls.n_classes)]
        cls.is_clean_mixed = [[] for _ in range(cls.n_classes)]
        cls.is_clean_comp_mix = [[] for _ in range(cls.n_classes)]
        for i in range(cls.n_classes):
            cls.is_clean_all_clean[i] = [1] * cls.n_dp
            cls.is_clean_all_poison[i] = [0] * cls.n_dp
            cls.is_clean_mixed[i] = [1, 0, 0, 1, 0, 1, 1, 1, 0, 0]
            cls.is_clean_comp_mix[i] = [0, 1, 1, 0, 1, 0, 0, 0, 1, 1]

    def test_analyze_correct_all_clean(self):
        if False:
            for i in range(10):
                print('nop')
        (errors_by_class, conf_matrix_json) = self.evaluator.analyze_correctness(self.is_clean_all_clean, self.is_clean_all_clean)
        json_object = json.loads(conf_matrix_json)
        self.assertEqual(len(json_object.keys()), self.n_classes)
        self.assertEqual(len(errors_by_class), self.n_classes)
        for i in range(self.n_classes):
            res_class_i = json_object['class_' + str(i)]
            self.assertEqual(res_class_i['TruePositive']['rate'], 'N/A')
            self.assertEqual(res_class_i['TrueNegative']['rate'], 100)
            self.assertEqual(res_class_i['FalseNegative']['rate'], 'N/A')
            self.assertEqual(res_class_i['FalsePositive']['rate'], 0)
            self.assertEqual(res_class_i['TruePositive']['numerator'], 0)
            self.assertEqual(res_class_i['TruePositive']['denominator'], 0)
            self.assertEqual(res_class_i['TrueNegative']['numerator'], self.n_dp)
            self.assertEqual(res_class_i['TrueNegative']['denominator'], self.n_dp)
            self.assertEqual(res_class_i['FalseNegative']['numerator'], 0)
            self.assertEqual(res_class_i['FalseNegative']['denominator'], 0)
            self.assertEqual(res_class_i['FalsePositive']['numerator'], 0)
            self.assertEqual(res_class_i['FalsePositive']['denominator'], self.n_dp)
            for item in errors_by_class[i]:
                self.assertEqual(item, 1)

    def test_analyze_correct_all_poison(self):
        if False:
            return 10
        (errors_by_class, conf_matrix_json) = self.evaluator.analyze_correctness(self.is_clean_all_poison, self.is_clean_all_poison)
        json_object = json.loads(conf_matrix_json)
        self.assertEqual(len(json_object.keys()), self.n_classes)
        self.assertEqual(len(errors_by_class), self.n_classes)
        for i in range(self.n_classes):
            res_class_i = json_object['class_' + str(i)]
            self.assertEqual(res_class_i['TruePositive']['rate'], 100)
            self.assertEqual(res_class_i['TrueNegative']['rate'], 'N/A')
            self.assertEqual(res_class_i['FalseNegative']['rate'], 0)
            self.assertEqual(res_class_i['FalsePositive']['rate'], 'N/A')
            self.assertEqual(res_class_i['TruePositive']['numerator'], self.n_dp)
            self.assertEqual(res_class_i['TruePositive']['denominator'], self.n_dp)
            self.assertEqual(res_class_i['TrueNegative']['numerator'], 0)
            self.assertEqual(res_class_i['TrueNegative']['denominator'], 0)
            self.assertEqual(res_class_i['FalseNegative']['numerator'], 0)
            self.assertEqual(res_class_i['FalseNegative']['denominator'], self.n_dp)
            self.assertEqual(res_class_i['FalsePositive']['numerator'], 0)
            self.assertEqual(res_class_i['FalsePositive']['denominator'], 0)
            for item in errors_by_class[i]:
                self.assertEqual(item, 0)

    def test_analyze_correct_mixed(self):
        if False:
            return 10
        (errors_by_class, conf_matrix_json) = self.evaluator.analyze_correctness(self.is_clean_mixed, self.is_clean_mixed)
        json_object = json.loads(conf_matrix_json)
        self.assertEqual(len(json_object.keys()), self.n_classes)
        self.assertEqual(len(errors_by_class), self.n_classes)
        for i in range(self.n_classes):
            res_class_i = json_object['class_' + str(i)]
            self.assertEqual(res_class_i['TruePositive']['rate'], 100)
            self.assertEqual(res_class_i['TrueNegative']['rate'], 100)
            self.assertEqual(res_class_i['FalseNegative']['rate'], 0)
            self.assertEqual(res_class_i['FalsePositive']['rate'], 0)
            self.assertEqual(res_class_i['TruePositive']['numerator'], self.n_dp_mix)
            self.assertEqual(res_class_i['TruePositive']['denominator'], self.n_dp_mix)
            self.assertEqual(res_class_i['TrueNegative']['numerator'], self.n_dp_mix)
            self.assertEqual(res_class_i['TrueNegative']['denominator'], self.n_dp_mix)
            self.assertEqual(res_class_i['FalseNegative']['numerator'], 0)
            self.assertEqual(res_class_i['FalseNegative']['denominator'], self.n_dp_mix)
            self.assertEqual(res_class_i['FalsePositive']['numerator'], 0)
            self.assertEqual(res_class_i['FalsePositive']['denominator'], self.n_dp_mix)
            for (j, item) in enumerate(errors_by_class[i]):
                self.assertEqual(item, self.is_clean_mixed[i][j])

    def test_analyze_fully_misclassified(self):
        if False:
            while True:
                i = 10
        (errors_by_class, conf_matrix_json) = self.evaluator.analyze_correctness(self.is_clean_all_clean, self.is_clean_all_poison)
        json_object = json.loads(conf_matrix_json)
        self.assertEqual(len(json_object.keys()), self.n_classes)
        self.assertEqual(len(errors_by_class), self.n_classes)
        print(json_object)
        for i in range(self.n_classes):
            res_class_i = json_object['class_' + str(i)]
            self.assertEqual(res_class_i['TruePositive']['rate'], 0)
            self.assertEqual(res_class_i['TrueNegative']['rate'], 'N/A')
            self.assertEqual(res_class_i['FalseNegative']['rate'], 100)
            self.assertEqual(res_class_i['FalsePositive']['rate'], 'N/A')
            self.assertEqual(res_class_i['TruePositive']['numerator'], 0)
            self.assertEqual(res_class_i['TruePositive']['denominator'], self.n_dp)
            self.assertEqual(res_class_i['TrueNegative']['numerator'], 0)
            self.assertEqual(res_class_i['TrueNegative']['denominator'], 0)
            self.assertEqual(res_class_i['FalseNegative']['numerator'], self.n_dp)
            self.assertEqual(res_class_i['FalseNegative']['denominator'], self.n_dp)
            self.assertEqual(res_class_i['FalsePositive']['numerator'], 0)
            self.assertEqual(res_class_i['FalsePositive']['denominator'], 0)
            for item in errors_by_class[i]:
                self.assertEqual(item, 3)

    def test_analyze_fully_misclassified_rev(self):
        if False:
            for i in range(10):
                print('nop')
        (errors_by_class, conf_matrix_json) = self.evaluator.analyze_correctness(self.is_clean_all_poison, self.is_clean_all_clean)
        json_object = json.loads(conf_matrix_json)
        self.assertEqual(len(json_object.keys()), self.n_classes)
        self.assertEqual(len(errors_by_class), self.n_classes)
        pprint.pprint(json_object)
        for i in range(self.n_classes):
            res_class_i = json_object['class_' + str(i)]
            self.assertEqual(res_class_i['TruePositive']['rate'], 'N/A')
            self.assertEqual(res_class_i['TrueNegative']['rate'], 0)
            self.assertEqual(res_class_i['FalseNegative']['rate'], 'N/A')
            self.assertEqual(res_class_i['FalsePositive']['rate'], 100)
            self.assertEqual(res_class_i['TruePositive']['numerator'], 0)
            self.assertEqual(res_class_i['TruePositive']['denominator'], 0)
            self.assertEqual(res_class_i['TrueNegative']['numerator'], 0)
            self.assertEqual(res_class_i['TrueNegative']['denominator'], self.n_dp)
            self.assertEqual(res_class_i['FalseNegative']['numerator'], 0)
            self.assertEqual(res_class_i['FalseNegative']['denominator'], 0)
            self.assertEqual(res_class_i['FalsePositive']['numerator'], self.n_dp)
            self.assertEqual(res_class_i['FalsePositive']['denominator'], self.n_dp)
            for item in errors_by_class[i]:
                self.assertEqual(item, 2)