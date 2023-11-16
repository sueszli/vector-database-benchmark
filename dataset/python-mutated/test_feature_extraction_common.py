import json
import os
import tempfile
from transformers.testing_utils import check_json_file_has_correct_format

class FeatureExtractionSavingTestMixin:
    test_cast_dtype = None

    def test_feat_extract_to_json_string(self):
        if False:
            for i in range(10):
                print('nop')
        feat_extract = self.feature_extraction_class(**self.feat_extract_dict)
        obj = json.loads(feat_extract.to_json_string())
        for (key, value) in self.feat_extract_dict.items():
            self.assertEqual(obj[key], value)

    def test_feat_extract_to_json_file(self):
        if False:
            for i in range(10):
                print('nop')
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)
        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, 'feat_extract.json')
            feat_extract_first.to_json_file(json_file_path)
            feat_extract_second = self.feature_extraction_class.from_json_file(json_file_path)
        self.assertEqual(feat_extract_second.to_dict(), feat_extract_first.to_dict())

    def test_feat_extract_from_and_save_pretrained(self):
        if False:
            i = 10
            return i + 15
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)
        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_file = feat_extract_first.save_pretrained(tmpdirname)[0]
            check_json_file_has_correct_format(saved_file)
            feat_extract_second = self.feature_extraction_class.from_pretrained(tmpdirname)
        self.assertEqual(feat_extract_second.to_dict(), feat_extract_first.to_dict())

    def test_init_without_params(self):
        if False:
            for i in range(10):
                print('nop')
        feat_extract = self.feature_extraction_class()
        self.assertIsNotNone(feat_extract)