from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import unittest
import pytest
import turicreate as tc
from turicreate.toolkits._internal_utils import _mac_ver
import tempfile
import shutil
import coremltools
import numpy as np
from turicreate.toolkits._main import ToolkitError as _ToolkitError

def _get_data(image_length):
    if False:
        print('Hello World!')
    from PIL import Image as _PIL_Image
    random = np.random.RandomState(100)
    _format = {'JPG': 0, 'PNG': 1, 'RAW': 2, 'UNDEFINED': 3}

    def from_pil_image(pil_img):
        if False:
            for i in range(10):
                print('nop')
        height = pil_img.size[1]
        width = pil_img.size[0]
        if pil_img.mode == 'L':
            image_data = bytearray([z for z in pil_img.getdata()])
            channels = 1
        elif pil_img.mode == 'RGB':
            image_data = bytearray([z for l in pil_img.getdata() for z in l])
            channels = 3
        else:
            image_data = bytearray([z for l in pil_img.getdata() for z in l])
            channels = 4
        format_enum = _format['RAW']
        image_data_size = len(image_data)
        img = tc.Image(_image_data=image_data, _width=width, _height=height, _channels=channels, _format_enum=format_enum, _image_data_size=image_data_size)
        return img
    num_examples = 100
    dims = (image_length, image_length)
    total_dims = dims[0] * dims[1]
    images = []
    for i in range(num_examples):

        def rand_image():
            if False:
                i = 10
                return i + 15
            return [random.randint(0, 255)] * total_dims
        pil_img = _PIL_Image.new('RGB', dims)
        pil_img.putdata(list(zip(rand_image(), rand_image(), rand_image())))
        images.append(from_pil_image(pil_img))
    data = tc.SFrame({'awesome_image': tc.SArray(images)})
    return data

class TempDirectory:
    name = None

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.name = tempfile.mkdtemp()

    def __enter__(self):
        if False:
            return 10
        return self.name

    def __exit__(self, type, value, traceback):
        if False:
            for i in range(10):
                print('nop')
        if self.name is not None:
            shutil.rmtree(self.name)

class ImageSimilarityTest(unittest.TestCase):

    @classmethod
    def setUpClass(self, input_image_shape=(3, 224, 224), model='resnet-50'):
        if False:
            while True:
                i = 10
        '\n        The setup class method for the basic test case with all default values.\n        '
        self.feature = 'awesome_image'
        self.label = None
        self.input_image_shape = input_image_shape
        self.pre_trained_model = model
        self.def_opts = {'model': 'resnet-50', 'verbose': True}
        self.sf = _get_data(self.input_image_shape[2])
        self.model = tc.image_similarity.create(self.sf, feature=self.feature, label=None, model=self.pre_trained_model)
        self.nn_model = self.model.feature_extractor
        self.lm_model = self.model.similarity_model
        self.opts = self.def_opts.copy()
        self.get_ans = {'similarity_model': lambda x: type(x) == tc.nearest_neighbors.NearestNeighborsModel, 'feature': lambda x: x == self.feature, 'training_time': lambda x: x > 0, 'input_image_shape': lambda x: x == self.input_image_shape, 'label': lambda x: x == self.label, 'feature_extractor': lambda x: callable(x.extract_features), 'num_features': lambda x: x == self.lm_model.num_features, 'num_examples': lambda x: x == self.lm_model.num_examples, 'model': lambda x: x == self.pre_trained_model}
        self.fields_ans = self.get_ans.keys()

    def assertListAlmostEquals(self, list1, list2, tol):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(len(list1), len(list2))
        for (a, b) in zip(list1, list2):
            self.assertAlmostEqual(a, b, delta=tol)

    def test_create_with_missing_feature(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(_ToolkitError):
            tc.image_similarity.create(self.sf, feature='wrong_feature', label=self.label)

    def test_create_with_missing_label(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(_ToolkitError):
            tc.image_similarity.create(self.sf, feature=self.feature, label='wrong_label')

    def test_create_with_empty_dataset(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(_ToolkitError):
            tc.image_similarity.create(self.sf[:0])

    def test_query(self):
        if False:
            return 10
        model = self.model
        preds = model.query(self.sf)
        self.assertEqual(len(preds), len(self.sf) * 5)

    def test_similarity_graph(self):
        if False:
            print('Hello World!')
        model = self.model
        preds = model.similarity_graph()
        self.assertEqual(len(preds.edges), len(self.sf) * 5)
        preds = model.similarity_graph(output_type='SFrame')
        self.assertEqual(len(preds), len(self.sf) * 5)

    def test_list_fields(self):
        if False:
            return 10
        model = self.model
        fields = model._list_fields()
        self.assertEqual(set(fields), set(self.fields_ans))

    def test_get(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check the get function. Compare with the answer supplied as a lambda\n        function for each field.\n        '
        model = self.model
        for field in self.fields_ans:
            ans = model._get(field)
            self.assertTrue(self.get_ans[field](ans), 'Get failed in field {}. Output was {}.'.format(field, ans))

    def test_query_input(self):
        if False:
            while True:
                i = 10
        model = self.model
        single_image = self.sf[self.feature][0]
        sims = model.query(single_image)
        self.assertIsNotNone(sims)
        sarray = self.sf[self.feature]
        sims = model.query(sarray)
        self.assertIsNotNone(sims)
        with self.assertRaises(TypeError):
            model.query('this is a junk value')

    def test_summary(self):
        if False:
            while True:
                i = 10
        model = self.model
        model.summary()

    def test_repr(self):
        if False:
            print('Hello World!')
        model = self.model
        self.assertEqual(type(str(model)), str)
        self.assertEqual(type(model.__repr__()), str)

    def test_export_coreml(self):
        if False:
            print('Hello World!')
        '\n        Check the export_coreml() function.\n        '

        def get_psnr(x, y):
            if False:
                i = 10
                return i + 15
            return 20 * np.log10(max(x.max(), y.max())) - 10 * np.log10(np.square(x - y).mean())
        filename = tempfile.mkstemp('ImageSimilarity.mlmodel')[1]
        self.model.export_coreml(filename)
        coreml_model = coremltools.models.MLModel(filename)
        img = self.sf[0:1][self.feature][0]
        img_fixed = tc.image_analysis.resize(img, *reversed(self.input_image_shape))
        tc_ret = self.model.query(img_fixed, k=self.sf.num_rows())
        if _mac_ver() >= (10, 13):
            from PIL import Image as _PIL_Image
            pil_img = _PIL_Image.fromarray(img_fixed.pixel_data)
            coreml_ret = coreml_model.predict({'awesome_image': pil_img})
            coreml_distances = np.array(coreml_ret['distance'])
            tc_distances = tc_ret.sort('reference_label')['distance'].to_numpy()
            psnr_value = get_psnr(coreml_distances, tc_distances)
            self.assertTrue(psnr_value > 50)

    def test_save_and_load(self):
        if False:
            while True:
                i = 10
        with TempDirectory() as filename:
            self.model.save(filename)
            self.model = tc.load_model(filename)
            self.test_query()
            print('Query passed')
            self.test_similarity_graph()
            print('Similarity graph passed')
            self.test_get()
            print('Get passed')
            self.test_summary()
            print('Summary passed')
            self.test_list_fields()
            print('List fields passed')
            self.test_export_coreml()
            print('Export coreml passed')

class ImageSimilaritySqueezeNetTest(ImageSimilarityTest):

    @classmethod
    def setUpClass(self):
        if False:
            i = 10
            return i + 15
        super(ImageSimilaritySqueezeNetTest, self).setUpClass(model='squeezenet_v1.1', input_image_shape=(3, 227, 227))

@unittest.skipIf(_mac_ver() < (10, 14), 'VisionFeaturePrint_Scene only supported on macOS 10.14+')
class ImageSimilarityVisionFeaturePrintSceneTest(ImageSimilarityTest):

    @classmethod
    def setUpClass(self):
        if False:
            print('Hello World!')
        super(ImageSimilarityVisionFeaturePrintSceneTest, self).setUpClass(model='VisionFeaturePrint_Scene', input_image_shape=(3, 299, 299))

@unittest.skipIf(tc.util._num_available_cuda_gpus() == 0, 'Requires CUDA GPU')
@pytest.mark.gpu
class ImageSimilarityGPUTest(unittest.TestCase):

    @classmethod
    def setUpClass(self, model='resnet-50'):
        if False:
            while True:
                i = 10
        self.feature = 'awesome_image'
        self.label = None
        self.input_image_shape = (3, 224, 224)
        self.pre_trained_model = model
        self.sf = _get_data(self.input_image_shape[2])

    def test_gpu_save_load_export(self):
        if False:
            print('Hello World!')
        old_num_gpus = tc.config.get_num_gpus()
        gpu_options = set([old_num_gpus, 0, 1])
        for in_gpus in gpu_options:
            for out_gpus in gpu_options:
                tc.config.set_num_gpus(in_gpus)
                model = tc.image_similarity.create(self.sf, feature=self.feature, label=None, model=self.pre_trained_model)
                with TempDirectory() as filename:
                    model.save(filename)
                    tc.config.set_num_gpus(out_gpus)
                    model = tc.load_model(filename)
        tc.config.set_num_gpus(old_num_gpus)