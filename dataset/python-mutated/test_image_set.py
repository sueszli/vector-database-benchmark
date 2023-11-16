import pytest
import cv2
from bigdl.dllib.nncontext import *
from bigdl.dllib.feature.image import *

class Test_Image_Set:

    def setup_method(self, method):
        if False:
            i = 10
            return i + 15
        ' setup any state tied to the execution of the given method in a\n        class.  setup_method is invoked for every test method of a class.\n        '
        self.sc = init_nncontext(init_spark_conf().setMaster('local[4]').setAppName('test image set'))
        resource_path = os.path.join(os.path.split(__file__)[0], '../../resources')
        self.image_path = os.path.join(resource_path, 'pascal/000025.jpg')
        self.grayimage_path = os.path.join(resource_path, 'gray/gray.bmp')
        self.image_folder = os.path.join(resource_path, 'imagenet')

    def teardown_method(self, method):
        if False:
            return 10
        ' teardown any state that was previously setup with a setup_method\n        call.\n        '
        self.sc.stop()

    def transformer_test(self, transformer):
        if False:
            for i in range(10):
                print('nop')
        image_set = ImageSet.read(self.image_path)
        transformed = transformer(image_set)
        transformed.get_image()
        image_set = ImageSet.read(self.image_path, self.sc)
        transformed = transformer(image_set)
        images = transformed.get_image()
        images.count()

    def test_get_image(self):
        if False:
            return 10
        image_set = ImageSet.read(self.image_path, resize_height=128, resize_width=128)
        image = image_set.get_image()
        image_set = ImageSet.read(self.image_path)
        image = image_set.get_image()
        assert image[0].shape[0] is 3
        image_set = ImageSet.read(self.image_path, self.sc)
        image = image_set.get_image().collect()
        assert image[0].shape[0] is 3
        image_set = ImageSet.read(self.grayimage_path)
        image = image_set.get_image()
        assert image[0].shape[0] is 1
        image_set = ImageSet.read(self.grayimage_path, self.sc)
        image = image_set.get_image().collect()
        assert image[0].shape[0] is 1

    def test_get_label(self):
        if False:
            while True:
                i = 10
        image_set = ImageSet.read(self.image_path)
        image_set.get_label()

    def test_is_local(self):
        if False:
            while True:
                i = 10
        image_set = ImageSet.read(self.image_path)
        assert image_set.is_local() is True
        image_set = ImageSet.read(self.image_path, self.sc)
        assert image_set.is_local() is False

    def test_is_distributed(self):
        if False:
            while True:
                i = 10
        image_set = ImageSet.read(self.image_path)
        assert image_set.is_distributed() is False
        image_set = ImageSet.read(self.image_path, self.sc)
        assert image_set.is_distributed() is True

    def test_image_set_transform(self):
        if False:
            return 10
        transformer = ImageMatToTensor()
        image_set = ImageSet.read(self.image_path)
        transformed = image_set.transform(transformer)
        transformed.get_image()

    def test_empty_get_predict_local(self):
        if False:
            while True:
                i = 10
        image_set = ImageSet.read(self.image_path)
        image_set.get_predict()

    def test_empty_get_predict_distributed(self):
        if False:
            i = 10
            return i + 15
        image_set = ImageSet.read(self.image_path, self.sc)
        image_set.get_predict()

    def test_local_image_set(self):
        if False:
            for i in range(10):
                print('nop')
        image = cv2.imread(self.image_path)
        local_image_set = LocalImageSet([image])
        print(local_image_set.get_image())

    def test_image_set_random_preprocess(self):
        if False:
            print('Hello World!')
        transformer = ImageRandomPreprocessing(ImageResize(10, 10), 1.0)
        image_set = ImageSet.read(self.image_path)
        transformed = image_set.transform(transformer)
        img = transformed.get_image()[0]
        assert img.shape == (3, 10, 10)

    def test_image_set_from_image_folder_with_sc(self):
        if False:
            while True:
                i = 10
        image_set = ImageSet.read(self.image_folder, sc=self.sc, with_label=True)
        label_map = image_set.label_map
        assert len(label_map) == 4
        imgs = image_set.get_image().collect()
        assert len(imgs) == 11
        labels = image_set.get_label().collect()
        labels = [l[0] for l in labels]
        assert len(labels) == 11
        assert len(set(labels)) == 4

    def test_image_set_from_image_folder_without_sc(self):
        if False:
            while True:
                i = 10
        image_set = ImageSet.read(self.image_folder, with_label=True)
        label_map = image_set.label_map
        assert len(label_map) == 4
        imgs = image_set.get_image()
        assert len(imgs) == 11
        labels = image_set.get_label()
        labels = [l[0] for l in labels]
        assert len(labels) == 11
        assert len(set(labels)) == 4

    def test_local_image_set_with_zero_based_label(self):
        if False:
            return 10
        image_set = ImageSet.read(self.image_folder, with_label=True, one_based_label=False)
        label_map = image_set.label_map
        for k in label_map:
            assert label_map[k] < 4.0
        for label in image_set.get_label():
            assert label < 4.0

    def test_distributed_image_set_with_zero_based_label(self):
        if False:
            for i in range(10):
                print('nop')
        image_set = ImageSet.read(self.image_folder, sc=self.sc, with_label=True, one_based_label=False)
        label_map = image_set.label_map
        for k in label_map:
            assert label_map[k] < 4.0
        for label in image_set.get_label().collect():
            assert label < 4.0

    def test_local_image_set_with_one_based_label(self):
        if False:
            while True:
                i = 10
        image_set = ImageSet.read(self.image_folder, with_label=True, one_based_label=True)
        label_map = image_set.label_map
        for k in label_map:
            assert label_map[k] <= 4.0 and label_map[k] > 0.0
        for label in image_set.get_label():
            assert label <= 4.0 and label > 0.0

    def test_distributed_image_set_with_one_based_label(self):
        if False:
            return 10
        image_set = ImageSet.read(self.image_folder, sc=self.sc, with_label=True, one_based_label=True)
        label_map = image_set.label_map
        for k in label_map:
            assert label_map[k] > 0.0 and label_map[k] <= 4.0
        for label in image_set.get_label().collect():
            assert label > 0.0 and label <= 4.0

    def test_distributed_image_set_from_rdds(self):
        if False:
            while True:
                i = 10
        image_rdd = self.sc.parallelize(np.zeros((4, 32, 32, 3)))
        label_rdd = self.sc.parallelize(np.random.randint(0, 4, size=(4, 1)))
        image_set = ImageSet.from_rdds(image_rdd, label_rdd)
        for image in image_set.get_image().collect():
            assert image.sum() == 0.0
        for label in image_set.get_label().collect():
            assert label >= 0.0 and label < 4.0
if __name__ == '__main__':
    pytest.main([__file__])