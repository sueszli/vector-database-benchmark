import pytest
import h5py
from math import pi
from bigdl.dllib.nncontext import *
from bigdl.dllib.feature.common import *
from bigdl.dllib.feature.image3d.transformation import *

class Test_Image3D:

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        ' setup any state tied to the execution of the given method in a\n        class.  setup_method is invoked for every test method of a class.\n        '
        self.sc = init_nncontext(create_spark_conf().setMaster('local[4]').setAppName('test image set').set('spark.shuffle.reduceLocality.enabled', 'false').set('spark.shuffle.blockTransferService', 'nio').set('spark.scheduler.minRegisteredResourcesRatio', '1.0').set('spark.speculation', 'false'))
        resource_path = os.path.join(os.path.split(__file__)[0], '../../resources')
        image_path = os.path.join(resource_path, 'image3d/a.mat')
        img = h5py.File(image_path)['meniscus_im']
        sample = np.array(img)
        self.sample = np.expand_dims(sample, 3)

    def teardown_method(self, method):
        if False:
            return 10
        ' teardown any state that was previously setup with a setup_method\n        call.\n        '
        self.sc.stop()

    def test_crop(self):
        if False:
            print('Hello World!')
        start_loc = [13, 80, 125]
        patch = [5, 40, 40]
        crop = Crop3D(start=start_loc, patch_size=patch)
        data_rdd = self.sc.parallelize([self.sample])
        image_set = DistributedImageSet(image_rdd=data_rdd)
        transformed = crop(image_set)
        image = transformed.get_image(key='imageTensor').first()
        assert image.shape == (5, 40, 40, 1)

    def test_crop_random(self):
        if False:
            return 10
        patch = [5, 40, 40]
        data_rdd = self.sc.parallelize([self.sample])
        image_set = DistributedImageSet(image_rdd=data_rdd)
        crop = RandomCrop3D(patch[0], patch[1], patch[2])
        transformed = image_set.transform(crop)
        image = transformed.get_image(key='imageTensor').first()
        assert image.shape == (5, 40, 40, 1)

    def test_crop_centor(self):
        if False:
            print('Hello World!')
        patch = [5, 40, 40]
        data_rdd = self.sc.parallelize([self.sample])
        image_set = DistributedImageSet(image_rdd=data_rdd)
        crop = CenterCrop3D(patch[0], patch[1], patch[2])
        transformed = image_set.transform(crop)
        image = transformed.get_image(key='imageTensor').first()
        assert image.shape == (5, 40, 40, 1)

    def test_rotation_1(self):
        if False:
            while True:
                i = 10
        crop = CenterCrop3D(5, 40, 40)
        data_rdd = self.sc.parallelize([self.sample])
        image_set = DistributedImageSet(image_rdd=data_rdd)
        cropped = image_set.transform(crop)
        yaw = 0.0
        pitch = 0.0
        roll = pi / 6
        rotate_30 = Rotate3D([yaw, pitch, roll])
        transformed = cropped.transform(rotate_30)
        image = transformed.get_image(key='imageTensor').first()
        assert image.shape == (5, 40, 40, 1)

    def test_rotation_2(self):
        if False:
            for i in range(10):
                print('nop')
        crop = CenterCrop3D(5, 40, 40)
        data_rdd = self.sc.parallelize([self.sample])
        image_set = DistributedImageSet(image_rdd=data_rdd)
        cropped = image_set.transform(crop)
        yaw = 0.0
        pitch = 0.0
        roll = pi / 2
        rotate_90 = Rotate3D([yaw, pitch, roll])
        transformed = cropped.transform(rotate_90)
        image = transformed.get_image(key='imageTensor').first()
        assert image.shape == (5, 40, 40, 1)

    def test_affine_transformation(self):
        if False:
            while True:
                i = 10
        data_rdd = self.sc.parallelize([self.sample])
        image_set = DistributedImageSet(image_rdd=data_rdd)
        crop = CenterCrop3D(5, 40, 40)
        cropped = image_set.transform(crop)
        affine = AffineTransform3D(np.random.rand(3, 3))
        transformed = cropped.transform(affine)
        image = transformed.get_image(key='imageTensor').first()
        assert image.shape == (5, 40, 40, 1)

    def test_pipeline(self):
        if False:
            while True:
                i = 10
        data_rdd = self.sc.parallelize([self.sample])
        image_set = DistributedImageSet(image_rdd=data_rdd)
        yaw = 0.0
        pitch = 0.0
        roll = pi / 6
        transformer = ChainedPreprocessing([CenterCrop3D(5, 40, 40), Rotate3D([yaw, pitch, roll])])
        transformed = transformer(image_set)
        assert transformed.is_distributed() is True
        image = transformed.get_image(key='imageTensor').first()
        assert image.shape == (5, 40, 40, 1)
if __name__ == '__main__':
    pytest.main([__file__])