from bigdl.dllib.feature.transform.vision.image import ImageFrame
from bigdl.dllib.utils.common import *
from bigdl.dllib.utils.file_utils import callZooFunc
from bigdl.dllib.utils.log4Error import *

class ImageSet(JavaValue):
    """
    ImageSet wraps a set of ImageFeature
    """

    def __init__(self, jvalue, bigdl_type='float'):
        if False:
            print('Hello World!')
        self.value = jvalue
        self.bigdl_type = bigdl_type
        if self.is_local():
            self.image_set = LocalImageSet(jvalue=self.value)
        else:
            self.image_set = DistributedImageSet(jvalue=self.value)

    def is_local(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        whether this is a LocalImageSet\n        '
        return callZooFunc(self.bigdl_type, 'isLocalImageSet', self.value)

    def is_distributed(self):
        if False:
            while True:
                i = 10
        '\n        whether this is a DistributedImageSet\n        '
        return callZooFunc(self.bigdl_type, 'isDistributedImageSet', self.value)

    @property
    def label_map(self):
        if False:
            return 10
        '\n        :return: the labelMap of this ImageSet, None if the ImageSet does not have a labelMap\n        '
        return callZooFunc(self.bigdl_type, 'imageSetGetLabelMap', self.value)

    @classmethod
    def read(cls, path, sc=None, min_partitions=1, resize_height=-1, resize_width=-1, image_codec=-1, with_label=False, one_based_label=True, bigdl_type='float'):
        if False:
            return 10
        '\n        Read images as Image Set\n        if sc is defined, Read image as DistributedImageSet from local file system or HDFS\n        if sc is null, Read image as LocalImageSet from local file system\n        :param path path to read images\n        if sc is defined, path can be local or HDFS. Wildcard character are supported.\n        if sc is null, path is local directory/image file/image file with wildcard character\n\n        if withLabel is set to true, path should be a directory that have two levels. The\n        first level is class folders, and the second is images. All images belong to a same\n        class should be put into the same class folder. So each image in the path is labeled by the\n        folder it belongs.\n\n        :param sc SparkContext\n        :param min_partitions A suggestion value of the minimal splitting number for input data.\n        :param resize_height height after resize, by default is -1 which will not resize the image\n        :param resize_width width after resize, by default is -1 which will not resize the image\n        :param image_codec specifying the color type of a loaded image, same as in OpenCV.imread.\n               By default is Imgcodecs.CV_LOAD_IMAGE_UNCHANGED(-1)\n        :param with_label whether to treat folders in the path as image classification labels\n               and read the labels into ImageSet.\n        :param one_based_label whether to use one based label\n        :return ImageSet\n        '
        return ImageSet(jvalue=callZooFunc(bigdl_type, 'readImageSet', path, sc, min_partitions, resize_height, resize_width, image_codec, with_label, one_based_label))

    @classmethod
    def from_image_frame(cls, image_frame, bigdl_type='float'):
        if False:
            while True:
                i = 10
        return ImageSet(jvalue=callZooFunc(bigdl_type, 'imageFrameToImageSet', image_frame))

    @classmethod
    def from_rdds(cls, image_rdd, label_rdd=None, bigdl_type='float'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a ImageSet from rdds of ndarray.\n\n        :param image_rdd: a rdd of ndarray, each ndarray should has dimension of 3 or 4 (3D images)\n        :param label_rdd: a rdd of ndarray\n        :return: a DistributedImageSet\n        '
        image_rdd = image_rdd.map(lambda x: JTensor.from_ndarray(x))
        if label_rdd is not None:
            label_rdd = label_rdd.map(lambda x: JTensor.from_ndarray(x))
        return ImageSet(jvalue=callZooFunc(bigdl_type, 'createDistributedImageSet', image_rdd, label_rdd), bigdl_type=bigdl_type)

    def transform(self, transformer):
        if False:
            i = 10
            return i + 15
        '\n        transformImageSet\n        '
        return ImageSet(callZooFunc(self.bigdl_type, 'transformImageSet', transformer, self.value), self.bigdl_type)

    def get_image(self, key='floats', to_chw=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        get image from ImageSet\n        '
        return self.image_set.get_image(key, to_chw)

    def get_label(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        get label from ImageSet\n        '
        return self.image_set.get_label()

    def get_predict(self, key='predict'):
        if False:
            return 10
        '\n        get prediction from ImageSet\n        '
        return self.image_set.get_predict(key)

    def to_image_frame(self, bigdl_type='float'):
        if False:
            i = 10
            return i + 15
        return ImageFrame(callZooFunc(bigdl_type, 'imageSetToImageFrame', self.value), bigdl_type)

class LocalImageSet(ImageSet):
    """
    LocalImageSet wraps a list of ImageFeature
    """

    def __init__(self, image_list=None, label_list=None, jvalue=None, bigdl_type='float'):
        if False:
            i = 10
            return i + 15
        invalidInputError(jvalue or image_list, 'jvalue and image_list cannot be None in the same time')
        if jvalue:
            self.value = jvalue
        else:
            image_tensor_list = list(map(lambda image: JTensor.from_ndarray(image), image_list))
            label_tensor_list = list(map(lambda label: JTensor.from_ndarray(label), label_list)) if label_list else None
            self.value = callZooFunc(bigdl_type, JavaValue.jvm_class_constructor(self), image_tensor_list, label_tensor_list)
        self.bigdl_type = bigdl_type

    def get_image(self, key='floats', to_chw=True):
        if False:
            while True:
                i = 10
        '\n        get image list from ImageSet\n        '
        tensors = callZooFunc(self.bigdl_type, 'localImageSetToImageTensor', self.value, key, to_chw)
        return list(map(lambda tensor: tensor.to_ndarray(), tensors))

    def get_label(self):
        if False:
            return 10
        '\n        get label list from ImageSet\n        '
        labels = callZooFunc(self.bigdl_type, 'localImageSetToLabelTensor', self.value)
        return map(lambda tensor: tensor.to_ndarray(), labels)

    def get_predict(self, key='predict'):
        if False:
            i = 10
            return i + 15
        '\n        get prediction list from ImageSet\n        '
        predicts = callZooFunc(self.bigdl_type, 'localImageSetToPredict', self.value, key)
        return list(map(lambda predict: (predict[0], list(map(lambda x: x.to_ndarray(), predict[1]))) if predict[1] else (predict[0], None), predicts))

class DistributedImageSet(ImageSet):
    """
    DistributedImageSet wraps an RDD of ImageFeature
    """

    def __init__(self, image_rdd=None, label_rdd=None, jvalue=None, bigdl_type='float'):
        if False:
            for i in range(10):
                print('nop')
        invalidInputError(jvalue or image_rdd, 'jvalue and image_rdd cannot be None in the same time')
        if jvalue:
            self.value = jvalue
        else:
            image_tensor_rdd = image_rdd.map(lambda image: JTensor.from_ndarray(image))
            label_tensor_rdd = label_rdd.map(lambda label: JTensor.from_ndarray(label)) if label_rdd else None
            self.value = callZooFunc(bigdl_type, JavaValue.jvm_class_constructor(self), image_tensor_rdd, label_tensor_rdd)
        self.bigdl_type = bigdl_type

    def get_image(self, key='floats', to_chw=True):
        if False:
            i = 10
            return i + 15
        '\n        get image rdd from ImageSet\n        '
        tensor_rdd = callZooFunc(self.bigdl_type, 'distributedImageSetToImageTensorRdd', self.value, key, to_chw)
        return tensor_rdd.map(lambda tensor: tensor.to_ndarray())

    def get_label(self):
        if False:
            while True:
                i = 10
        '\n        get label rdd from ImageSet\n        '
        tensor_rdd = callZooFunc(self.bigdl_type, 'distributedImageSetToLabelTensorRdd', self.value)
        return tensor_rdd.map(lambda tensor: tensor.to_ndarray())

    def get_predict(self, key='predict'):
        if False:
            i = 10
            return i + 15
        '\n        get prediction rdd from ImageSet\n        '
        predicts = callZooFunc(self.bigdl_type, 'distributedImageSetToPredict', self.value, key)
        return predicts.map(lambda predict: (predict[0], list(map(lambda x: x.to_ndarray(), predict[1]))) if predict[1] else (predict[0], None))