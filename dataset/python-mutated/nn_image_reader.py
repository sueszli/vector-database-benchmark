import sys
from bigdl.dllib.utils.file_utils import callZooFunc
if sys.version >= '3':
    long = int
    unicode = str

class NNImageReader:
    """
    Primary DataFrame-based image loading interface, defining API to read images from files
    to DataFrame.
    """

    @staticmethod
    def readImages(path, sc=None, minPartitions=1, resizeH=-1, resizeW=-1, image_codec=-1, bigdl_type='float'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Read the directory of images into DataFrame from the local or remote source.\n        :param path Directory to the input data files, the path can be comma separated paths as the\n                list of inputs. Wildcards path are supported similarly to sc.binaryFiles(path).\n        :param min_partitions A suggestion value of the minimal splitting number for input data.\n        :param resizeH height after resize, by default is -1 which will not resize the image\n        :param resizeW width after resize, by default is -1 which will not resize the image\n        :param image_codec specifying the color type of a loaded image, same as in OpenCV.imread.\n               By default is Imgcodecs.CV_LOAD_IMAGE_UNCHANGED(-1).\n               >0 Return a 3-channel color image. Note In the current implementation the\n                  alpha channel, if any, is stripped from the output image. Use negative value\n                  if you need the alpha channel.\n               =0 Return a grayscale image.\n               <0 Return the loaded image as is (with alpha channel if any).\n        :return DataFrame with a single column "image"; Each record in the column represents\n                one image record: Row (uri, height, width, channels, CvType, bytes).\n        '
        df = callZooFunc(bigdl_type, 'nnReadImage', path, sc, minPartitions, resizeH, resizeW, image_codec)
        df._sc._jsc = sc._jsc
        return df