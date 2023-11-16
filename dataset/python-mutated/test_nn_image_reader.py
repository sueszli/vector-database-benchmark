import os
import pytest
from bigdl.dllib.feature.transform.vision.image import *
from bigdl.dllib.utils.common import *
from pyspark.sql.types import *
from bigdl.dllib.nncontext import *
from bigdl.dllib.nnframes import *

class TestNNImageReader:

    def setup_method(self, method):
        if False:
            return 10
        '\n        setup any state tied to the execution of the given method in a\n        class. setup_method is invoked for every test method of a class.\n        '
        sparkConf = init_spark_conf().setMaster('local[1]').setAppName('TestNNImageReader')
        self.sc = init_nncontext(sparkConf)
        self.resource_path = os.path.join(os.path.split(__file__)[0], '../resources')

    def teardown_method(self, method):
        if False:
            print('Hello World!')
        '\n        teardown any state that was previously setup with a setup_method\n        call.\n        '
        self.sc.stop()

    def test_get_pascal_image(self):
        if False:
            return 10
        image_path = os.path.join(self.resource_path, 'pascal/000025.jpg')
        image_frame = NNImageReader.readImages(image_path, self.sc)
        assert image_frame.count() == 1
        assert type(image_frame).__name__ == 'DataFrame'
        first_row = image_frame.take(1)[0][0]
        assert first_row[0].endswith('pascal/000025.jpg')
        assert first_row[1] == 375
        assert first_row[2] == 500
        assert first_row[3] == 3
        assert first_row[4] == 16
        assert len(first_row[5]) == 562500

    def test_read_image_withOriginColumn(self):
        if False:
            i = 10
            return i + 15
        image_path = os.path.join(self.resource_path, 'pascal/000025.jpg')
        image_frame = NNImageReader.readImages(image_path, self.sc)
        first_row = image_frame.take(1)[0][0]
        image_origin = first_row[0]
        originDF = with_origin_column(image_frame).select('origin')
        origin = originDF.take(1)[0][0]
        assert image_origin == origin
if __name__ == '__main__':
    pytest.main([__file__])