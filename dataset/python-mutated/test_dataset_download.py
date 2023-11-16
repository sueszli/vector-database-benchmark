import os
import unittest
from paddle.dataset.common import DATA_HOME, download, md5file

class TestDataSetDownload(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        flower_path = DATA_HOME + '/flowers/imagelabels.mat'
        if os.path.exists(flower_path):
            os.remove(flower_path)

    def test_download_url(self):
        if False:
            while True:
                i = 10
        LABEL_URL = 'http://paddlemodels.bj.bcebos.com/flowers/imagelabels.mat'
        LABEL_MD5 = 'e0620be6f572b9609742df49c70aed4d'
        catch_exp = False
        try:
            download(LABEL_URL, 'flowers', LABEL_MD5)
        except Exception as e:
            catch_exp = True
        self.assertTrue(not catch_exp)
        file_path = DATA_HOME + '/flowers/imagelabels.mat'
        self.assertTrue(os.path.exists(file_path))
        self.assertTrue(md5file(file_path), LABEL_MD5)
if __name__ == '__main__':
    unittest.main()