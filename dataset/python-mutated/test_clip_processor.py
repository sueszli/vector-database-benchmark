import os
import tempfile
import urllib
from PIL import Image
from embedchain.models.clip_processor import ClipProcessor

class TestClipProcessor:

    def test_load_model(self):
        if False:
            return 10
        model = ClipProcessor.load_model()
        assert model is not None

    def test_get_image_features(self):
        if False:
            while True:
                i = 10
        with tempfile.TemporaryDirectory() as tmp_dir:
            urllib.request.urlretrieve('https://upload.wikimedia.org/wikipedia/en/a/a9/Example.jpg', 'image.jpg')
            image = Image.open('image.jpg')
            image.save(os.path.join(tmp_dir, 'image.jpg'))
            model = ClipProcessor.load_model()
            ClipProcessor.get_image_features(os.path.join(tmp_dir, 'image.jpg'), model)
            os.remove(os.path.join(tmp_dir, 'image.jpg'))
            os.remove('image.jpg')

    def test_get_text_features(self):
        if False:
            i = 10
            return i + 15
        query = 'This is a text query.'
        text_features = ClipProcessor.get_text_features(query)
        assert text_features is not None
        assert isinstance(text_features, list)
        assert len(text_features) == 512