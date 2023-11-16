import requests
from transformers import pipeline
from io import BytesIO
from PIL import Image, ImageFile
from typing import Dict
from ray import serve
from ray.dag.input_node import InputNode
from ray.serve.drivers import DAGDriver

@serve.deployment
def downloader(image_url: str) -> ImageFile.ImageFile:
    if False:
        i = 10
        return i + 15
    image_bytes = requests.get(image_url).content
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    return image

@serve.deployment
class ImageClassifier:

    def __init__(self):
        if False:
            return 10
        self.model = pipeline('image-classification', model='google/vit-base-patch16-224')

    def classify(self, image: ImageFile.ImageFile) -> Dict[str, float]:
        if False:
            return 10
        results = self.model(image)
        return {pred['label']: pred['score'] for pred in results}

@serve.deployment
class Translator:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.model = pipeline('translation_en_to_de', model='t5-small')

    def translate(self, dict: Dict[str, float]) -> Dict[str, float]:
        if False:
            print('Hello World!')
        results = {}
        for (label, score) in dict.items():
            translated_label = self.model(label)[0]['translation_text']
            results[translated_label] = score
        return results
with InputNode(input_type=str) as user_input:
    classifier = ImageClassifier.bind()
    translator = Translator.bind()
    downloaded_image = downloader.bind(user_input)
    classes = classifier.classify.bind(downloaded_image)
    translated_classes = translator.translate.bind(classes)
    serve_entrypoint = DAGDriver.bind(translated_classes)