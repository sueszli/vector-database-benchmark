from typing import TYPE_CHECKING
import torch
from ..models.auto import AutoModelForVisualQuestionAnswering, AutoProcessor
from ..utils import requires_backends
from .base import PipelineTool
if TYPE_CHECKING:
    from PIL import Image

class ImageQuestionAnsweringTool(PipelineTool):
    default_checkpoint = 'dandelin/vilt-b32-finetuned-vqa'
    description = 'This is a tool that answers a question about an image. It takes an input named `image` which should be the image containing the information, as well as a `question` which should be the question in English. It returns a text that is the answer to the question.'
    name = 'image_qa'
    pre_processor_class = AutoProcessor
    model_class = AutoModelForVisualQuestionAnswering
    inputs = ['image', 'text']
    outputs = ['text']

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        requires_backends(self, ['vision'])
        super().__init__(*args, **kwargs)

    def encode(self, image: 'Image', question: str):
        if False:
            return 10
        return self.pre_processor(image, question, return_tensors='pt')

    def forward(self, inputs):
        if False:
            return 10
        with torch.no_grad():
            return self.model(**inputs).logits

    def decode(self, outputs):
        if False:
            print('Hello World!')
        idx = outputs.argmax(-1).item()
        return self.model.config.id2label[idx]