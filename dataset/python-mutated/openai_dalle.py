import openai
from superagi.config.config import get_config
from superagi.image_llms.base_image_llm import BaseImageLlm

class OpenAiDalle(BaseImageLlm):

    def __init__(self, api_key, image_model=None, number_of_results=1):
        if False:
            return 10
        '\n        Args:\n            api_key (str): The OpenAI API key.\n            image_model (str): The image model.\n            number_of_results (int): The number of results.\n        '
        self.number_of_results = number_of_results
        self.api_key = api_key
        self.image_model = image_model
        openai.api_key = api_key
        openai.api_base = get_config('OPENAI_API_BASE', 'https://api.openai.com/v1')

    def get_image_model(self):
        if False:
            while True:
                i = 10
        '\n        Returns:\n            str: The image model.\n        '
        return self.image_model

    def generate_image(self, prompt: str, size: int=512):
        if False:
            while True:
                i = 10
        '\n        Call the OpenAI image API.\n\n        Args:\n            prompt (str): The prompt.\n            size (int): The size.\n            num (int): The number of images.\n\n        Returns:\n            dict: The response.\n        '
        response = openai.Image.create(prompt=prompt, n=self.number_of_results, size=f'{size}x{size}')
        return response