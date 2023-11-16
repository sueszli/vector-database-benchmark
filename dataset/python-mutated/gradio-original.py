import gradio as gr
from transformers import pipeline
import requests
generator1 = pipeline('text-generation', model='gpt2')
generator2 = pipeline('text-generation', model='distilgpt2')

def model1(text):
    if False:
        return 10
    generated_list = generator1(text, do_sample=True, min_length=20, max_length=100)
    generated = generated_list[0]['generated_text']
    return generated

def model2(text):
    if False:
        i = 10
        return i + 15
    generated_list = generator2(text, do_sample=True, min_length=20, max_length=100)
    generated = generated_list[0]['generated_text']
    return generated
demo = gr.Interface(lambda text: f'{model1(text)}\n------------\n{model2(text)}', 'textbox', 'textbox')
demo.launch(prevent_thread_lock=True)
response = requests.post('http://127.0.0.1:7860/api/predict/', json={'data': ['My name is Lewis']})
assert response.status_code == 200
print('gradio-original.py: Response from example code is', response.json()['data'])