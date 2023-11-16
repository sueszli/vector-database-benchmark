from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch, torch_neuronx
hf_model = 'j-hartmann/emotion-english-distilroberta-base'
neuron_model = './sentiment_neuron.pt'
model = AutoModelForSequenceClassification.from_pretrained(hf_model)
tokenizer = AutoTokenizer.from_pretrained(hf_model)
sequence_0 = 'The company HuggingFace is based in New York City'
sequence_1 = "HuggingFace's headquarters are situated in Manhattan"
example_inputs = tokenizer.encode_plus(sequence_0, sequence_1, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
neuron_inputs = (example_inputs['input_ids'], example_inputs['attention_mask'])
n_model = torch_neuronx.trace(model, neuron_inputs)
n_model.save(neuron_model)
print(f'Saved Neuron-compiled model {neuron_model}')
from fastapi import FastAPI
import torch
from ray import serve
from ray.serve.handle import DeploymentHandle
app = FastAPI()
hf_model = 'j-hartmann/emotion-english-distilroberta-base'
neuron_model = './sentiment_neuron.pt'

@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:

    def __init__(self, bert_base_model_handle: DeploymentHandle) -> None:
        if False:
            return 10
        self.handle = bert_base_model_handle

    @app.get('/infer')
    async def infer(self, sentence: str):
        return await self.handle.infer.remote(sentence)

@serve.deployment(ray_actor_options={'resources': {'neuron_cores': 1}}, autoscaling_config={'min_replicas': 1, 'max_replicas': 2})
class BertBaseModel:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        import torch, torch_neuronx
        from transformers import AutoTokenizer
        self.model = torch.jit.load(neuron_model)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
        self.classmap = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'neutral', 5: 'sadness', 6: 'surprise'}

    def infer(self, sentence: str):
        if False:
            return 10
        inputs = self.tokenizer.encode_plus(sentence, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        output = self.model(*(inputs['input_ids'], inputs['attention_mask']))
        class_id = torch.argmax(output['logits'], dim=1).item()
        return self.classmap[class_id]
entrypoint = APIIngress.bind(BertBaseModel.bind())
if __name__ == '__main__':
    import requests
    import ray
    ray.init(resources={'neuron_cores': 2})
    serve.run(entrypoint)
    prompt = 'Ray is super cool.'
    resp = requests.get(f'http://127.0.0.1:8000/infer?sentence={prompt}')
    print(resp.status_code, resp.json())
    assert resp.status_code == 200