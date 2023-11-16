import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model
config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none')
model = AutoModelForVision2Seq.from_pretrained('Salesforce/blip2-opt-2.7b', load_in_8bit=True)
processor = AutoProcessor.from_pretrained('Salesforce/blip2-opt-2.7b')
model = get_peft_model(model, config)
model.print_trainable_parameters()
dataset = load_dataset('ybelkada/football-dataset', split='train')

class ImageCaptioningDataset(Dataset):

    def __init__(self, dataset, processor):
        if False:
            return 10
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.dataset)

    def __getitem__(self, idx):
        if False:
            while True:
                i = 10
        item = self.dataset[idx]
        encoding = self.processor(images=item['image'], padding='max_length', return_tensors='pt')
        encoding = {k: v.squeeze() for (k, v) in encoding.items()}
        encoding['text'] = item['text']
        return encoding

def collator(batch):
    if False:
        while True:
            i = 10
    processed_batch = {}
    for key in batch[0].keys():
        if key != 'text':
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer([example['text'] for example in batch], padding=True, return_tensors='pt')
            processed_batch['input_ids'] = text_inputs['input_ids']
            processed_batch['attention_mask'] = text_inputs['attention_mask']
    return processed_batch
train_dataset = ImageCaptioningDataset(dataset, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=collator)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-05)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.train()
for epoch in range(50):
    print('Epoch:', epoch)
    for (idx, batch) in enumerate(train_dataloader):
        input_ids = batch.pop('input_ids').to(device)
        pixel_values = batch.pop('pixel_values').to(device, torch.float16)
        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)
        loss = outputs.loss
        print('Loss:', loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if idx % 10 == 0:
            generated_output = model.generate(pixel_values=pixel_values)
            print(processor.batch_decode(generated_output, skip_special_tokens=True))