import torch
import torch.nn as nn
from transformers import AutoModel
import numpy as np

class ClassificationHead(nn.Module):

    def __init__(self, hidden_size, num_labels, dropout_rate=0.1) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled_output):
        if False:
            return 10
        return self.classifier(self.dropout(pooled_output))

class StyleEncoder(nn.Module):

    def __init__(self, config) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.bert_path)
        self.pitch_clf = ClassificationHead(config.bert_hidden_size, config.pitch_n_labels)
        self.speed_clf = ClassificationHead(config.bert_hidden_size, config.speed_n_labels)
        self.energy_clf = ClassificationHead(config.bert_hidden_size, config.energy_n_labels)
        self.emotion_clf = ClassificationHead(config.bert_hidden_size, config.emotion_n_labels)
        self.style_embed_proj = nn.Linear(config.bert_hidden_size, config.style_dim)

    def forward(self, input_ids, token_type_ids, attention_mask):
        if False:
            return 10
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs['pooler_output']
        pitch_outputs = self.pitch_clf(pooled_output)
        speed_outputs = self.speed_clf(pooled_output)
        energy_outputs = self.energy_clf(pooled_output)
        emotion_outputs = self.emotion_clf(pooled_output)
        pred_style_embed = self.style_embed_proj(pooled_output)
        res = {'pooled_output': pooled_output, 'pitch_outputs': pitch_outputs, 'speed_outputs': speed_outputs, 'energy_outputs': energy_outputs, 'emotion_outputs': emotion_outputs}
        return res

class StylePretrainLoss(nn.Module):

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs, outputs):
        if False:
            return 10
        pitch_loss = self.loss(outputs['pitch_outputs'], inputs['pitch'])
        energy_loss = self.loss(outputs['energy_outputs'], inputs['energy'])
        speed_loss = self.loss(outputs['speed_outputs'], inputs['speed'])
        emotion_loss = self.loss(outputs['emotion_outputs'], inputs['emotion'])
        return {'pitch_loss': pitch_loss, 'energy_loss': energy_loss, 'speed_loss': speed_loss, 'emotion_loss': emotion_loss}

class StylePretrainLoss2(StylePretrainLoss):

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs, outputs):
        if False:
            while True:
                i = 10
        res = super().forward(inputs, outputs)
        speaker_loss = self.loss(outputs['speaker_outputs'], inputs['speaker'])
        res['speaker_loss'] = speaker_loss
        return res

def flat_accuracy(preds, labels):
    if False:
        while True:
            i = 10
    '\n    Function to calculate the accuracy of our predictions vs labels\n    '
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)