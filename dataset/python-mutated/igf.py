import copy
import logging
import random
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, GPT2LMHeadModel, get_linear_schedule_with_warmup
logger = logging.getLogger(__name__)

def set_seed(seed):
    if False:
        for i in range(10):
            print('nop')
    '\n    For reproducible training\n\n    Args:\n        seed: A seed for reproducible training\n\n    '
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_perplexity(model, test_data, context_len):
    if False:
        i = 10
        return i + 15
    '\n    Computes perplexity of the transformer model on data in test_data\n\n    Args:\n        model: Pre-trained GPT2 model\n        test_data: Data on which perplexity calculation is required\n        context_len: The maximum total input sequence length after tokenization. Sequences longer\n                     than this will be truncated, sequences shorter will be padded\n\n    Returns:\n        Perplexity on input test data\n\n    '
    model.eval()
    device = next(model.parameters()).device
    eval_batch_size = 1
    context = torch.zeros((eval_batch_size, context_len), dtype=torch.long, device=device)
    eval_dataloader = DataLoader(test_data, shuffle=False, batch_size=eval_batch_size)
    eval_loss = torch.zeros(1, device=device)
    nb_eval_examples = 0
    for batch in eval_dataloader:
        batch.to(device)
        context.zero_()
        for i in range(eval_batch_size):
            context[i, :] = batch[i]
        outputs = model(context, labels=context)
        eval_loss += outputs[0].sum().item()
        nb_eval_examples += batch.size(0)
    eval_loss = eval_loss / nb_eval_examples
    perplexity = torch.exp(eval_loss)
    model.train()
    return perplexity

def load_gpt2(model_name='gpt2'):
    if False:
        while True:
            i = 10
    '\n    load original gpt2 and save off for quicker loading\n\n    Args:\n        model_name: GPT-2\n\n    Returns:\n        GPT-2 model\n\n    '
    model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True)
    torch.save(model.state_dict(), model_name + 'local.pt')
    return model

def recopy_gpt2(orig_model, device, max_steps):
    if False:
        for i in range(10):
            print('nop')
    '\n    Reset the model to the original pretrained GPT-2 weights after each iteration\n\n    Args:\n        orig_model: Original pretrained GPT-2 model imported from Transformers library\n        device: CPU/GPU\n        max_steps: number of training steps\n\n    Returns:\n        Original PreTrained GPT-2 model,\n        lm_optimizer: Adam optimizer with Decoupled weight decay\n        lm_scheduler: linear scheduler with the appropriate schedule\n\n    '
    model = copy.deepcopy(orig_model)
    model.to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{'params': [p for (n, p) in model.named_parameters() if not any((nd in n for nd in no_decay))], 'weight_decay': 0.0}, {'params': [p for (n, p) in model.named_parameters() if any((nd in n for nd in no_decay))], 'weight_decay': 0.0}]
    lm_optimizer = AdamW(optimizer_grouped_parameters, lr=5e-05, eps=1e-08)
    lm_scheduler = get_linear_schedule_with_warmup(lm_optimizer, 0, max_steps)
    torch.cuda.empty_cache()
    return (model, lm_optimizer, lm_scheduler)

def intermittent_save(contexts, real_perps, past_perps, filename):
    if False:
        i = 10
        return i + 15
    '\n    save the perplexity differences to filename\n\n    Args:\n        contexts: Example on which the perplexity is calculated\n        real_perps: Perplexity after back-propagating on the selected context\n        past_perps: Perplexity of model before training on the context\n        filename: File to store perplexity differences\n\n    Returns:\n        file with perplexity differences\n\n    '
    avg = np.array(real_perps).mean()
    std = np.array(real_perps).std()
    perp_diff = (real_perps - avg) / std
    data_final = list(zip(contexts, perp_diff, past_perps))
    joblib.dump(data_final, filename)

def collect_objective_set(model, orig_perp, context_len, train_data, objective_set, max_steps, device, filename='dev.jbl', recopy_model=recopy_gpt2):
    if False:
        i = 10
        return i + 15
    '\n    Collect individual IGF values from pre-trained transformer model\n    max_steps samples of training data to train secondary model\n\n    Args:\n        model: Pre-trained GPT2 model\n        orig_perp: Perplexity of original pretrained GPT-2 model\n        context_len: The maximum total input sequence length after tokenization. Sequences longer\n                    than this will be truncated, sequences shorter will be padded\n        train_data: Data to train model\n        objective_set: Contexts used to create (X,IG(X)) pairs which is the training data for secondary learner\n        max_steps: To calculate training epochs of model\n        device: GPU/CPU\n        filename: To store intermediate perplexity differences\n        recopy_model: Reset the model to the original pretrained GPT-2 weights after each iteration\n\n    Returns:\n        file stored intermediate perplexity differences in intermediate stages\n\n    '
    contexts = []
    real_perps = []
    past_perps = []
    orig_model = copy.deepcopy(model)
    orig_model.to(device='cpu')
    torch.cuda.empty_cache()
    model.train()
    (model, lm_optimizer, lm_scheduler) = recopy_model(orig_model, device, max_steps)
    for step in tqdm(range(max_steps)):
        context = torch.zeros((1, context_len), dtype=torch.long, device=device)
        story = random.choice(train_data)
        start = random.randint(0, len(story[0]) - context_len - 1)
        context[0, :] = story[0][start:start + context_len]
        lm_optimizer.zero_grad()
        outputs = model(context, labels=context)
        lm_loss = outputs[0]
        past_perp = compute_perplexity(model, context, context_len)
        model.train()
        lm_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        lm_optimizer.step()
        lm_scheduler.step()
        real_perp = compute_perplexity(model, objective_set, context_len)
        if step % 1000 == 0 and step > 1:
            intermittent_save(contexts, real_perps, past_perps, filename)
        (model, lm_optimizer, lm_scheduler) = recopy_model(orig_model, device, max_steps)
        past_perps.append(past_perp.item())
        real_perps.append(orig_perp - real_perp.item())
        contexts.append(np.array(context.cpu()))
    intermittent_save(contexts, real_perps, past_perps, filename)

def generate_datasets(context_len, file='data/tokenized_stories_train_wikitext103.jbl', number=100, min_len=1026, trim=True):
    if False:
        while True:
            i = 10
    '\n    Generate objective set and training set\n\n    Args:\n        context_len: The maximum total input sequence length after tokenization. Sequences longer\n                than this will be truncated, sequences shorter will be padded\n        file: Tokenized data split into training set and objective set\n        number: size of objective dataset\n        min_len: minimum length of a context in objective set\n        trim: If True truncate the context if it exceeds context length\n\n    Returns:\n        Generated objective set and training data\n\n\n    '
    data = joblib.load(file)
    print('data loaded')
    objective_set = []
    if trim:
        for (i, example) in enumerate(data):
            if len(example[0]) > min_len:
                start = random.randint(0, len(example[0]) - context_len - 1)
                objective_set.append(example[0, start:start + context_len])
            if len(objective_set) >= number:
                break
        train_data = []
        for j in range(i + 1, len(data)):
            if len(data[j][0]) > min_len:
                train_data.append(data[j])
    else:
        objective_set = data[0:number]
        train_data = data[number:]
    joblib.dump(objective_set, 'objective_set.jbl')
    print('objective set saved')
    return (train_data, objective_set)

def train_secondary_learner(secondary_learner, train_dataset, max_epochs, batch_size, eval_freq=50, igf_model_path='secondary_learner.pt'):
    if False:
        while True:
            i = 10
    '\n    Train the secondary learner (igf_model)\n\n    Args:\n        secondary_learner: secondary learner\n        train_dataset: data to train secondary learner\n        max_epochs: number of epochs to train secondary learner\n        batch_size: batch size of training data of secondary learner\n        eval_freq: secondary model evaluation can be triggered at eval_freq\n        igf_model_path: path to store trained secondary learner\n\n    Returns:\n        Trained secondary learner\n\n    '
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_dataset = train_dataset[:512]
    train_dataset = train_dataset[512:]
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    loss = nn.MSELoss()
    test_loss = nn.MSELoss(reduction='sum')
    secondary_learner.to(device)
    q_optimizer = torch.optim.Adam(secondary_learner.parameters(), lr=1e-05)
    secondary_learner.train()
    best_test_loss = float('inf')
    for epoch in range(int(max_epochs)):
        tr_q_loss = 0.0
        secondary_learner.train()
        for (step, batch) in enumerate(train_dataloader):
            context = batch[0].to(device)
            real_q = batch[1].to(device)
            predicted_q = secondary_learner(context)
            q_optimizer.zero_grad()
            q_loss = loss(predicted_q, real_q.float())
            q_loss.backward()
            q_optimizer.step()
            tr_q_loss += q_loss.item()
            if step % eval_freq == 0 and step > 0 or step + 1 == len(train_dataloader):
                tr_loss = tr_q_loss / (step + 1)
                secondary_learner.eval()
                q_loss2 = 0.0
                sum_q2 = 0.0
                predicted = []
                actual = []
                for (step2, batch2) in enumerate(test_dataloader):
                    features2 = batch2[0].to(device)
                    real_q2 = batch2[1].to(device)
                    predicted_q2 = secondary_learner(features2)
                    q_loss2 += test_loss(predicted_q2, real_q2).item()
                    sum_q2 += torch.sum(predicted_q2).item()
                    for (ei, i) in enumerate(predicted_q2.cpu().detach().numpy()):
                        predicted.append(i.item())
                    for (ei, i) in enumerate(real_q2.cpu().detach().numpy()):
                        actual.append(i.item())
                q_loss2 /= len(test_dataset)
                print('Epoch: ', epoch, 'step: ', step, 'Avg. q:', sum_q2 / len(test_dataset), 'Train Loss: ', tr_loss, 'Test Loss: ', q_loss2)
                if q_loss2 < best_test_loss:
                    joblib.dump((predicted, actual), 'pred_vs_actual.jbl')
                    torch.save(secondary_learner.state_dict(), igf_model_path)
                    best_test_loss = q_loss2
            secondary_learner.train()
    return secondary_learner

class SecondaryLearner(nn.Module):
    """
    Our secondary learner
    """

    def __init__(self, model):
        if False:
            print('Hello World!')
        '\n        We use a simple convolutional network as our secondary learner\n\n        Args:\n            model: Pre-trained GPT2 model\n        '
        super(SecondaryLearner, self).__init__()
        self.embeddings = model.transformer.wte
        self.embeddings.weight = copy.deepcopy(model.transformer.wte.weight)
        self.conv = nn.Conv1d(self.embeddings.weight.size(1), 256, 3, padding=1)
        self.fc = nn.Sequential(nn.Linear(256, 32), nn.Dropout(p=0.1), nn.Linear(32, 32), nn.Linear(32, 1))

    def forward(self, context):
        if False:
            for i in range(10):
                print('nop')
        '\n        Forward pass through the secondary learner\n\n        Args:\n            context: Context input to the secondary learner\n\n        Returns:\n            tensor after squeeze operation\n\n        '
        pooled = torch.max(self.conv(self.embeddings(context).squeeze(1).transpose(1, 2)), 2)[0]
        qs = self.fc(pooled)
        return qs.squeeze(1)

    @classmethod
    def from_pretrained(cls, state_path, model):
        if False:
            while True:
                i = 10
        '\n        Load the secondary learner\n\n        Args:\n            state_path: Path to save secondary learner\n            model: Pretrained GPT-2\n\n        Returns:\n            secondary learner\n        '
        secondary_learner = cls(model)
        state_dict = torch.load(state_path)
        secondary_learner.load_state_dict(state_dict)
        secondary_learner.embeddings = model.transformer.wte
        secondary_learner.embeddings.weight = copy.deepcopy(model.transformer.wte.weight)
        return secondary_learner