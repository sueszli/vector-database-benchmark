import torch
from transformers import CamembertForMaskedLM, CamembertTokenizer

def fill_mask(masked_input, model, tokenizer, topk=5):
    if False:
        while True:
            i = 10
    assert masked_input.count('<mask>') == 1
    input_ids = torch.tensor(tokenizer.encode(masked_input, add_special_tokens=True)).unsqueeze(0)
    logits = model(input_ids)[0]
    masked_index = (input_ids.squeeze() == tokenizer.mask_token_id).nonzero().item()
    logits = logits[0, masked_index, :]
    prob = logits.softmax(dim=0)
    (values, indices) = prob.topk(k=topk, dim=0)
    topk_predicted_token_bpe = ' '.join([tokenizer.convert_ids_to_tokens(indices[i].item()) for i in range(len(indices))])
    masked_token = tokenizer.mask_token
    topk_filled_outputs = []
    for (index, predicted_token_bpe) in enumerate(topk_predicted_token_bpe.split(' ')):
        predicted_token = predicted_token_bpe.replace('▁', ' ')
        if ' {0}'.format(masked_token) in masked_input:
            topk_filled_outputs.append((masked_input.replace(' {0}'.format(masked_token), predicted_token), values[index].item(), predicted_token))
        else:
            topk_filled_outputs.append((masked_input.replace(masked_token, predicted_token), values[index].item(), predicted_token))
    return topk_filled_outputs
tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
model = CamembertForMaskedLM.from_pretrained('camembert-base')
model.eval()
masked_input = 'Le camembert est <mask> :)'
print(fill_mask(masked_input, model, tokenizer, topk=3))