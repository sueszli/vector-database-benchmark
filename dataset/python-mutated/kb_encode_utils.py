import os
from functools import partial
from glob import glob
import faiss
from datasets import Features, Sequence, Value, concatenate_datasets, load_dataset, load_from_disk
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast

def split_text(text, n=100, character=' '):
    if False:
        for i in range(10):
            print('nop')
    'Split the text every ``n``-th occurrence of ``character``'
    text = text.split(character)
    return [character.join(text[i:i + n]).strip() for i in range(0, len(text), n)]

def split_documents(documents):
    if False:
        print('Hello World!')
    'Split documents into passages'
    (titles, texts) = ([], [])
    for (title, text) in zip(documents['title'], documents['text']):
        if text is not None:
            for passage in split_text(text):
                titles.append(title if title is not None else '')
                texts.append(passage)
    return {'title': titles, 'text': texts}

def embed_update(ctx_encoder, total_processes, device, process_num, shard_dir, csv_path):
    if False:
        return 10
    kb_dataset = load_dataset('csv', data_files=[csv_path], split='train', delimiter='\t', column_names=['title', 'text'])
    kb_dataset = kb_dataset.map(split_documents, batched=True, num_proc=1)
    kb_list = [kb_dataset.shard(total_processes, i, contiguous=True) for i in range(total_processes)]
    data_shrad = kb_list[process_num]
    arrow_folder = 'data_' + str(process_num)
    passages_path = os.path.join(shard_dir, arrow_folder)
    context_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained('facebook/dpr-ctx_encoder-multiset-base')
    ctx_encoder = ctx_encoder.to(device=device)

    def embed(documents: dict, ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizerFast, device) -> dict:
        if False:
            while True:
                i = 10
        'Compute the DPR embeddings of document passages'
        input_ids = ctx_tokenizer(documents['title'], documents['text'], truncation=True, padding='longest', return_tensors='pt')['input_ids']
        embeddings = ctx_encoder(input_ids.to(device=device), return_dict=True).pooler_output
        return {'embeddings': embeddings.detach().cpu().numpy()}
    new_features = Features({'text': Value('string'), 'title': Value('string'), 'embeddings': Sequence(Value('float32'))})
    dataset = data_shrad.map(partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=context_tokenizer, device=device), batched=True, batch_size=16, features=new_features)
    dataset.save_to_disk(passages_path)

def add_index(shard_dir, index_path):
    if False:
        for i in range(10):
            print('nop')
    data_shard_list = []
    for shard_address in glob(str(shard_dir) + '/*/'):
        data_shard_list.append(load_from_disk(shard_address))
    concat = concatenate_datasets(data_shard_list)
    faiss.omp_set_num_threads(96)
    index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)
    concat.add_faiss_index('embeddings', custom_index=index)
    concat.get_index('embeddings').save(index_path)