import os
from collections import deque
import torch
from torch.utils.data import Dataset

class CNNDMDataset(Dataset):
    """Abstracts the dataset used to train seq2seq models.

    The class will process the documents that are located in the specified
    folder. The preprocessing will work on any document that is reasonably
    formatted. On the CNN/DailyMail dataset it will extract both the story
    and the summary.

    CNN/Daily News:

    The CNN/Daily News raw datasets are downloaded from [1]. The stories are
    stored in different files; the summary appears at the end of the story as
    sentences that are prefixed by the special `@highlight` line. To process
    the data, untar both datasets in the same folder, and pass the path to this
    folder as the "data_dir argument. The formatting code was inspired by [2].

    [1] https://cs.nyu.edu/~kcho/
    [2] https://github.com/abisee/cnn-dailymail/
    """

    def __init__(self, path='', prefix='train'):
        if False:
            while True:
                i = 10
        'We initialize the class by listing all the documents to summarize.\n        Files are not read in memory due to the size of some datasets (like CNN/DailyMail).\n        '
        assert os.path.isdir(path)
        self.documents = []
        story_filenames_list = os.listdir(path)
        for story_filename in story_filenames_list:
            if 'summary' in story_filename:
                continue
            path_to_story = os.path.join(path, story_filename)
            if not os.path.isfile(path_to_story):
                continue
            self.documents.append(path_to_story)

    def __len__(self):
        if False:
            while True:
                i = 10
        'Returns the number of documents.'
        return len(self.documents)

    def __getitem__(self, idx):
        if False:
            while True:
                i = 10
        document_path = self.documents[idx]
        document_name = document_path.split('/')[-1]
        with open(document_path, encoding='utf-8') as source:
            raw_story = source.read()
            (story_lines, summary_lines) = process_story(raw_story)
        return (document_name, story_lines, summary_lines)

def process_story(raw_story):
    if False:
        for i in range(10):
            print('nop')
    'Extract the story and summary from a story file.\n\n    Arguments:\n        raw_story (str): content of the story file as an utf-8 encoded string.\n\n    Raises:\n        IndexError: If the story is empty or contains no highlights.\n    '
    nonempty_lines = list(filter(lambda x: len(x) != 0, [line.strip() for line in raw_story.split('\n')]))
    nonempty_lines = [_add_missing_period(line) for line in nonempty_lines]
    story_lines = []
    lines = deque(nonempty_lines)
    while True:
        try:
            element = lines.popleft()
            if element.startswith('@highlight'):
                break
            story_lines.append(element)
        except IndexError:
            return (story_lines, [])
    summary_lines = list(filter(lambda t: not t.startswith('@highlight'), lines))
    return (story_lines, summary_lines)

def _add_missing_period(line):
    if False:
        while True:
            i = 10
    END_TOKENS = ['.', '!', '?', '...', "'", '`', '"', '’', '’', ')']
    if line.startswith('@highlight'):
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + '.'

def truncate_or_pad(sequence, block_size, pad_token_id):
    if False:
        for i in range(10):
            print('nop')
    "Adapt the source and target sequences' lengths to the block size.\n    If the sequence is shorter we append padding token to the right of the sequence.\n    "
    if len(sequence) > block_size:
        return sequence[:block_size]
    else:
        sequence.extend([pad_token_id] * (block_size - len(sequence)))
        return sequence

def build_mask(sequence, pad_token_id):
    if False:
        print('Hello World!')
    'Builds the mask. The attention mechanism will only attend to positions\n    with value 1.'
    mask = torch.ones_like(sequence)
    idx_pad_tokens = sequence == pad_token_id
    mask[idx_pad_tokens] = 0
    return mask

def encode_for_summarization(story_lines, summary_lines, tokenizer):
    if False:
        for i in range(10):
            print('nop')
    'Encode the story and summary lines, and join them\n    as specified in [1] by using `[SEP] [CLS]` tokens to separate\n    sentences.\n    '
    story_lines_token_ids = [tokenizer.encode(line) for line in story_lines]
    story_token_ids = [token for sentence in story_lines_token_ids for token in sentence]
    summary_lines_token_ids = [tokenizer.encode(line) for line in summary_lines]
    summary_token_ids = [token for sentence in summary_lines_token_ids for token in sentence]
    return (story_token_ids, summary_token_ids)

def compute_token_type_ids(batch, separator_token_id):
    if False:
        for i in range(10):
            print('nop')
    'Segment embeddings as described in [1]\n\n    The values {0,1} were found in the repository [2].\n\n    Attributes:\n        batch: torch.Tensor, size [batch_size, block_size]\n            Batch of input.\n        separator_token_id: int\n            The value of the token that separates the segments.\n\n    [1] Liu, Yang, and Mirella Lapata. "Text summarization with pretrained encoders."\n        arXiv preprint arXiv:1908.08345 (2019).\n    [2] https://github.com/nlpyang/PreSumm (/src/prepro/data_builder.py, commit fac1217)\n    '
    batch_embeddings = []
    for sequence in batch:
        sentence_num = -1
        embeddings = []
        for s in sequence:
            if s == separator_token_id:
                sentence_num += 1
            embeddings.append(sentence_num % 2)
        batch_embeddings.append(embeddings)
    return torch.tensor(batch_embeddings)