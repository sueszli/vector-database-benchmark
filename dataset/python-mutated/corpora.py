"""several datasets with preset arguments"""
import os
import random
from collections import defaultdict
from multiprocessing import Process, Queue
from queue import Empty
import json
import tqdm
from torch.utils import data
from modelscope.models.nlp.mglm.utils import print_rank_0
from .datasets import csv_dataset, json_dataset
from .lazy_loader import LazyLoader
NUM_PROCESSES = 100

def punctuation_standardization(string: str):
    if False:
        while True:
            i = 10
    punctuation_dict = {'“': '"', '”': '"', '’': "'", '‘': "'", '–': '-'}
    for (key, value) in punctuation_dict.items():
        string = string.replace(key, value)
    return string

class KeyDataset(data.Dataset):

    def __init__(self, text_loader, mask_loader, **kwargs):
        if False:
            return 10
        self.texts = text_loader
        self.masks = mask_loader
        self.is_lazy = False
        if isinstance(self.texts, LazyLoader) and isinstance(self.masks, LazyLoader):
            self.text_lens = self.texts.lens
            self.is_lazy = True

    def get_text_len(self, idx):
        if False:
            print('Hello World!')
        return self.text_lens[idx]

    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        text = self.texts[index]
        mask_length = self.masks[index]
        mask = []
        for (i, length) in enumerate(mask_length):
            if i % 2 == 0:
                mask += [0] * length
            else:
                mask += [1] * length
        assert len(text) == len(mask)
        return {'tokens': text, 'loss_masks': mask}

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.texts)

class PromptDataset(data.Dataset):

    def __init__(self, prompt_loader, text_loader, tokenizer=None, to_tokenize=False, **kwargs):
        if False:
            i = 10
            return i + 15
        self.prompts = prompt_loader
        self.texts = text_loader
        self.tokenizer = tokenizer
        self.to_tokenize = to_tokenize
        if isinstance(self.prompts, LazyLoader) and isinstance(self.texts, LazyLoader):
            self.prompt_lens = self.prompts.lens
            self.text_lens = self.texts.lens
            self.is_lazy = True

    def get_text_len(self, idx):
        if False:
            print('Hello World!')
        return self.prompt_lens[idx] + self.text_lens[idx]

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        prompt = self.prompts[index]
        text = self.texts[index]
        if self.to_tokenize:
            prompt = self.tokenizer.EncodeAsIds(prompt).tokenization
            text = self.tokenizer.EncodeAsIds(text).tokenization
        return {'tokens': prompt + text, 'loss_masks': [0] * len(prompt) + [1] * len(text)}

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.prompts)

class DataReader:
    PATH = None
    assert_str = None
    reserve_punct = False
    split_row = True
    TASK_QUEUE_LIMIT = 10000000
    DONE_QUEUE_LIMIT = 10000000

    def tokenize_worker(self, input, output, info, tokenizer, tokenize):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def print_info(self, info):
        if False:
            for i in range(10):
                print('nop')
        pass

    def __init__(self, writers, tokenizer=None, tokenize=False, **kwargs):
        if False:
            print('Hello World!')
        print(self.PATH)
        print(self.assert_str)
        assert os.path.exists(self.PATH), self.assert_str
        print_rank_0(f'Creating dataset from {self.PATH}')
        self.tokenizer = tokenizer
        self.tokenize = tokenize
        self.writers = writers

    def process(self):
        if False:
            while True:
                i = 10
        if os.path.isdir(self.PATH):
            paths = [os.path.join(top, name) for (top, _, names) in os.walk(self.PATH) for name in names]
        else:
            paths = [self.PATH]
        (task_queue, done_queue, info_queue) = (Queue(maxsize=self.TASK_QUEUE_LIMIT), Queue(maxsize=self.DONE_QUEUE_LIMIT), Queue())
        processes = []
        for i in range(NUM_PROCESSES):
            process = Process(target=self.tokenize_worker, args=(task_queue, done_queue, info_queue, self.tokenizer, self.tokenize))
            process.start()
            processes.append(process)

        def read_input_to_queue():
            if False:
                while True:
                    i = 10
            for path in paths:
                print_rank_0(f'Start reading {path}')
                with open(path, encoding='utf-8') as file:
                    items = json.load(file)
                    for item in items:
                        task_queue.put(item)
            print_rank_0('Read input complete')
            for i in range(len(processes)):
                task_queue.put('STOP')
        process = Process(target=read_input_to_queue)
        process.start()
        count = len(processes)
        progress_bar = tqdm.tqdm()
        while True:
            data = done_queue.get()
            if data == 'COMPLETE':
                count -= 1
                if count == 0:
                    break
            else:
                self.write_result(data, self.writers)
                progress_bar.update()
        progress_bar.close()
        self.print_info(info_queue)

    @staticmethod
    def write_result(data, writers):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    @staticmethod
    def get_token_count(contents):
        if False:
            while True:
                i = 10
        return sum(map(len, contents))

    @classmethod
    def process_sample(cls, text, tokenizer, tokenize):
        if False:
            return 10
        if isinstance(text, str) and tokenize:
            if not cls.reserve_punct:
                text = punctuation_standardization(text)
            text = tokenizer.EncodeAsIds(text).tokenization if text else []
        return text

    @staticmethod
    def trim_field(content, max_length):
        if False:
            print('Hello World!')
        if len(content) > max_length:
            content = content[:max_length]
            content += '......'
        return content

    def process_line(self, data, tokenizer, tokenize):
        if False:
            print('Hello World!')
        raise NotImplementedError

class PromptReader(DataReader):
    is_json = True

    def tokenize_worker(self, input, output, info, tokenizer, tokenize):
        if False:
            i = 10
            return i + 15
        for row in iter(input.get, 'STOP'):
            if row:
                if self.is_json:
                    row = row.rstrip()
                    row = json.loads(row)
                (prompts, texts) = self.process_line(row, tokenizer, tokenize)
                for (prompt, text) in zip(prompts, texts):
                    output.put((prompt, text))
        output.put('COMPLETE')

    @staticmethod
    def write_result(data, writers):
        if False:
            print('Hello World!')
        (prompt, text) = data
        writers['prompt'].write(prompt)
        writers['text'].write(text)

class KeyReader(DataReader):
    PATH = '/root/data/wikipedia/wiki-key.txt'
    assert_str = 'make sure to set PATH for wikipedia data_utils/corpora.py'

    def process_line(self, data, tokenizer, tokenize):
        if False:
            i = 10
            return i + 15
        (keys, contents) = (data['key'], data['content'])
        assert len(keys) == len(contents)
        for i in range(1, len(keys)):
            keys[i] = ' ' + keys[i]
        contents = [' ' + content for content in contents]
        keys = [tokenizer.EncodeAsIds(key).tokenization for key in keys]
        contents = [tokenizer.EncodeAsIds(content).tokenization for content in contents]
        summary = sum(keys, [])
        summary_prefix = self.process_sample('Summary: ', tokenizer, tokenize)
        summary_mask = [len(summary_prefix), len(summary)]
        summary = summary_prefix + summary
        (text, text_mask) = ([], [])
        for (key, content) in zip(keys, contents):
            content = content + [tokenizer.get_command('eop').Id]
            text += key
            text += content
            text_mask.append(len(key))
            text_mask.append(len(content))
        return ((summary, summary_mask), (text, text_mask))

    def tokenize_worker(self, input, output, info, tokenizer, tokenize):
        if False:
            return 10
        for row in iter(input.get, 'STOP'):
            data = json.loads(row)
            (summary, content) = self.process_line(data, tokenizer, tokenize)
            output.put((summary, content))
        output.put('COMPLETE')

    @staticmethod
    def write_result(data, writers):
        if False:
            for i in range(10):
                print('nop')
        (summary, content) = data
        writers['text'].write(summary[0])
        writers['mask'].write(summary[1])
        writers['text'].write(content[0])
        writers['mask'].write(content[1])

class zhihu(PromptReader):
    PATH = '/dataset/fd5061f6/data/tokenize_data/zhihu.lazy'
    reserve_punct = True
    assert_str = 'make sure to set PATH for zhihu data_utils/corpora.py'
    qtitle_prefix = '问题：'
    qcontent_prefix = '问题描述：'
    user_prefix = '回答用户：'
    answer_prefix = ' 回答：'

    def process_line(self, data, tokenizer, tokenize):
        if False:
            while True:
                i = 10
        (prompts, texts) = ([], [])
        ans_length = len(data.get('ans-content', ''))
        ans_up = data.get('ans-up-num', '')
        ans_up = int(ans_up) if ans_up else 0
        if ans_length > 100 or ans_up > 1000:
            qtitle = data['q_title']
            qcontent = data['q-content']
            if qcontent is None:
                qcontent = ''
            qcontent = self.trim_field(qcontent, max_length=100)
            user = data.get('user-signature', '')
            prompt = self.qtitle_prefix + qtitle + self.qcontent_prefix + qcontent + self.user_prefix + user + self.answer_prefix
            text = data['ans-content']
            (prompt, text) = (self.process_sample(prompt, tokenizer, tokenize), self.process_sample(text, tokenizer, tokenize))
            prompts.append(prompt)
            texts.append(text)
        return (prompts, texts)

class zhidao(PromptReader):
    PATH = '/root/data/zhidao/zhidao'
    reserve_punct = True
    assert_str = 'make sure to set PATH for zhidao data_utils/corpora.py'
    qtitle_prefix = '问题：'
    qcontent_prefix = '问题描述：'
    answer_prefix = '回答：'

    def process_line(self, data, tokenizer, tokenize):
        if False:
            return 10
        if 'title' not in data:
            return ([], [])
        (prompts, texts) = ([], [])
        qtitle = data['title']
        qcontent = data.get('content', '')
        qcontent = self.trim_field(qcontent, max_length=100)
        prompt = self.qtitle_prefix + qtitle + self.qcontent_prefix + qcontent + self.answer_prefix
        prompt = self.process_sample(prompt, tokenizer, tokenize)
        if 'best_answer' in data:
            text = data['best_answer']['content']
            if len(text) > 10:
                text = self.process_sample(text, tokenizer, tokenize)
                prompts.append(prompt)
                texts.append(text)
        for answer in data.get('other_answers', []):
            text = answer['content']
            if len(text) > 100:
                text = self.process_sample(text, tokenizer, tokenize)
                prompts.append(prompt)
                texts.append(text)
        return (prompts, texts)

class baike(PromptReader):
    PATH = '/dataset/fd5061f6/data/tokenize_data/baike.lazy'
    reserve_punct = True
    assert_str = 'make sure to set PATH for baike data_utils/corpora.py'

    def process_line(self, data, tokenizer, tokenize):
        if False:
            for i in range(10):
                print('nop')
        (prompts, texts) = ([], [])
        text = data.get('title', '') + data.get('abstract', '') + data.get('content', '')
        if text:
            (p, t) = (self.process_sample('', tokenizer, tokenize), self.process_sample(text, tokenizer, tokenize))
            prompts.append(p)
            texts.append(t)
        return (prompts, texts)

class wikipedia(PromptReader):
    """
    dataset for wikipedia with arguments configured for convenience

    command line usage: `--train-data wikipedia`
    """
    PATH = '/root/data/bert_data/wiki.txt'
    assert_str = 'make sure to set PATH for wikipedia data_utils/corpora.py'

    def process_line(self, data, tokenizer, tokenize):
        if False:
            i = 10
            return i + 15
        text = data['text']
        (prompt, text) = (self.process_sample('', tokenizer, tokenize), self.process_sample(text, tokenizer, tokenize))
        return ([prompt], [text])

class TestDataset(PromptReader):
    PATH = '/root/data/test.json'
    assert_str = 'make sure to set PATH for wikipedia data_utils/corpora.py'

    def process_line(self, data, tokenizer, tokenize):
        if False:
            i = 10
            return i + 15
        (prompt, text) = (data['prompt'], data['text'])
        (prompt, text) = (self.process_sample(prompt, tokenizer, tokenize), self.process_sample(text, tokenizer, tokenize))
        return ([prompt], [text])

class OpenWebText(PromptReader):
    PATH = '/dataset/fd5061f6/english_data/openwebtext2'
    assert_str = 'make sure to set PATH for openwebtext data_utils/corpora.py'

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        import fasttext
        super().__init__(*args, **kwargs)
        self.model = fasttext.load_model('/dataset/fd5061f6/english_data/lid.176.bin')
        print_rank_0('Load language detection model')

    def process_line(self, data, tokenizer, tokenize):
        if False:
            i = 10
            return i + 15
        text = data['text']
        if len(text) > 100:
            lang = self.model.predict(text.replace('\n', ''))[0][0]
            if lang == '__label__en':
                (prompt, text) = (self.process_sample('', tokenizer, tokenize), self.process_sample(text, tokenizer, tokenize))
                return ([prompt], [text])
        return ([], [])

class CCNews(PromptReader):
    PATH = '/mnt/cc_news.json'
    assert_str = 'make sure to set PATH for cc-news data_utils/corpora.py'

    def process_line(self, data, tokenizer, tokenize):
        if False:
            return 10
        text = ''
        title = data.get('title', None)
        description = data.get('description', None)
        maintext = data.get('maintext', None)
        if title:
            text += title.strip() + ' '
        if description and (not maintext or not maintext.startswith(description)):
            text += description.strip() + ' '
        if maintext:
            text += maintext
        if len(text) > 100:
            (prompt, text) = (self.process_sample('', tokenizer, tokenize), self.process_sample(text, tokenizer, tokenize))
            return ([prompt], [text])
        else:
            return ([], [])

class BertData(PromptReader):
    is_json = False
    PATH = '/dataset/fd5061f6/english_data/wikibook'

    def process_line(self, data, tokenizer, tokenize):
        if False:
            print('Hello World!')
        if data:
            (prompt, text) = ('', data)
            (prompt, text) = (self.process_sample(prompt, tokenizer, tokenize), self.process_sample(text, tokenizer, tokenize))
            return ([prompt], [text])
        else:
            return ([], [])

class Pile(PromptReader):
    is_json = True
    PATH = '/mnt/train'
    filtered_sources = ['Github', 'StackExchange', 'DM Mathematics', 'Ubuntu IRC', 'EuroParl', 'YoutubeSubtitles', 'Enron Emails']
    downsample_sources = {'PubMed Central': 0.3, 'ArXiv': 0.3, 'FreeLaw': 0.3}

    def print_info(self, info):
        if False:
            print('Hello World!')
        total_dict = defaultdict(int)
        while True:
            try:
                source_dict = info.get(block=False)
                for (source, length) in source_dict.items():
                    total_dict[source] += length
            except Empty:
                break
        print_rank_0(total_dict)

    def tokenize_worker(self, input, output, info, tokenizer, tokenize):
        if False:
            print('Hello World!')
        source_dict = defaultdict(int)
        for row in iter(input.get, 'STOP'):
            row = row.rstrip()
            if row:
                if self.is_json:
                    row = json.loads(row)
                (prompts, texts, source) = self.process_line(row, tokenizer, tokenize)
                length = 0
                for (prompt, text) in zip(prompts, texts):
                    length += len(text)
                    output.put((prompt, text))
                if source:
                    source_dict[source] += length
        output.put('COMPLETE')
        info.put(source_dict)

    def process_line(self, data, tokenizer, tokenize):
        if False:
            i = 10
            return i + 15
        source = data['meta'].get('pile_set_name', None)
        text = data.get('text', None)
        if source and text:
            if source in self.filtered_sources:
                return ([], [], None)
            elif source in self.downsample_sources and random.random() > self.downsample_sources[source]:
                return ([], [], None)
            else:
                (prompt, text) = (self.process_sample('', tokenizer, tokenize), self.process_sample(text, tokenizer, tokenize))
                return ([prompt], [text], source)
        else:
            return ([], [], None)

class Stories(PromptReader):
    is_json = True
    PATH = '/dataset/fd5061f6/english_data/stories_31G.jsonl'

    def process_line(self, data, tokenizer, tokenize):
        if False:
            print('Hello World!')
        text = data.get('text', None)
        if text:
            (prompt, text) = (self.process_sample('', tokenizer, tokenize), self.process_sample(text, tokenizer, tokenize))
            return ([prompt], [text])
        else:
            return ([], [])

class BertBaseData(BertData):
    PATH = '/root/data/formatted_one_article_per_line'

class BertLargeData(BertData):
    PATH = '/dataset/c07bd62b/cognitive/zhengxiao/formatted_one_article_per_line_large'

class WuDaoCorpus(PromptReader):
    PATH = '/wudao'
    is_json = False
    reserve_punct = True
    split_row = False

    def process_line(self, item, tokenizer, tokenize):
        if False:
            print('Hello World!')
        (prompts, texts) = ([], [])
        text = ''
        title = item.get('title', None)
        content = item.get('content', None)
        if title:
            text += title.strip() + ' '
        if content:
            text += content
        if len(text) > 100:
            (prompt, text) = (self.process_sample('', tokenizer, tokenize), self.process_sample(text, tokenizer, tokenize))
            prompts.append(prompt)
            texts.append(text)
        return (prompts, texts)
NAMED_CORPORA = {'wikipedia': wikipedia, 'wikipedia-key': KeyReader, 'openwebtext': OpenWebText, 'zhihu': zhihu, 'zhidao': zhidao, 'baike': baike, 'test': TestDataset, 'wikibook': BertData, 'bert-base': BertBaseData, 'bert-large': BertLargeData, 'cc-news': CCNews, 'pile': Pile, 'stories': Stories, 'wudao': WuDaoCorpus}