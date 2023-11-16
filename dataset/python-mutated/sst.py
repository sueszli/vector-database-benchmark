import os
from typing import List, Optional, Set
import pandas as pd
from ludwig.datasets.dataset_config import DatasetConfig
from ludwig.datasets.loaders.dataset_loader import DatasetLoader

class SSTLoader(DatasetLoader):
    """The SST dataset.

    This dataset is constructed using the Stanford Sentiment Treebank Dataset.
    This dataset contains binary labels (positive or negative) for each sample.

    The original dataset specified 5 labels:
    very negative, negative, neutral, positive, very positive with
    the following cutoffs:
    [0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]
    """

    def __init__(self, config: DatasetConfig, cache_dir: Optional[str]=None, include_subtrees=False, discard_neutral=False, convert_parentheses=True, remove_duplicates=False):
        if False:
            i = 10
            return i + 15
        super().__init__(config, cache_dir=cache_dir)
        self.include_subtrees = include_subtrees
        self.discard_neutral = discard_neutral
        self.convert_parentheses = convert_parentheses
        self.remove_duplicates = remove_duplicates

    @staticmethod
    def get_sentiment_label(id2sent, phrase_id):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def transform_files(self, file_paths: List[str]) -> List[str]:
        if False:
            print('Hello World!')
        'Load dataset files into a dataframe.'
        sentences_df = pd.read_csv(os.path.join(self.raw_dataset_dir, 'stanfordSentimentTreebank/datasetSentences.txt'), sep='\t')
        sentences_df['sentence'] = sentences_df['sentence'].apply(format_text)
        datasplit_df = pd.read_csv(os.path.join(self.raw_dataset_dir, 'stanfordSentimentTreebank/datasetSplit.txt'), sep=',')
        phrase2id = {}
        with open(os.path.join(self.raw_dataset_dir, 'stanfordSentimentTreebank/dictionary.txt')) as f:
            Lines = f.readlines()
            for line in Lines:
                if line:
                    split_line = line.split('|')
                    phrase = split_line[0]
                    phrase2id[phrase] = int(split_line[1])
        id2sent = {}
        with open(os.path.join(self.raw_dataset_dir, 'stanfordSentimentTreebank/sentiment_labels.txt')) as f:
            Lines = f.readlines()
            for line in Lines:
                if line:
                    split_line = line.split('|')
                    try:
                        id2sent[int(split_line[0])] = float(split_line[1])
                    except ValueError:
                        pass
        trees_pointers = None
        trees_phrases = None
        if self.include_subtrees:
            trees_pointers = []
            with open(os.path.join(self.raw_dataset_dir, 'stanfordSentimentTreebank/STree.txt')) as f:
                Lines = f.readlines()
                for line in Lines:
                    if line:
                        trees_pointers.append([int(s.strip()) for s in line.split('|')])
            trees_phrases = []
            with open(os.path.join(self.raw_dataset_dir, 'stanfordSentimentTreebank/SOStr.txt')) as f:
                Lines = f.readlines()
                for line in Lines:
                    if line:
                        trees_phrases.append([s.strip() for s in line.split('|')])
        splits = {'train': 1, 'test': 2, 'dev': 3}
        generated_csv_filenames = []
        for (split_name, split_id) in splits.items():
            sentence_idcs = get_sentence_idcs_in_split(datasplit_df, split_id)
            pairs = []
            if split_name == 'train' and self.include_subtrees:
                phrases = []
                for sentence_idx in sentence_idcs:
                    sentence_idx -= 1
                    subtrees = sentence_subtrees(sentence_idx, trees_pointers, trees_phrases)
                    sentence_idx += 1
                    sentence_phrase = list(sentences_df[sentences_df['sentence_index'] == sentence_idx]['sentence'])[0]
                    sentence_phrase = convert_parentheses(sentence_phrase)
                    label = self.get_sentiment_label(id2sent, phrase2id[sentence_phrase])
                    if not self.discard_neutral or label != -1:
                        for phrase in subtrees:
                            label = self.get_sentiment_label(id2sent, phrase2id[phrase])
                            if not self.discard_neutral or label != -1:
                                if not self.convert_parentheses:
                                    phrase = convert_parentheses_back(phrase)
                                    phrase = phrase.replace('\xa0', ' ')
                                pairs.append([phrase, label])
            else:
                phrases = get_sentences_with_idcs(sentences_df, sentence_idcs)
                for phrase in phrases:
                    phrase = convert_parentheses(phrase)
                    label = self.get_sentiment_label(id2sent, phrase2id[phrase])
                    if not self.discard_neutral or label != -1:
                        if not self.convert_parentheses:
                            phrase = convert_parentheses_back(phrase)
                            phrase = phrase.replace('\xa0', ' ')
                        pairs.append([phrase, label])
            final_csv = pd.DataFrame(pairs)
            final_csv.columns = ['sentence', 'label']
            if self.remove_duplicates:
                final_csv = final_csv.drop_duplicates(subset=['sentence'])
            csv_filename = os.path.join(self.raw_dataset_dir, f'{split_name}.csv')
            generated_csv_filenames.append(csv_filename)
            final_csv.to_csv(csv_filename, index=False)
        return super().transform_files(generated_csv_filenames)

class SST2Loader(SSTLoader):
    """The SST2 dataset.

    This dataset is constructed using the Stanford Sentiment Treebank Dataset.
    This dataset contains binary labels (positive or negative) for each sample.

    The original dataset specified 5 labels:
    very negative, negative, neutral, positive, very positive with
    the following cutoffs:
    [0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]

    In the construction of this dataset, we remove all neutral phrases
    and assign a negative label if the original rating falls
    into the following range: [0, 0.4] and a positive label
    if the original rating is between (0.6, 1.0].
    """

    def __init__(self, config: DatasetConfig, cache_dir: Optional[str]=None, include_subtrees=False, convert_parentheses=True, remove_duplicates=False):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config, cache_dir=cache_dir, include_subtrees=include_subtrees, discard_neutral=True, convert_parentheses=convert_parentheses, remove_duplicates=remove_duplicates)

    def get_sentiment_label(self, id2sent, phrase_id):
        if False:
            print('Hello World!')
        sentiment = id2sent[phrase_id]
        if sentiment <= 0.4:
            return 0
        elif sentiment > 0.6:
            return 1
        return -1

class SST3Loader(SSTLoader):
    """The SST3 dataset.

    This dataset is constructed using the Stanford Sentiment Treebank Dataset.
    This dataset contains five labels (very negative, negative, neutral,
    positive, very positive) for each sample.

    In the original dataset, the  5 labels: very negative, negative, neutral, positive,
    and very positive have the following cutoffs:
    [0, 0.4], (0.4, 0.6], (0.6, 1.0]

    This class pulls in an array of mixins for different types of functionality
    which belongs in the workflow for ingesting and transforming
    training data into a destination dataframe that can be use by Ludwig.
    """

    def __init__(self, config: DatasetConfig, cache_dir: Optional[str]=None, include_subtrees=False, convert_parentheses=True, remove_duplicates=False):
        if False:
            print('Hello World!')
        super().__init__(config, cache_dir=cache_dir, include_subtrees=include_subtrees, convert_parentheses=convert_parentheses, remove_duplicates=remove_duplicates)

    def get_sentiment_label(self, id2sent, phrase_id):
        if False:
            for i in range(10):
                print('nop')
        sentiment = id2sent[phrase_id]
        if sentiment <= 0.4:
            return 'negative'
        elif sentiment <= 0.6:
            return 'neutral'
        elif sentiment <= 1.0:
            return 'positive'
        return 'neutral'

class SST5Loader(SSTLoader):
    """The SST5 dataset.

    This dataset is constructed using the Stanford Sentiment Treebank Dataset.
    This dataset contains five labels (very negative, negative, neutral,
    positive, very positive) for each sample.

    In the original dataset, the  5 labels: very negative, negative, neutral, positive,
    and very positive have the following cutoffs:
    [0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]

    This class pulls in an array of mixins for different types of functionality
    which belongs in the workflow for ingesting and transforming
    training data into a destination dataframe that can be use by Ludwig.
    """

    def __init__(self, config: DatasetConfig, cache_dir: Optional[str]=None, include_subtrees=False, convert_parentheses=True, remove_duplicates=False):
        if False:
            while True:
                i = 10
        super().__init__(config, cache_dir=cache_dir, include_subtrees=include_subtrees, convert_parentheses=convert_parentheses, remove_duplicates=remove_duplicates)

    def get_sentiment_label(self, id2sent, phrase_id):
        if False:
            print('Hello World!')
        sentiment = id2sent[phrase_id]
        if sentiment <= 0.2:
            return 'very_negative'
        elif sentiment <= 0.4:
            return 'negative'
        elif sentiment <= 0.6:
            return 'neutral'
        elif sentiment <= 0.8:
            return 'positive'
        elif sentiment <= 1.0:
            return 'very_positive'
        return 'neutral'

def format_text(text: str):
    if False:
        print('Hello World!')
    'Formats text by decoding into utf-8.'
    return ' '.join([w.encode('latin1').decode('utf-8') for w in text.strip().split(' ')])

def convert_parentheses(text: str):
    if False:
        return 10
    'Replaces -LRB- and -RRB- tokens present in SST with ( and )'
    return text.replace('-LRB-', '(').replace('-RRB-', ')')

def convert_parentheses_back(text: str):
    if False:
        for i in range(10):
            print('nop')
    'Replaces ( and ) tokens with -LRB- and -RRB-'
    return text.replace('(', '-LRB-').replace(')', '-RRB-')

def get_sentence_idcs_in_split(datasplit: pd.DataFrame, split_id: int):
    if False:
        print('Hello World!')
    'Given a dataset split is (1 for train, 2 for test, 3 for dev), returns the set of corresponding sentence\n    indices in sentences_df.'
    return set(datasplit[datasplit['splitset_label'] == split_id]['sentence_index'])

def get_sentences_with_idcs(sentences: pd.DataFrame, sentences_idcs: Set[int]):
    if False:
        while True:
            i = 10
    'Given a set of sentence indices, returns the corresponding sentences texts in sentences.'
    criterion = sentences['sentence_index'].map(lambda x: x in sentences_idcs)
    return sentences[criterion]['sentence'].tolist()

def sentence_subtrees(sentence_idx, trees_pointers, trees_phrases):
    if False:
        for i in range(10):
            print('nop')
    tree_pointers = trees_pointers[sentence_idx]
    tree_phrases = trees_phrases[sentence_idx]
    tree = SSTTree(tree_pointers, tree_phrases)
    return tree.subtrees()

def visit_postorder(node, visit_list):
    if False:
        return 10
    if node:
        visit_postorder(node.left, visit_list)
        visit_postorder(node.right, visit_list)
        visit_list.append(node.val)

class SSTTree:

    class Node:

        def __init__(self, key, val=None):
            if False:
                return 10
            self.left = None
            self.right = None
            self.key = key
            self.val = val

    def create_node(self, parent, i):
        if False:
            print('Hello World!')
        if self.nodes[i] is not None:
            return
        self.nodes[i] = self.Node(i)
        if parent[i] == -1:
            self.root = self.nodes[i]
            return
        if self.nodes[parent[i]] is None:
            self.create_node(parent, parent[i])
        parent = self.nodes[parent[i]]
        if parent.left is None:
            parent.left = self.nodes[i]
        else:
            parent.right = self.nodes[i]

    def create_tree(self, parents, tree_phrases):
        if False:
            i = 10
            return i + 15
        n = len(parents)
        self.nodes = [None for i in range(n)]
        self.root = [None]
        for i in range(n):
            self.create_node(parents, i)
        for (i, phrase) in enumerate(tree_phrases):
            self.nodes[i].val = phrase
        for node in self.nodes:
            if node.val is None:
                node.val = ' '.join((node.left.val, node.right.val))

    def __init__(self, tree_pointers, tree_phrases):
        if False:
            print('Hello World!')
        self.create_tree([int(elem) - 1 for elem in tree_pointers], tree_phrases)

    def subtrees(self):
        if False:
            for i in range(10):
                print('nop')
        visit_list = []
        visit_postorder(self.root, visit_list)
        return visit_list