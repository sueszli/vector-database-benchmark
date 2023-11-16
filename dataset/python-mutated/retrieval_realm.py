"""REALM Retriever model implementation."""
import os
from typing import Optional, Union
import numpy as np
from huggingface_hub import hf_hub_download
from ... import AutoTokenizer
from ...utils import logging
_REALM_BLOCK_RECORDS_FILENAME = 'block_records.npy'
logger = logging.get_logger(__name__)

def convert_tfrecord_to_np(block_records_path: str, num_block_records: int) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    import tensorflow.compat.v1 as tf
    blocks_dataset = tf.data.TFRecordDataset(block_records_path, buffer_size=512 * 1024 * 1024)
    blocks_dataset = blocks_dataset.batch(num_block_records, drop_remainder=True)
    np_record = next(blocks_dataset.take(1).as_numpy_iterator())
    return np_record

class ScaNNSearcher:
    """Note that ScaNNSearcher cannot currently be used within the model. In future versions, it might however be included."""

    def __init__(self, db, num_neighbors, dimensions_per_block=2, num_leaves=1000, num_leaves_to_search=100, training_sample_size=100000):
        if False:
            print('Hello World!')
        'Build scann searcher.'
        from scann.scann_ops.py.scann_ops_pybind import builder as Builder
        builder = Builder(db=db, num_neighbors=num_neighbors, distance_measure='dot_product')
        builder = builder.tree(num_leaves=num_leaves, num_leaves_to_search=num_leaves_to_search, training_sample_size=training_sample_size)
        builder = builder.score_ah(dimensions_per_block=dimensions_per_block)
        self.searcher = builder.build()

    def search_batched(self, question_projection):
        if False:
            return 10
        (retrieved_block_ids, _) = self.searcher.search_batched(question_projection.detach().cpu())
        return retrieved_block_ids.astype('int64')

class RealmRetriever:
    """The retriever of REALM outputting the retrieved evidence block and whether the block has answers as well as answer
    positions."

        Parameters:
            block_records (`np.ndarray`):
                A numpy array which cantains evidence texts.
            tokenizer ([`RealmTokenizer`]):
                The tokenizer to encode retrieved texts.
    """

    def __init__(self, block_records, tokenizer):
        if False:
            while True:
                i = 10
        super().__init__()
        self.block_records = block_records
        self.tokenizer = tokenizer

    def __call__(self, retrieved_block_ids, question_input_ids, answer_ids, max_length=None, return_tensors='pt'):
        if False:
            for i in range(10):
                print('nop')
        retrieved_blocks = np.take(self.block_records, indices=retrieved_block_ids, axis=0)
        question = self.tokenizer.decode(question_input_ids[0], skip_special_tokens=True)
        text = []
        text_pair = []
        for retrieved_block in retrieved_blocks:
            text.append(question)
            text_pair.append(retrieved_block.decode())
        concat_inputs = self.tokenizer(text, text_pair, padding=True, truncation=True, return_special_tokens_mask=True, max_length=max_length)
        concat_inputs_tensors = concat_inputs.convert_to_tensors(return_tensors)
        if answer_ids is not None:
            return self.block_has_answer(concat_inputs, answer_ids) + (concat_inputs_tensors,)
        else:
            return (None, None, None, concat_inputs_tensors)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *init_inputs, **kwargs):
        if False:
            print('Hello World!')
        if os.path.isdir(pretrained_model_name_or_path):
            block_records_path = os.path.join(pretrained_model_name_or_path, _REALM_BLOCK_RECORDS_FILENAME)
        else:
            block_records_path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename=_REALM_BLOCK_RECORDS_FILENAME, **kwargs)
        block_records = np.load(block_records_path, allow_pickle=True)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        return cls(block_records, tokenizer)

    def save_pretrained(self, save_directory):
        if False:
            for i in range(10):
                print('nop')
        np.save(os.path.join(save_directory, _REALM_BLOCK_RECORDS_FILENAME), self.block_records)
        self.tokenizer.save_pretrained(save_directory)

    def block_has_answer(self, concat_inputs, answer_ids):
        if False:
            i = 10
            return i + 15
        'check if retrieved_blocks has answers.'
        has_answers = []
        start_pos = []
        end_pos = []
        max_answers = 0
        for input_id in concat_inputs.input_ids:
            input_id_list = input_id.tolist()
            first_sep_idx = input_id_list.index(self.tokenizer.sep_token_id)
            second_sep_idx = first_sep_idx + 1 + input_id_list[first_sep_idx + 1:].index(self.tokenizer.sep_token_id)
            start_pos.append([])
            end_pos.append([])
            for answer in answer_ids:
                for idx in range(first_sep_idx + 1, second_sep_idx):
                    if answer[0] == input_id_list[idx]:
                        if input_id_list[idx:idx + len(answer)] == answer:
                            start_pos[-1].append(idx)
                            end_pos[-1].append(idx + len(answer) - 1)
            if len(start_pos[-1]) == 0:
                has_answers.append(False)
            else:
                has_answers.append(True)
                if len(start_pos[-1]) > max_answers:
                    max_answers = len(start_pos[-1])
        for (start_pos_, end_pos_) in zip(start_pos, end_pos):
            if len(start_pos_) < max_answers:
                padded = [-1] * (max_answers - len(start_pos_))
                start_pos_ += padded
                end_pos_ += padded
        return (has_answers, start_pos, end_pos)