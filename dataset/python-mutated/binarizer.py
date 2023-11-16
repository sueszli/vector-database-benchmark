import logging
import os
import typing as tp
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Pool
import torch
from fairseq.data import Dictionary, indexed_dataset
from fairseq.file_chunker_utils import Chunker, find_offsets
from fairseq.file_io import PathManager
from fairseq.tokenizer import tokenize_line
logger = logging.getLogger('binarizer')

@dataclass
class BinarizeSummary:
    """
    Keep track of what's going on in the binarizer
    """
    num_seq: int = 0
    replaced: tp.Optional[Counter] = None
    num_tok: int = 0

    @property
    def num_replaced(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        if self.replaced is None:
            return 0
        return sum(self.replaced.values())

    @property
    def replaced_percent(self) -> float:
        if False:
            while True:
                i = 10
        return 100 * self.num_replaced / self.num_tok

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        base = f'{self.num_seq} sents, {self.num_tok} tokens'
        if self.replaced is None:
            return base
        return f'{base}, {self.replaced_percent:.3}% replaced'

    def merge(self, other: 'BinarizeSummary'):
        if False:
            while True:
                i = 10
        replaced = None
        if self.replaced is not None:
            replaced = self.replaced
        if other.replaced is not None:
            if replaced is None:
                replaced = other.replaced
            else:
                replaced += other.replaced
        self.replaced = replaced
        self.num_seq += other.num_seq
        self.num_tok += other.num_tok

class Binarizer(ABC):
    """
    a binarizer describes how to take a string and build a tensor out of it
    """

    @abstractmethod
    def binarize_line(self, line: str, summary: BinarizeSummary) -> torch.IntTensor:
        if False:
            print('Hello World!')
        ...

def _worker_prefix(output_prefix: str, worker_id: int):
    if False:
        print('Hello World!')
    return f'{output_prefix}.pt{worker_id}'

class FileBinarizer:
    """
    An file binarizer can take a file, tokenize it, and binarize each line to a tensor
    """

    @classmethod
    def multiprocess_dataset(cls, input_file: str, dataset_impl: str, binarizer: Binarizer, output_prefix: str, vocab_size=None, num_workers=1) -> BinarizeSummary:
        if False:
            i = 10
            return i + 15
        final_summary = BinarizeSummary()
        offsets = find_offsets(input_file, num_workers)
        (first_chunk, *more_chunks) = zip(offsets, offsets[1:])
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            worker_results = [pool.apply_async(cls._binarize_chunk_and_finalize, args=(binarizer, input_file, start_offset, end_offset, _worker_prefix(output_prefix, worker_id), dataset_impl), kwds={'vocab_size': vocab_size} if vocab_size is not None else {}) for (worker_id, (start_offset, end_offset)) in enumerate(more_chunks, start=1)]
            pool.close()
            pool.join()
            for r in worker_results:
                summ = r.get()
                final_summary.merge(summ)
        (final_ds, summ) = cls._binarize_file_chunk(binarizer, input_file, offset_start=first_chunk[0], offset_end=first_chunk[1], output_prefix=output_prefix, dataset_impl=dataset_impl, vocab_size=vocab_size if vocab_size is not None else None)
        final_summary.merge(summ)
        if num_workers > 1:
            for worker_id in range(1, num_workers):
                worker_output_prefix = _worker_prefix(output_prefix, worker_id)
                final_ds.merge_file_(worker_output_prefix)
                try:
                    os.remove(indexed_dataset.data_file_path(worker_output_prefix))
                    os.remove(indexed_dataset.index_file_path(worker_output_prefix))
                except Exception as e:
                    logger.error(f"couldn't remove {worker_output_prefix}.*", exc_info=e)
        idx_file = indexed_dataset.index_file_path(output_prefix)
        final_ds.finalize(idx_file)
        return final_summary

    @staticmethod
    def _binarize_file_chunk(binarizer: Binarizer, filename: str, offset_start: int, offset_end: int, output_prefix: str, dataset_impl: str, vocab_size=None) -> tp.Tuple[tp.Any, BinarizeSummary]:
        if False:
            for i in range(10):
                print('nop')
        '\n        creates a dataset builder and append binarized items to it. This function does not\n        finalize the builder, this is useful if you want to do other things with your bin file\n        like appending/merging other files\n        '
        bin_file = indexed_dataset.data_file_path(output_prefix)
        ds = indexed_dataset.make_builder(bin_file, impl=dataset_impl, vocab_size=vocab_size)
        summary = BinarizeSummary()
        with Chunker(PathManager.get_local_path(filename), offset_start, offset_end) as line_iterator:
            for line in line_iterator:
                ds.add_item(binarizer.binarize_line(line, summary))
        return (ds, summary)

    @classmethod
    def _binarize_chunk_and_finalize(cls, binarizer: Binarizer, filename: str, offset_start: int, offset_end: int, output_prefix: str, dataset_impl: str, vocab_size=None):
        if False:
            print('Hello World!')
        '\n        same as above, but also finalizes the builder\n        '
        (ds, summ) = cls._binarize_file_chunk(binarizer, filename, offset_start, offset_end, output_prefix, dataset_impl, vocab_size=vocab_size)
        idx_file = indexed_dataset.index_file_path(output_prefix)
        ds.finalize(idx_file)
        return summ

class VocabularyDatasetBinarizer(Binarizer):
    """
    Takes a Dictionary/Vocabulary, assign ids to each
    token using the dictionary encode_line function.
    """

    def __init__(self, dict: Dictionary, tokenize: tp.Callable[[str], tp.List[str]]=tokenize_line, append_eos: bool=True, reverse_order: bool=False, already_numberized: bool=False) -> None:
        if False:
            while True:
                i = 10
        self.dict = dict
        self.tokenize = tokenize
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.already_numberized = already_numberized
        super().__init__()

    def binarize_line(self, line: str, summary: BinarizeSummary):
        if False:
            i = 10
            return i + 15
        if summary.replaced is None:
            summary.replaced = Counter()

        def replaced_consumer(word, idx):
            if False:
                print('Hello World!')
            if idx == self.dict.unk_index and word != self.dict.unk_word:
                summary.replaced.update([word])
        if self.already_numberized:
            id_strings = line.strip().split()
            id_list = [int(id_string) for id_string in id_strings]
            if self.reverse_order:
                id_list.reverse()
            if self.append_eos:
                id_list.append(self.dict.eos())
            ids = torch.IntTensor(id_list)
        else:
            ids = self.dict.encode_line(line=line, line_tokenizer=self.tokenize, add_if_not_exist=False, consumer=replaced_consumer, append_eos=self.append_eos, reverse_order=self.reverse_order)
        summary.num_seq += 1
        summary.num_tok += len(ids)
        return ids

class AlignmentDatasetBinarizer(Binarizer):
    """
    binarize by parsing a set of alignments and packing
    them in a tensor (see utils.parse_alignment)
    """

    def __init__(self, alignment_parser: tp.Callable[[str], torch.IntTensor]) -> None:
        if False:
            return 10
        super().__init__()
        self.alignment_parser = alignment_parser

    def binarize_line(self, line: str, summary: BinarizeSummary):
        if False:
            while True:
                i = 10
        ids = self.alignment_parser(line)
        summary.num_seq += 1
        summary.num_tok += len(ids)
        return ids

class LegacyBinarizer:

    @classmethod
    def binarize(cls, filename: str, dico: Dictionary, consumer: tp.Callable[[torch.IntTensor], None], tokenize: tp.Callable[[str], tp.List[str]]=tokenize_line, append_eos: bool=True, reverse_order: bool=False, offset: int=0, end: int=-1, already_numberized: bool=False) -> tp.Dict[str, int]:
        if False:
            return 10
        binarizer = VocabularyDatasetBinarizer(dict=dico, tokenize=tokenize, append_eos=append_eos, reverse_order=reverse_order, already_numberized=already_numberized)
        return cls._consume_file(filename, binarizer, consumer, offset_start=offset, offset_end=end)

    @classmethod
    def binarize_alignments(cls, filename: str, alignment_parser: tp.Callable[[str], torch.IntTensor], consumer: tp.Callable[[torch.IntTensor], None], offset: int=0, end: int=-1) -> tp.Dict[str, int]:
        if False:
            while True:
                i = 10
        binarizer = AlignmentDatasetBinarizer(alignment_parser)
        return cls._consume_file(filename, binarizer, consumer, offset_start=offset, offset_end=end)

    @staticmethod
    def _consume_file(filename: str, binarizer: Binarizer, consumer: tp.Callable[[torch.IntTensor], None], offset_start: int, offset_end: int) -> tp.Dict[str, int]:
        if False:
            print('Hello World!')
        summary = BinarizeSummary()
        with Chunker(PathManager.get_local_path(filename), offset_start, offset_end) as line_iterator:
            for line in line_iterator:
                consumer(binarizer.binarize_line(line, summary))
        return {'nseq': summary.num_seq, 'nunk': summary.num_replaced, 'ntok': summary.num_tok, 'replaced': summary.replaced}