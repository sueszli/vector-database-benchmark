from pathlib import Path
from typing import Iterable, Literal, Optional
from model_training.custom_datasets.formatting import DatasetEntrySft, Role, Utterance
from oasst_data import ExportMessageNode, read_dataset_message_trees, read_message_trees, visit_threads_depth_first
from oasst_data.schemas import ExportMessageTree
from torch import Generator
from torch.utils.data import Dataset, random_split

class ListDataset(Dataset):

    def __init__(self, data: list):
        if False:
            print('Hello World!')
        super().__init__()
        self.data = data

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self.data)

    def __getitem__(self, index):
        if False:
            return 10
        return self.data[index]

def load_oasst_export(input_file_path: Optional[str | Path]=None, hf_dataset_name: Optional[str]='OpenAssistant/oasst1', val_split: float=0.2, lang: str='en', top_k: Optional[int]=None, manual_seed: int=287631038922, data_path: str | Path=None, mode: Literal['sft', 'rm', 'rl']='sft') -> tuple[ListDataset, ListDataset]:
    if False:
        i = 10
        return i + 15
    if mode not in ('sft', 'rm', 'rl'):
        raise ValueError(f'Unknown dataset mode: {mode}')
    lang_codes: list[str] = lang.split(',')
    generator = Generator()
    generator.manual_seed(manual_seed)
    tree_iter: Iterable[ExportMessageTree] = None
    if input_file_path:
        if not isinstance(input_file_path, Path):
            input_file_path = Path(input_file_path)
        if not input_file_path.is_absolute() and data_path:
            if not isinstance(data_path, Path):
                data_path = Path(data_path)
            input_file_path = data_path / input_file_path
        tree_iter = read_message_trees(input_file_path)
    elif hf_dataset_name:
        tree_iter = read_dataset_message_trees(hf_dataset_name, split='train+validation')
    else:
        raise RuntimeError('Either `input_file_path` or `hf_dataset_name` must be specified.')
    threads_per_tree = []
    for tree in tree_iter:
        if tree.tree_state != 'ready_for_export' or not tree.prompt.review_result or tree.prompt.lang not in lang_codes:
            continue
        if mode in ('sft', 'rm'):
            if tree.tree_state != 'ready_for_export':
                continue
        elif mode == 'rl':
            if tree.tree_state not in ('ready_for_export', 'prompt_lottery_waiting'):
                continue
        threads: list[list[ExportMessageNode]] = []

        def thread_filter(thread: list[ExportMessageNode]) -> bool:
            if False:
                i = 10
                return i + 15
            if any((m.deleted or m.synthetic for m in thread)):
                return False
            if top_k is not None:
                for (i, m) in enumerate(thread):
                    if m.role == 'assistant':
                        if m.rank is None:
                            if i > 0 and len(thread[i - 1].replies) > 1:
                                return False
                        elif m.rank >= top_k:
                            return False
            return True

        def leaf_filter(thread: list[ExportMessageNode]) -> bool:
            if False:
                for i in range(10):
                    print('nop')
            if mode == 'sft':
                return len(thread) > 1 and (not thread[-1].replies) and (thread[-1].role == 'assistant' or thread[-2].replies[0] == thread[-1]) and thread_filter(thread)
            elif mode == 'rm':
                if thread[-1].replies is None:
                    return False
                return thread[-1].role == 'prompter' and len([r for r in thread[-1].replies if r.rank is not None]) > 1 and thread_filter(thread)
            elif mode == 'rl':
                return thread[-1].role == 'prompter' and (not any((m.deleted or m.synthetic for m in thread)))
            raise RuntimeError()
        visit_threads_depth_first(tree.prompt, visitor=threads.append, predicate=leaf_filter)
        if mode == 'sft':
            for t in threads:
                if t[-1].role == 'prompter':
                    t.pop()
        threads_per_tree.append(threads)

    def process_thread(thread: list[ExportMessageNode]):
        if False:
            while True:
                i = 10
        if mode == 'sft':
            assert all((m.role == 'prompter' for m in thread[0::2])) and all((m.role == 'assistant' for m in thread[1::2]))
            conversation: list[Utterance] = [Utterance(text=m.text, role=Role.prompter if m.role == 'prompter' else Role.assistant, lang=m.lang, quality=m.get_label_value('quality'), humor=m.get_label_value('humor'), creativity=m.get_label_value('creativity')) for m in thread]
            return DatasetEntrySft(conversation=conversation)
        elif mode == 'rm':
            prefix = [m.text for m in thread]
            replies = [r for r in thread[-1].replies if r.role == 'assistant' and r.rank is not None]
            replies = sorted(replies, key=lambda r: r.rank)
            replies = [r.text for r in replies]
            return (prefix, replies)
        elif mode == 'rl':
            return ([m.text for m in thread],)
        raise RuntimeError()
    trees = ListDataset(threads_per_tree)
    splits = random_split(trees, lengths=[1.0 - val_split, val_split], generator=generator)

    def flatten(ds: ListDataset) -> ListDataset:
        if False:
            i = 10
            return i + 15
        return ListDataset([process_thread(thread) for tree_threads in ds for thread in tree_threads])
    train = flatten(splits[0])
    val = flatten(splits[1])
    if input_file_path:
        print(f'OASST JSONL file {str(input_file_path)}: len(train)={len(train)!r}, len(val)={len(val)!r}')
    else:
        print(f'OASST HF dataset {hf_dataset_name}: len(train)={len(train)!r}, len(val)={len(val)!r}')
    return (train, val)