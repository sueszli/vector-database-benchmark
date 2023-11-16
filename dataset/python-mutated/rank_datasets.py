import random
from collections import defaultdict
from typing import List
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
SEED = 2020

class SHPDataset(Dataset):
    """
    Dataset class to load stanfordnlp/SHP for Reward Modeling
    """
    name = 'SHP'

    def __init__(self, split: str | list[str] | None, max_answers: int=5):
        if False:
            print('Hello World!')
        super().__init__()
        self.questions = []
        self.answers = []
        if not isinstance(split, list):
            split = [split]
        dataset_splits = load_dataset('stanfordnlp/SHP', split=split)
        answers_by_id = defaultdict(dict)
        history_by_id = dict()
        for split in dataset_splits:
            for row in split:
                post_id = row['post_id']
                history_by_id[post_id] = row['history']
                answers_by_id[post_id][row['human_ref_A']] = row['score_A']
                answers_by_id[post_id][row['human_ref_B']] = row['score_B']
        for (post_id, history) in history_by_id.items():
            self.questions.append(history)
            answers = answers_by_id[post_id]
            answers_sorted = [x[0] for x in sorted(answers.items(), key=lambda x: -1 * x[1])]
            self.answers.append(answers_sorted[:max_answers])

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.questions)

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        return ([self.questions[index]], self.answers[index])

class HellaSwagDataset(Dataset):
    """
    Dataset class to use data from https://arxiv.org/pdf/1905.07830.pdf
    for Reward modeling

    Note: In order to disable dialog-formatting None is returned as context.
    """
    name = 'hellaswag'

    def __init__(self, split: str | list[str] | None, seed: int=SEED) -> None:
        if False:
            return 10
        super().__init__()
        np.random.seed(seed)
        self.dataset_list = []
        if not isinstance(split, List):
            split = [split]
        dataset = load_dataset('AlekseyKorshuk/hellaswag', split=split)
        for data in dataset:
            for item in data:
                context = item.get('ctx')
                endings = item.get('endings')
                selected = endings.pop(item.get('label'))
                ordered_ends = [selected, np.random.choice(endings)]
                self.dataset_list.append({'context': context, 'completions': ordered_ends})

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        return len(self.dataset_list)

    def __getitem__(self, idx) -> tuple[str | None, list[list]]:
        if False:
            print('Hello World!')
        (context, completions) = self.dataset_list[idx].values()
        return (None, [context + c for c in completions])

class HFDataset(Dataset):
    """
    Dataset class to use data from openai/summarize_from_feedback for Reward modeling.
    Summaries ranked by overall score.
    """
    name = 'open_ai_summarize_from_feedback'

    def __init__(self, split: str | list[str] | None=None, subset: str='axis') -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        if not isinstance(split, List):
            split = [split]
        dataset = load_dataset('openai/summarize_from_feedback', subset, split=split)
        self.subset = subset
        self.axis_post_ids = []
        self.axis_post_dict = defaultdict(dict)
        self.comparisons = []
        if subset == 'axis':
            self._handle_axis(dataset)
        else:
            self._handle_comparisons(dataset)

    def _handle_comparisons(self, dataset):
        if False:
            i = 10
            return i + 15
        for data in dataset:
            for item in data:
                choice = item['choice']
                full_post = item['info']['post']
                summaries = [item['summaries'][choice]['text'], item['summaries'][1 - choice]['text']]
                self.comparisons.append([[full_post], summaries])

    def _handle_axis(self, dataset):
        if False:
            for i in range(10):
                print('nop')
        for data in dataset:
            for item in data:
                if item['summary'].get('axes').get('overall') is not None:
                    post_id = item.get('info')['id']
                    if post_id not in self.axis_post_ids:
                        self.axis_post_ids.append(post_id)
                        item_content = item['info']['post'] or item['info']['article']
                        self.axis_post_dict[post_id].update({'post': item_content, 'summaries': [item['summary']]})
                    else:
                        self.axis_post_dict[post_id]['summaries'].append(item['summary'])

    def __len__(self):
        if False:
            print('Hello World!')
        if self.subset == 'axis':
            return len(self.axis_post_ids)
        return len(self.comparisons)

    def __getitem__(self, idx):
        if False:
            i = 10
            return i + 15
        (post, summaries) = self.post_dict[self.post_ids[idx]].values()
        summaries = sorted(summaries, key=lambda x: x['axes']['overall'], reverse=True)
        summaries = [summary['text'] for summary in summaries]
        return ([post], summaries)

class AugmentedOA(Dataset):

    def __init__(self, json_filename: str, split: str='train') -> None:
        if False:
            return 10
        super().__init__()
        import json
        assert split in ('train', 'val')
        pairs = []
        with open(json_filename, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data['split'] == split:
                    augmented = data['augmented']
                    if split == 'val':
                        augmented = []
                    pairs.append((data['prefixes'], data['responses'], augmented))
        self.pairs = pairs

    def __len__(self):
        if False:
            return 10
        return len(self.pairs)

    def __getitem__(self, idx):
        if False:
            i = 10
            return i + 15
        (prefixes, user_answer_ranks, bad_samples) = self.pairs[idx]
        rank = user_answer_ranks
        if len(bad_samples) > 0:
            additional = random.choice(bad_samples)
            rank = user_answer_ranks + [additional]
        return (prefixes, rank)

class AnthropicRLHF(Dataset):
    name = 'anthropic_rlhf'

    @staticmethod
    def _split_dialogue(text: str) -> list[tuple[str, str]]:
        if False:
            print('Hello World!')
        lines = text.split('\n\n')
        dialogue: list[tuple[str, str]] = []
        role = None
        messages = []
        for line in lines:
            if line.startswith('Human:'):
                speaker = 'Human'
                message = line[7:]
            elif line.startswith('Assistant:'):
                speaker = 'Assistant'
                message = line[11:]
            else:
                continue
            if role != speaker:
                if role is not None:
                    dialogue.append((role, '\n'.join(messages)))
                    messages = []
                role = speaker
            messages.append(message.strip())
        if role is not None and len(messages) > 0:
            dialogue.append((role, '\n'.join(messages)))
        return dialogue

    def __init__(self, split: str='train') -> None:
        if False:
            print('Hello World!')
        super().__init__()
        assert split in ('train', 'test')
        self.split = split
        self.data = []
        dataset = load_dataset('Anthropic/hh-rlhf')[split]
        for entry in dataset:
            chosen = entry['chosen']
            if 'Assistant' not in chosen:
                continue
            rejected = entry['rejected']
            chosen = self._split_dialogue(chosen)
            rejected = self._split_dialogue(rejected)
            assert rejected[0][0] == 'Human' and chosen[0][0] == 'Human'
            if len(rejected) == len(chosen):
                prefix = [line for (speaker, line) in chosen[:-1]]
                good_reply = chosen[-1][1]
                bad_reply = rejected[-1][1]
                self.data.append((prefix, [good_reply, bad_reply]))

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[str, list[str]]:
        if False:
            i = 10
            return i + 15
        return self.data[index]