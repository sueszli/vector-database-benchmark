"""
    SFT dataset to reject toxic questions

"""
import random
from datasets import load_dataset
from torch.utils.data import Dataset

class ProsocialDialogueExplaination(Dataset):
    name = 'explain_prosocial'
    TEMPLATE = [("'{0}' Why is this sentence {1}", 'This sentence is {1} because {0}'), ('Explain to me why this sentence is {1}: {0}', 'This sentence is {1} because {0}'), ("'{0}' Why is this sentence {1}", 'This sentence may need {1} because {0}'), ("'{0}' Explain to me why we need to {1} of this sentence", 'We need to be {1} of this sentence because {0}')]

    def __init__(self, split='train', cache_dir='.cache') -> None:
        if False:
            return 10
        super().__init__()
        dataset = load_dataset('Englishman2022/prosocial-dialog-filtered', data_files='train.json', cache_dir=cache_dir, revision='e121e4fd886fadc030d633274c053b71839f9c20')[split]
        self.pairs = []
        for row in dataset:
            for (safety_annotation, safe_answer) in zip(row['safety_annotations'], row['safety_annotation_reasons']):
                (prompt_template, answer_template) = random.choice(self.TEMPLATE)
                self.pairs.append((prompt_template.format(row['context'], safety_annotation), answer_template.format(safe_answer, safety_annotation)))

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.pairs)

    def __getitem__(self, idx):
        if False:
            print('Hello World!')
        return self.pairs[idx]

class ProsocialDialogue(Dataset):
    name = 'prosocial_dialogue'
    '\n        ProsocialDialog, we set up a human-AI collaborative data creation framework,\n        where GPT-3 generates the potentially unsafe utterances, and crowdworkers\n        provide prosocial responses to them. This approach allows us to circumvent\n        two substantial challenges:\n        (1) there are no available large-scale corpora of multiturn prosocial conversations\n            between humans\n        (2) asking humans to write unethical, toxic, or problematic utterances could result\n            in psychological harms (Roberts, 2017; Steiger et al., 2021).\n    '

    def __init__(self, split='train', cache_dir='.cache') -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        dataset = load_dataset('Englishman2022/prosocial-dialog-filtered', data_files='train.json', cache_dir=cache_dir, revision='e121e4fd886fadc030d633274c053b71839f9c20')[split]
        self.pairs = []
        for row in dataset:
            prompt = row['context']
            for answer in row['rots']:
                self.pairs.append((prompt, answer))

    def __len__(self):
        if False:
            return 10
        return len(self.pairs)

    def __getitem__(self, idx):
        if False:
            return 10
        return self.pairs[idx]