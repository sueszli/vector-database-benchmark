from model_training.custom_datasets.rank_datasets import AnthropicRLHF, HellaSwagDataset, SHPDataset
from torch.utils.data import Dataset

def load_anthropic_rlhf() -> tuple[Dataset, Dataset]:
    if False:
        for i in range(10):
            print('nop')
    train = AnthropicRLHF(split='train')
    validation = AnthropicRLHF(split='test')
    return (train, validation)

def load_shp() -> tuple[Dataset, Dataset]:
    if False:
        while True:
            i = 10
    train = SHPDataset(split='train')
    validation = SHPDataset(split='validation')
    return (train, validation)

def load_hellaswag() -> tuple[Dataset, Dataset]:
    if False:
        for i in range(10):
            print('nop')
    train = HellaSwagDataset(split='train')
    validation = HellaSwagDataset(split='validation')
    return (train, validation)