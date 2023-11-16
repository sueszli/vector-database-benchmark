def create_empty_dataset(dataset):
    if False:
        return 10
    'Creates an empty dataset for models with no inputs and outputs.\n\n    This function generates an empty dataset, i.e., ``__getitem__()`` only\n    returns ``None``. Its dataset is compatible with the original one.\n    Such datasets used for models which do not take any inputs,\n    neither return any outputs. We expect models, e.g., whose ``forward()``\n    is starting with ``chainermn.functions.recv()`` and ending with\n    ``chainermn.functions.send()``.\n\n    Args:\n        dataset: Dataset to convert.\n\n    Returns:\n        ~chainer.datasets.TransformDataset:\n            Dataset consists of only patterns in the original one.\n    '
    return [()] * len(dataset)