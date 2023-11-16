from typing import Sequence
from torch.utils.data import DataLoader
from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.utils.pytorch import get_forward_args, get_forward_defaults, get_conditional_args

def complement_input_sample(model, input_sample):
    if False:
        while True:
            i = 10
    '\n    This function will give a complemented input sample\n    Mainly using default value to complete.\n    '
    forward_args = get_forward_args(model)
    forward_defaults = get_forward_defaults(model)
    input_sample_length = 1
    if isinstance(input_sample, Sequence):
        input_sample_length = len(input_sample)
    if len(forward_args) == input_sample_length:
        return input_sample
    if len(forward_args) > len(forward_defaults) + input_sample_length:
        invalidInputError(False, 'not enough input_sample provided!')
    if isinstance(input_sample, Sequence):
        input_sample_complement = input_sample
        input_sample_complement += forward_defaults[-(len(forward_args) - input_sample_length):]
    else:
        input_sample_complement = []
        input_sample_complement.append(input_sample)
        input_sample_complement += list(forward_defaults[-(len(forward_args) - input_sample_length):])
    return tuple(input_sample_complement)

def get_input_example(model, input_sample, forward_args):
    if False:
        i = 10
        return i + 15
    if isinstance(input_sample, DataLoader):
        input_sample = next(iter(input_sample))
        if isinstance(input_sample, Sequence):
            if len(input_sample) <= 2:
                input_sample = input_sample[0]
            else:
                input_sample = tuple(input_sample[:len(forward_args)])
    elif input_sample is None:
        if getattr(model, 'example_input_array', None) is not None:
            input_sample = model.example_input_array
        elif getattr(model, 'trainer', None):
            for dataloader_fn in [model.test_dataloader, model.train_dataloader, model.val_dataloader]:
                try:
                    dataloader = dataloader_fn()
                    input_sample = next(iter(dataloader))
                    if isinstance(input_sample, Sequence):
                        input_sample = tuple(list(input_sample)[:len(forward_args)])
                    break
                except Exception as _e:
                    pass
            if input_sample is None:
                try:
                    predict_dataloader = model.predict_dataloader()
                    input_sample = tuple(next(iter(predict_dataloader)))
                except Exception as _e:
                    pass
        else:
            invalidInputError(False, 'You must specify an input_sample or call `Trainer.fit` on the model first to use `eval_openvino`')
    model.example_input_array = input_sample
    return input_sample