import torch
try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None

def get_rng_state():
    if False:
        print('Hello World!')
    '\n    Get random number generator state of torch, xla and cuda.\n    '
    state = {'torch_rng_state': torch.get_rng_state()}
    if xm is not None:
        state['xla_rng_state'] = xm.get_rng_state()
    if torch.cuda.is_available():
        state['cuda_rng_state'] = torch.cuda.get_rng_state()
    return state

def set_rng_state(state):
    if False:
        print('Hello World!')
    '\n    Set random number generator state of torch, xla and cuda.\n    '
    torch.set_rng_state(state['torch_rng_state'])
    if xm is not None:
        xm.set_rng_state(state['xla_rng_state'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state['cuda_rng_state'])

class set_torch_seed(object):
    """
    Set random seed to torch, xla and cuda.
    """

    def __init__(self, seed):
        if False:
            i = 10
            return i + 15
        assert isinstance(seed, int)
        self.rng_state = get_rng_state()
        torch.manual_seed(seed)
        if xm is not None:
            xm.set_rng_state(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def __enter__(self):
        if False:
            print('Hello World!')
        return self

    def __exit__(self, *exc):
        if False:
            for i in range(10):
                print('nop')
        set_rng_state(self.rng_state)