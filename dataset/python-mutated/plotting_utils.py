from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import torch

def validate_numpy_array(value: Any):
    if False:
        print('Hello World!')
    '\n    Validates the input and makes sure it returns a numpy array (i.e on CPU)\n\n    Args:\n        value (Any): the input value\n\n    Raises:\n        TypeError: if the value is not a numpy array or torch tensor\n\n    Returns:\n        np.ndarray: numpy array of the value\n    '
    if isinstance(value, np.ndarray):
        pass
    elif isinstance(value, list):
        value = np.array(value)
    elif torch.is_tensor(value):
        value = value.cpu().numpy()
    else:
        raise TypeError('Value must be a numpy array, a torch tensor or a list')
    return value

def get_spec_from_most_probable_state(log_alpha_scaled, means, decoder=None):
    if False:
        return 10
    'Get the most probable state means from the log_alpha_scaled.\n\n    Args:\n        log_alpha_scaled (torch.Tensor): Log alpha scaled values.\n            - Shape: :math:`(T, N)`\n        means (torch.Tensor): Means of the states.\n            - Shape: :math:`(N, T, D_out)`\n        decoder (torch.nn.Module): Decoder module to decode the latent to melspectrogram. Defaults to None.\n    '
    max_state_numbers = torch.max(log_alpha_scaled, dim=1)[1]
    max_len = means.shape[0]
    n_mel_channels = means.shape[2]
    max_state_numbers = max_state_numbers.unsqueeze(1).unsqueeze(1).expand(max_len, 1, n_mel_channels)
    means = torch.gather(means, 1, max_state_numbers).squeeze(1).to(log_alpha_scaled.dtype)
    if decoder is not None:
        mel = decoder(means.T.unsqueeze(0), torch.tensor([means.shape[0]], device=means.device), reverse=True)[0].squeeze(0).T
    else:
        mel = means
    return mel

def plot_transition_probabilities_to_numpy(states, transition_probabilities, output_fig=False):
    if False:
        for i in range(10):
            print('nop')
    'Generates trainsition probabilities plot for the states and the probability of transition.\n\n    Args:\n        states (torch.IntTensor): the states\n        transition_probabilities (torch.FloatTensor): the transition probabilities\n    '
    states = validate_numpy_array(states)
    transition_probabilities = validate_numpy_array(transition_probabilities)
    (fig, ax) = plt.subplots(figsize=(30, 3))
    ax.plot(transition_probabilities, 'o')
    ax.set_title('Transition probability of state')
    ax.set_xlabel('hidden state')
    ax.set_ylabel('probability')
    ax.set_xticks([i for i in range(len(transition_probabilities))])
    ax.set_xticklabels([int(x) for x in states], rotation=90)
    plt.tight_layout()
    if not output_fig:
        plt.close()
    return fig