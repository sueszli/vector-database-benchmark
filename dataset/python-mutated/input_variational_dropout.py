import torch

class InputVariationalDropout(torch.nn.Dropout):
    """
    Apply the dropout technique in Gal and Ghahramani, [Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142) to a
    3D tensor.

    This module accepts a 3D tensor of shape `(batch_size, num_timesteps, embedding_dim)`
    and samples a single dropout mask of shape `(batch_size, embedding_dim)` and applies
    it to every time step.
    """

    def forward(self, input_tensor):
        if False:
            print('Hello World!')
        '\n        Apply dropout to input tensor.\n\n        # Parameters\n\n        input_tensor : `torch.FloatTensor`\n            A tensor of shape `(batch_size, num_timesteps, embedding_dim)`\n\n        # Returns\n\n        output : `torch.FloatTensor`\n            A tensor of shape `(batch_size, num_timesteps, embedding_dim)` with dropout applied.\n        '
        ones = input_tensor.data.new_ones(input_tensor.shape[0], input_tensor.shape[-1])
        dropout_mask = torch.nn.functional.dropout(ones, self.p, self.training, inplace=False)
        if self.inplace:
            input_tensor *= dropout_mask.unsqueeze(1)
            return None
        else:
            return dropout_mask.unsqueeze(1) * input_tensor