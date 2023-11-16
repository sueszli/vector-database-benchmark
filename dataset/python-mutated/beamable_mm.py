import torch
import torch.nn as nn

class BeamableMM(nn.Module):
    """This module provides an optimized MM for beam decoding with attention.

    It leverage the fact that the source-side of the input is replicated beam
    times and the target-side of the input is of width one. This layer speeds up
    inference by replacing the inputs {(bsz x 1 x nhu), (bsz x sz2 x nhu)}
    with smaller inputs {(bsz/beam x beam x nhu), (bsz/beam x sz2 x nhu)}.
    """

    def __init__(self, beam_size=None):
        if False:
            while True:
                i = 10
        super(BeamableMM, self).__init__()
        self.beam_size = beam_size

    def forward(self, input1, input2):
        if False:
            print('Hello World!')
        if not self.training and self.beam_size is not None and (input1.dim() == 3) and (input1.size(1) == 1):
            (bsz, beam) = (input1.size(0), self.beam_size)
            input1 = input1[:, 0, :].unfold(0, beam, beam).transpose(2, 1)
            input2 = input2.unfold(0, beam, beam)[:, :, :, 0]
            if input1.size(0) == 1:
                output = torch.mm(input1[0, :, :], input2[0, :, :])
            else:
                output = input1.bmm(input2)
            return output.view(bsz, 1, -1)
        else:
            return input1.bmm(input2)

    def set_beam_size(self, beam_size):
        if False:
            print('Hello World!')
        self.beam_size = beam_size