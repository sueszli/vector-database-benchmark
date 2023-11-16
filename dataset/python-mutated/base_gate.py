from paddle import nn

class BaseGate(nn.Layer):

    def __init__(self, num_expert, world_size):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.world_size = world_size
        self.num_expert = num_expert
        self.tot_expert = world_size * num_expert
        self.loss = None

    def forward(self, x):
        if False:
            print('Hello World!')
        raise NotImplementedError('Please implement the forward function.')

    def set_loss(self, loss):
        if False:
            return 10
        self.loss = loss

    def get_loss(self, clear=True):
        if False:
            return 10
        loss = self.loss
        if clear:
            self.loss = None
        return loss