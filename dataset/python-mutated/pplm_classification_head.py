from torch import nn

class ClassificationHead(nn.Module):
    """Classification Head for  transformer encoders"""

    def __init__(self, class_size, embed_size):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        self.mlp = nn.Linear(embed_size, class_size)

    def forward(self, hidden_state):
        if False:
            i = 10
            return i + 15
        logits = self.mlp(hidden_state)
        return logits