"""
Token Detection module
"""
import inspect
import os
import torch
from transformers import PreTrainedModel

class TokenDetection(PreTrainedModel):
    """
    Runs the replaced token detection training objective. This method was first proposed by the ELECTRA model.
    The method consists of a masked language model generator feeding data to a discriminator that determines
    which of the tokens are incorrect. More on this training objective can be found in the ELECTRA paper.
    """

    def __init__(self, generator, discriminator, tokenizer, weight=50.0):
        if False:
            print('Hello World!')
        '\n        Creates a new TokenDetection class.\n\n        Args:\n            generator: Generator model, must be a masked language model\n            discriminator: Discriminator model, must be a model that can detect replaced tokens. Any model can\n                           can be customized for this task. See ElectraForPretraining for more.\n        '
        super().__init__(discriminator.config)
        self.generator = generator
        self.discriminator = discriminator
        self.tokenizer = tokenizer
        self.weight = weight
        if self.generator.config.model_type == self.discriminator.config.model_type:
            self.discriminator.set_input_embeddings(self.generator.get_input_embeddings())
        self.gattention = 'attention_mask' in inspect.signature(self.generator.forward).parameters
        self.dattention = 'attention_mask' in inspect.signature(self.discriminator.forward).parameters

    def forward(self, input_ids=None, labels=None, attention_mask=None, token_type_ids=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Runs a forward pass through the model. This method runs the masked language model then randomly samples\n        the generated tokens and builds a binary classification problem for the discriminator (detecting if each token is correct).\n\n        Args:\n            input_ids: token ids\n            labels: token labels\n            attention_mask: attention mask\n            token_type_ids: segment token indices\n\n        Returns:\n            (loss, generator outputs, discriminator outputs, discriminator labels)\n        '
        dinputs = input_ids.clone()
        inputs = {'attention_mask': attention_mask} if self.gattention else {}
        goutputs = self.generator(input_ids, labels=labels, token_type_ids=token_type_ids, **inputs)
        preds = torch.softmax(goutputs[1], dim=-1)
        preds = preds.view(-1, self.config.vocab_size)
        tokens = torch.multinomial(preds, 1).view(-1)
        tokens = tokens.view(dinputs.shape[0], -1)
        mask = labels.ne(-100)
        dinputs[mask] = tokens[mask]
        correct = tokens == labels
        dlabels = mask.long()
        dlabels[correct] = 0
        inputs = {'attention_mask': attention_mask} if self.dattention else {}
        doutputs = self.discriminator(dinputs, labels=dlabels, token_type_ids=token_type_ids, **inputs)
        loss = goutputs[0] + self.weight * doutputs[0]
        return (loss, goutputs[1], doutputs[1], dlabels)

    def save_pretrained(self, output, state_dict=None, **kwargs):
        if False:
            return 10
        '\n        Saves current model to output directory.\n\n        Args:\n            output: output directory\n            state_dict: model state\n            kwargs: additional keyword arguments\n        '
        super().save_pretrained(output, state_dict, **kwargs)
        gpath = os.path.join(output, 'generator')
        self.tokenizer.save_pretrained(gpath)
        self.generator.save_pretrained(gpath)
        dpath = os.path.join(output, 'discriminator')
        self.tokenizer.save_pretrained(dpath)
        self.discriminator.save_pretrained(dpath)