"""
RetriBERT model
"""
import math
from typing import Optional
import torch
import torch.utils.checkpoint as checkpoint
from torch import nn
from ....modeling_utils import PreTrainedModel
from ....utils import add_start_docstrings, logging
from ...bert.modeling_bert import BertModel
from .configuration_retribert import RetriBertConfig
logger = logging.get_logger(__name__)
RETRIBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ['yjernite/retribert-base-uncased']

class RetriBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = RetriBertConfig
    load_tf_weights = None
    base_model_prefix = 'retribert'

    def _init_weights(self, module):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the weights'
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
RETRIBERT_START_DOCSTRING = '\n\n    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the\n    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads\n    etc.)\n\n    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage\n    and behavior.\n\n    Parameters:\n        config ([`RetriBertConfig`]): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.\n'

@add_start_docstrings('Bert Based model to embed queries or document for document retrieval.', RETRIBERT_START_DOCSTRING)
class RetriBertModel(RetriBertPreTrainedModel):

    def __init__(self, config: RetriBertConfig) -> None:
        if False:
            print('Hello World!')
        super().__init__(config)
        self.projection_dim = config.projection_dim
        self.bert_query = BertModel(config)
        self.bert_doc = None if config.share_encoders else BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.project_query = nn.Linear(config.hidden_size, config.projection_dim, bias=False)
        self.project_doc = nn.Linear(config.hidden_size, config.projection_dim, bias=False)
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.post_init()

    def embed_sentences_checkpointed(self, input_ids, attention_mask, sent_encoder, checkpoint_batch_size=-1):
        if False:
            i = 10
            return i + 15
        if checkpoint_batch_size < 0 or input_ids.shape[0] < checkpoint_batch_size:
            return sent_encoder(input_ids, attention_mask=attention_mask)[1]
        else:
            device = input_ids.device
            input_shape = input_ids.size()
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            head_mask = [None] * sent_encoder.config.num_hidden_layers
            extended_attention_mask: torch.Tensor = sent_encoder.get_extended_attention_mask(attention_mask, input_shape)

            def partial_encode(*inputs):
                if False:
                    i = 10
                    return i + 15
                encoder_outputs = sent_encoder.encoder(inputs[0], attention_mask=inputs[1], head_mask=head_mask)
                sequence_output = encoder_outputs[0]
                pooled_output = sent_encoder.pooler(sequence_output)
                return pooled_output
            embedding_output = sent_encoder.embeddings(input_ids=input_ids, position_ids=None, token_type_ids=token_type_ids, inputs_embeds=None)
            pooled_output_list = []
            for b in range(math.ceil(input_ids.shape[0] / checkpoint_batch_size)):
                b_embedding_output = embedding_output[b * checkpoint_batch_size:(b + 1) * checkpoint_batch_size]
                b_attention_mask = extended_attention_mask[b * checkpoint_batch_size:(b + 1) * checkpoint_batch_size]
                pooled_output = checkpoint.checkpoint(partial_encode, b_embedding_output, b_attention_mask)
                pooled_output_list.append(pooled_output)
            return torch.cat(pooled_output_list, dim=0)

    def embed_questions(self, input_ids, attention_mask=None, checkpoint_batch_size=-1):
        if False:
            i = 10
            return i + 15
        q_reps = self.embed_sentences_checkpointed(input_ids, attention_mask, self.bert_query, checkpoint_batch_size)
        return self.project_query(q_reps)

    def embed_answers(self, input_ids, attention_mask=None, checkpoint_batch_size=-1):
        if False:
            for i in range(10):
                print('nop')
        a_reps = self.embed_sentences_checkpointed(input_ids, attention_mask, self.bert_query if self.bert_doc is None else self.bert_doc, checkpoint_batch_size)
        return self.project_doc(a_reps)

    def forward(self, input_ids_query: torch.LongTensor, attention_mask_query: Optional[torch.FloatTensor], input_ids_doc: torch.LongTensor, attention_mask_doc: Optional[torch.FloatTensor], checkpoint_batch_size: int=-1) -> torch.FloatTensor:
        if False:
            while True:
                i = 10
        '\n        Args:\n            input_ids_query (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n                Indices of input sequence tokens in the vocabulary for the queries in a batch.\n\n                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and\n                [`PreTrainedTokenizer.__call__`] for details.\n\n                [What are input IDs?](../glossary#input-ids)\n            attention_mask_query (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:\n\n                - 1 for tokens that are **not masked**,\n                - 0 for tokens that are **masked**.\n\n                [What are attention masks?](../glossary#attention-mask)\n            input_ids_doc (`torch.LongTensor` of shape `(batch_size, sequence_length)`):\n                Indices of input sequence tokens in the vocabulary for the documents in a batch.\n            attention_mask_doc (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):\n                Mask to avoid performing attention on documents padding token indices.\n            checkpoint_batch_size (`int`, *optional*, defaults to `-1`):\n                If greater than 0, uses gradient checkpointing to only compute sequence representation on\n                `checkpoint_batch_size` examples at a time on the GPU. All query representations are still compared to\n                all document representations in the batch.\n\n        Return:\n            `torch.FloatTensor``: The bidirectional cross-entropy loss obtained while trying to match each query to its\n            corresponding document and each document to its corresponding query in the batch\n        '
        device = input_ids_query.device
        q_reps = self.embed_questions(input_ids_query, attention_mask_query, checkpoint_batch_size)
        a_reps = self.embed_answers(input_ids_doc, attention_mask_doc, checkpoint_batch_size)
        compare_scores = torch.mm(q_reps, a_reps.t())
        loss_qa = self.ce_loss(compare_scores, torch.arange(compare_scores.shape[1]).to(device))
        loss_aq = self.ce_loss(compare_scores.t(), torch.arange(compare_scores.shape[0]).to(device))
        loss = (loss_qa + loss_aq) / 2
        return loss