from fairseq.models.roberta.hub_interface import RobertaHubInterface
import torch
import torch.nn.functional as F

class XMODHubInterface(RobertaHubInterface):

    def extract_features(self, tokens: torch.LongTensor, return_all_hiddens: bool=False, lang_id=None) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > self.model.max_positions():
            raise ValueError('tokens exceeds maximum length: {} > {}'.format(tokens.size(-1), self.model.max_positions()))
        (features, extra) = self.model(tokens.to(device=self.device), features_only=True, return_all_hiddens=return_all_hiddens, lang_id=lang_id)
        if return_all_hiddens:
            inner_states = extra['inner_states']
            return [inner_state.transpose(0, 1) for inner_state in inner_states]
        else:
            return features

    def predict(self, head: str, tokens: torch.LongTensor, return_logits: bool=False, lang_id=None):
        if False:
            print('Hello World!')
        features = self.extract_features(tokens.to(device=self.device), lang_id=lang_id)
        logits = self.model.classification_heads[head](features)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)