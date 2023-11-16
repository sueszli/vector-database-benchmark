import torch
from fairseq import utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, LabelSmoothedCrossEntropyCriterionConfig, label_smoothed_nll_loss

@register_criterion('speech_text_pretrain_cross_entropy', dataclass=LabelSmoothedCrossEntropyCriterionConfig)
class SpeechTextPreTrainCrossEntCriterion(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, report_accuracy=False):
        if False:
            return 10
        super().__init__(task, sentence_avg, label_smoothing, report_accuracy=report_accuracy)

    def forward(self, model, sample, reduce=True):
        if False:
            while True:
                i = 10
        net_output = model(**sample['net_input'])
        (loss, nll_loss, nsentences, ntokens, n_correct) = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = nsentences if self.sentence_avg else ntokens
        logging_output = {'loss': loss.data, 'nll_loss': nll_loss.data, 'ntokens': ntokens, 'nsentences': nsentences, 'sample_size': sample_size}
        if self.report_accuracy:
            logging_output['n_correct'] = utils.item(n_correct)
            logging_output['total'] = utils.item(ntokens)
        return (loss, sample_size, logging_output)

    def get_lprobs_and_target(self, model, net_output, sample):
        if False:
            print('Hello World!')
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        assert self.ignore_prefix_size == 0
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, 'batch_first', False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        return (lprobs, target)

    def compute_loss(self, model, net_output, sample, reduce=True):
        if False:
            while True:
                i = 10
        (lprobs, target) = self.get_lprobs_and_target(model, net_output, sample)
        n_correct = 0
        if isinstance(target, dict):
            t_lprobs = target['target_logprobs']
            if not lprobs.batch_first:
                lprobs = lprobs.transpose(0, 1)
                t_lprobs = t_lprobs.transpose(0, 1)
            (nsentences, seq_len) = lprobs.size()[:2]
            ntokens = nsentences * seq_len
            t_probs = t_lprobs.exp()
            mask_indices = net_output[1]['mask_indices'][0] if len(net_output[1]['mask_indices']) > 0 else None
            if mask_indices is not None:
                t_probs = t_probs.masked_fill(mask_indices.eq(False).unsqueeze(-1), 0)
                ntokens = mask_indices.int().sum()
            t_probs = t_probs.detach()
            t_lprobs = t_lprobs.detach()
            loss = -(t_probs * (lprobs - t_lprobs)).sum() if reduce else -(t_probs * (lprobs - t_lprobs)).sum(-1, keepdim=True)
            nll_loss = loss
        else:
            nsentences = target.size(0)
            mask = target.ne(self.padding_idx)
            (loss, nll_loss) = label_smoothed_nll_loss(lprobs.view(-1, lprobs.size(-1)), target.view(-1), self.eps, ignore_index=self.padding_idx, reduce=reduce)
            n_correct = torch.sum(lprobs.argmax(-1).masked_select(mask).eq(target.masked_select(mask)))
            ntokens = torch.sum(mask)
        return (loss, nll_loss, nsentences, ntokens, n_correct)