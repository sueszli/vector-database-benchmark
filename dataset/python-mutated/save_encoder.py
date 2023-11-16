"""
Translate pre-processed data with a trained model.
"""
import numpy as np
import torch
from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.sequence_generator import EnsembleModel
from fairseq.utils import safe_hasattr

def get_avg_pool(models, sample, prefix_tokens, src_dict, remove_bpe, has_langtok=False):
    if False:
        i = 10
        return i + 15
    model = EnsembleModel(models)
    encoder_input = {k: v for (k, v) in sample['net_input'].items() if k != 'prev_output_tokens'}
    encoder_outs = model.forward_encoder(encoder_input)
    np_encoder_outs = encoder_outs[0].encoder_out.cpu().numpy().astype(np.float32)
    encoder_mask = 1 - encoder_outs[0].encoder_padding_mask.cpu().numpy().astype(np.float32)
    encoder_mask = np.expand_dims(encoder_mask.T, axis=2)
    if has_langtok:
        encoder_mask = encoder_mask[1:, :, :]
        np_encoder_outs = np_encoder_outs[1, :, :]
    masked_encoder_outs = encoder_mask * np_encoder_outs
    avg_pool = (masked_encoder_outs / encoder_mask.sum(axis=0)).sum(axis=0)
    return avg_pool

def main(args):
    if False:
        while True:
            i = 10
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, '--replace-unk requires a raw text dataset (--raw-text)'
    args.beam = 1
    utils.import_user_module(args)
    if args.max_tokens is None:
        args.max_tokens = 12000
    print(args)
    use_cuda = torch.cuda.is_available() and (not args.cpu)
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary
    print('| loading model(s) from {}'.format(args.path))
    (models, _model_args) = checkpoint_utils.load_model_ensemble(args.path.split(':'), arg_overrides=eval(args.model_overrides), task=task)
    for model in models:
        model.make_generation_fast_(beamable_mm_beam_size=None if args.no_beamable_mm else args.beam, need_attn=args.print_alignment)
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()
    align_dict = utils.load_align_dict(args.replace_unk)
    itr = task.get_batch_iterator(dataset=task.dataset(args.gen_subset), max_tokens=args.max_tokens, max_positions=utils.resolve_max_positions(task.max_positions()), ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test, required_batch_size_multiple=args.required_batch_size_multiple, num_shards=args.num_shards, shard_id=args.shard_id, num_workers=args.num_workers).next_epoch_itr(shuffle=False)
    num_sentences = 0
    source_sentences = []
    shard_id = 0
    all_avg_pool = None
    encoder_has_langtok = safe_hasattr(task.args, 'encoder_langtok') and task.args.encoder_langtok is not None and safe_hasattr(task.args, 'lang_tok_replacing_bos_eos') and (not task.args.lang_tok_replacing_bos_eos)
    with progress_bar.build_progress_bar(args, itr) as t:
        for sample in t:
            if sample is None:
                print('Skipping None')
                continue
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue
            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample['target'][:, :args.prefix_size]
            with torch.no_grad():
                avg_pool = get_avg_pool(models, sample, prefix_tokens, src_dict, args.post_process, has_langtok=encoder_has_langtok)
                if all_avg_pool is not None:
                    all_avg_pool = np.concatenate((all_avg_pool, avg_pool))
                else:
                    all_avg_pool = avg_pool
            if not isinstance(sample['id'], list):
                sample_ids = sample['id'].tolist()
            else:
                sample_ids = sample['id']
            for (i, sample_id) in enumerate(sample_ids):
                src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
                if align_dict is not None:
                    src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                elif src_dict is not None:
                    src_str = src_dict.string(src_tokens, args.post_process)
                else:
                    src_str = ''
                if not args.quiet:
                    if src_dict is not None:
                        print('S-{}\t{}'.format(sample_id, src_str))
                source_sentences.append(f'{sample_id}\t{src_str}')
            num_sentences += sample['nsentences']
            if all_avg_pool.shape[0] >= 1000000:
                with open(f'{args.encoder_save_dir}/all_avg_pool.{args.source_lang}.{shard_id}', 'w') as avg_pool_file:
                    all_avg_pool.tofile(avg_pool_file)
                with open(f'{args.encoder_save_dir}/sentences.{args.source_lang}.{shard_id}', 'w') as sentence_file:
                    sentence_file.writelines((f'{line}\n' for line in source_sentences))
                all_avg_pool = None
                source_sentences = []
                shard_id += 1
    if all_avg_pool is not None:
        with open(f'{args.encoder_save_dir}/all_avg_pool.{args.source_lang}.{shard_id}', 'w') as avg_pool_file:
            all_avg_pool.tofile(avg_pool_file)
        with open(f'{args.encoder_save_dir}/sentences.{args.source_lang}.{shard_id}', 'w') as sentence_file:
            sentence_file.writelines((f'{line}\n' for line in source_sentences))
    return None

def cli_main():
    if False:
        while True:
            i = 10
    parser = options.get_generation_parser()
    parser.add_argument('--encoder-save-dir', default='', type=str, metavar='N', help='directory to save encoder outputs')
    args = options.parse_args_and_arch(parser)
    main(args)
if __name__ == '__main__':
    cli_main()