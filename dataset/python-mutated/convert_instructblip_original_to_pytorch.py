"""
Convert InstructBLIP checkpoints from the original repository.

URL: https://github.com/salesforce/LAVIS/tree/main/projects/instructblip
"""
import argparse
import requests
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from transformers import AutoTokenizer, BlipImageProcessor, InstructBlipConfig, InstructBlipForConditionalGeneration, InstructBlipProcessor, InstructBlipQFormerConfig, InstructBlipVisionConfig, LlamaConfig, LlamaTokenizerFast, T5Config, T5TokenizerFast
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

def load_demo_image():
    if False:
        return 10
    url = 'https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg'
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    return image

def create_rename_keys(config):
    if False:
        return 10
    rename_keys = []
    rename_keys.append(('visual_encoder.cls_token', 'vision_model.embeddings.class_embedding'))
    rename_keys.append(('visual_encoder.pos_embed', 'vision_model.embeddings.position_embedding'))
    rename_keys.append(('visual_encoder.patch_embed.proj.weight', 'vision_model.embeddings.patch_embedding.weight'))
    rename_keys.append(('visual_encoder.patch_embed.proj.bias', 'vision_model.embeddings.patch_embedding.bias'))
    rename_keys.append(('ln_vision.weight', 'vision_model.post_layernorm.weight'))
    rename_keys.append(('ln_vision.bias', 'vision_model.post_layernorm.bias'))
    for i in range(config.vision_config.num_hidden_layers):
        rename_keys.append((f'visual_encoder.blocks.{i}.norm1.weight', f'vision_model.encoder.layers.{i}.layer_norm1.weight'))
        rename_keys.append((f'visual_encoder.blocks.{i}.norm1.bias', f'vision_model.encoder.layers.{i}.layer_norm1.bias'))
        rename_keys.append((f'visual_encoder.blocks.{i}.norm2.weight', f'vision_model.encoder.layers.{i}.layer_norm2.weight'))
        rename_keys.append((f'visual_encoder.blocks.{i}.norm2.bias', f'vision_model.encoder.layers.{i}.layer_norm2.bias'))
        rename_keys.append((f'visual_encoder.blocks.{i}.attn.qkv.weight', f'vision_model.encoder.layers.{i}.self_attn.qkv.weight'))
        rename_keys.append((f'visual_encoder.blocks.{i}.attn.proj.weight', f'vision_model.encoder.layers.{i}.self_attn.projection.weight'))
        rename_keys.append((f'visual_encoder.blocks.{i}.attn.proj.bias', f'vision_model.encoder.layers.{i}.self_attn.projection.bias'))
        rename_keys.append((f'visual_encoder.blocks.{i}.mlp.fc1.weight', f'vision_model.encoder.layers.{i}.mlp.fc1.weight'))
        rename_keys.append((f'visual_encoder.blocks.{i}.mlp.fc1.bias', f'vision_model.encoder.layers.{i}.mlp.fc1.bias'))
        rename_keys.append((f'visual_encoder.blocks.{i}.mlp.fc2.weight', f'vision_model.encoder.layers.{i}.mlp.fc2.weight'))
        rename_keys.append((f'visual_encoder.blocks.{i}.mlp.fc2.bias', f'vision_model.encoder.layers.{i}.mlp.fc2.bias'))
    rename_keys.append(('Qformer.bert.embeddings.LayerNorm.weight', 'qformer.embeddings.layernorm.weight'))
    rename_keys.append(('Qformer.bert.embeddings.LayerNorm.bias', 'qformer.embeddings.layernorm.bias'))
    return rename_keys

def rename_key(dct, old, new):
    if False:
        i = 10
        return i + 15
    val = dct.pop(old)
    dct[new] = val

def read_in_q_v_bias(state_dict, config):
    if False:
        for i in range(10):
            print('nop')
    for i in range(config.vision_config.num_hidden_layers):
        q_bias = state_dict.pop(f'visual_encoder.blocks.{i}.attn.q_bias')
        v_bias = state_dict.pop(f'visual_encoder.blocks.{i}.attn.v_bias')
        qkv_bias = torch.cat((q_bias, torch.zeros_like(v_bias, requires_grad=False), v_bias))
        state_dict[f'vision_model.encoder.layers.{i}.self_attn.qkv.bias'] = qkv_bias

def get_blip2_config(model_name):
    if False:
        for i in range(10):
            print('nop')
    image_size = 364 if 'coco' in model_name else 224
    vision_config = InstructBlipVisionConfig(image_size=image_size).to_dict()
    if 't5-xl' in model_name:
        text_config = T5Config.from_pretrained('google/flan-t5-xl', dense_act_fn='gelu', bos_token_id=1).to_dict()
    elif 't5-xxl' in model_name:
        text_config = T5Config.from_pretrained('google/flan-t5-xxl', dense_act_fn='gelu', bos_token_id=1).to_dict()
    elif 'vicuna-7b' in model_name:
        text_config = LlamaConfig.from_pretrained('decapoda-research/llama-7b-hf', vocab_size=32001).to_dict()
    elif 'vicuna-13b' in model_name:
        text_config = LlamaConfig.from_pretrained('decapoda-research/llama-13b-hf', vocab_size=32001).to_dict()
    else:
        raise ValueError('Model name not supported')
    qformer_config = InstructBlipQFormerConfig(vocab_size=30523).to_dict()
    config = InstructBlipConfig(vision_config=vision_config, text_config=text_config, qformer_config=qformer_config)
    return (config, image_size)

@torch.no_grad()
def convert_blip2_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    if False:
        while True:
            i = 10
    "\n    Copy/paste/tweak model's weights to Transformers design.\n    "
    qformer_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', truncation_side='left')
    qformer_tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    if 't5' in model_name:
        tokenizer = T5TokenizerFast.from_pretrained('google/flan-t5-xl', truncation_side='left')
    elif 'vicuna' in model_name:
        tokenizer = LlamaTokenizerFast.from_pretrained('huggyllama/llama-7b', truncation_side='left', bos_token='</s>', unk_token='</s>')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    (config, image_size) = get_blip2_config(model_name)
    hf_model = InstructBlipForConditionalGeneration(config).eval()
    model_name_to_original = {'instructblip-vicuna-7b': ('blip2_vicuna_instruct', 'vicuna7b'), 'instructblip-vicuna-13b': ('blip2_vicuna_instruct', 'vicuna13b'), 'instructblip-flan-t5-xl': ('blip2_t5_instruct', 'flant5xl'), 'instructblip-flan-t5-xxl': ('blip2_t5_instruct', 'flant5xxl')}
    (name, type) = model_name_to_original[model_name]
    print('Loading original model...')
    hf_model_device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    lavis_device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    (original_model, vis_processors, _) = load_model_and_preprocess(name=name, model_type=type, is_eval=True, device=lavis_device)
    original_model.eval()
    print('Done!')
    state_dict = original_model.state_dict()
    rename_keys = create_rename_keys(config)
    for (src, dest) in rename_keys:
        rename_key(state_dict, src, dest)
    for (key, val) in state_dict.copy().items():
        val = state_dict.pop(key)
        if key.startswith('Qformer.bert'):
            key = key.replace('Qformer.bert', 'qformer')
        if 'attention.self' in key:
            key = key.replace('self', 'attention')
        if 'llm_proj' in key:
            key = key.replace('llm_proj', 'language_projection')
        if 't5_proj' in key:
            key = key.replace('t5_proj', 'language_projection')
        if key.startswith('llm_model'):
            key = key.replace('llm_model', 'language_model')
        if key.startswith('t5'):
            key = key.replace('t5', 'language')
        state_dict[key] = val
    read_in_q_v_bias(state_dict, config)
    hf_model.load_state_dict(state_dict, strict=True)
    image = load_demo_image()
    prompt = 'What is unusual about this image?'
    image_processor = BlipImageProcessor(size={'height': image_size, 'width': image_size}, image_mean=OPENAI_CLIP_MEAN, image_std=OPENAI_CLIP_STD)
    processor = InstructBlipProcessor(image_processor=image_processor, tokenizer=tokenizer, qformer_tokenizer=qformer_tokenizer)
    inputs = processor(images=image, text=prompt, return_tensors='pt').to(hf_model_device)
    original_pixel_values = vis_processors['eval'](image).unsqueeze(0).to(lavis_device)
    pixel_values = inputs.pixel_values
    assert torch.allclose(original_pixel_values.to(pixel_values.device), pixel_values)
    original_model.to(lavis_device)
    hf_model.to(hf_model_device)
    with torch.no_grad():
        if 'vicuna' in model_name:
            original_logits = original_model({'image': original_pixel_values, 'text_input': [prompt]}).logits
            logits = hf_model(**inputs).logits
        else:
            original_logits = original_model({'image': original_pixel_values, 'text_input': [prompt], 'text_output': ['\n']}).logits
            label_input_ids = tokenizer('\n', return_tensors='pt').input_ids.to(hf_model_device)
            labels = label_input_ids.masked_fill(label_input_ids == tokenizer.pad_token_id, -100)
            logits = hf_model(**inputs, labels=labels).logits
    print('First values of original logits:', original_logits[0, :3, :3])
    print('First values of HF logits:', logits[0, :3, :3])
    assert original_logits.shape == logits.shape
    atol = 0.0001 if 'vicuna' in model_name else 1e-05
    assert torch.allclose(original_logits.to(logits.device), logits, atol=atol)
    print('Looks ok!')
    print('Generating with original model...')
    original_outputs = original_model.generate({'image': original_pixel_values, 'prompt': prompt}, num_beams=5)
    print('Generating with HF model...')
    outputs = hf_model.generate(**inputs, do_sample=False, num_beams=5, max_length=256, min_length=1, top_p=0.9, repetition_penalty=1.5, length_penalty=1.0, temperature=1)
    if 'vicuna' in model_name:
        outputs[outputs == 0] = 2
    print('Original generation:', original_outputs)
    output_text = processor.batch_decode(outputs, skip_special_tokens=True)
    output_text = [text.strip() for text in output_text]
    print('HF generation:', output_text)
    if pytorch_dump_folder_path is not None:
        processor.save_pretrained(pytorch_dump_folder_path)
        hf_model.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        processor.push_to_hub(f'Salesforce/{model_name}')
        hf_model.push_to_hub(f'Salesforce/{model_name}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    choices = ['instructblip-vicuna-7b', 'instructblip-vicuna-13b', 'instructblip-flan-t5-xl', 'instructblip-flan-t5-xxl']
    parser.add_argument('--model_name', default='instructblip-flan-t5-xl', choices=choices, type=str, help='Path to hf config.json of model to convert')
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=str, help='Path to the output PyTorch model.')
    parser.add_argument('--push_to_hub', action='store_true', help='Whether to push the model and processor to the hub after converting')
    args = parser.parse_args()
    convert_blip2_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)