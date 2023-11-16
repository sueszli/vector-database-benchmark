import os
from pathlib import Path

def write_model_card(model_card_dir, src_lang, tgt_lang, model_name):
    if False:
        for i in range(10):
            print('nop')
    texts = {'en': "Machine learning is great, isn't it?", 'ru': 'Машинное обучение - это здорово, не так ли?', 'de': 'Maschinelles Lernen ist großartig, nicht wahr?'}
    scores = {'wmt19-de-en-6-6-base': [0, 38.37], 'wmt19-de-en-6-6-big': [0, 39.9]}
    pair = f'{src_lang}-{tgt_lang}'
    readme = f'\n---\n\nlanguage:\n- {src_lang}\n- {tgt_lang}\nthumbnail:\ntags:\n- translation\n- wmt19\n- allenai\nlicense: apache-2.0\ndatasets:\n- wmt19\nmetrics:\n- bleu\n---\n\n# FSMT\n\n## Model description\n\nThis is a ported version of fairseq-based [wmt19 transformer](https://github.com/jungokasai/deep-shallow/) for {src_lang}-{tgt_lang}.\n\nFor more details, please, see [Deep Encoder, Shallow Decoder: Reevaluating the Speed-Quality Tradeoff in Machine Translation](https://arxiv.org/abs/2006.10369).\n\n2 models are available:\n\n* [wmt19-de-en-6-6-big](https://huggingface.co/allenai/wmt19-de-en-6-6-big)\n* [wmt19-de-en-6-6-base](https://huggingface.co/allenai/wmt19-de-en-6-6-base)\n\n\n## Intended uses & limitations\n\n#### How to use\n\n```python\nfrom transformers import FSMTForConditionalGeneration, FSMTTokenizer\nmname = "allenai/{model_name}"\ntokenizer = FSMTTokenizer.from_pretrained(mname)\nmodel = FSMTForConditionalGeneration.from_pretrained(mname)\n\ninput = "{texts[src_lang]}"\ninput_ids = tokenizer.encode(input, return_tensors="pt")\noutputs = model.generate(input_ids)\ndecoded = tokenizer.decode(outputs[0], skip_special_tokens=True)\nprint(decoded) # {texts[tgt_lang]}\n\n```\n\n#### Limitations and bias\n\n\n## Training data\n\nPretrained weights were left identical to the original model released by allenai. For more details, please, see the [paper](https://arxiv.org/abs/2006.10369).\n\n## Eval results\n\nHere are the BLEU scores:\n\nmodel   |  transformers\n-------|---------\n{model_name}  |  {scores[model_name][1]}\n\nThe score was calculated using this code:\n\n```bash\ngit clone https://github.com/huggingface/transformers\ncd transformers\nexport PAIR={pair}\nexport DATA_DIR=data/$PAIR\nexport SAVE_DIR=data/$PAIR\nexport BS=8\nexport NUM_BEAMS=5\nmkdir -p $DATA_DIR\nsacrebleu -t wmt19 -l $PAIR --echo src > $DATA_DIR/val.source\nsacrebleu -t wmt19 -l $PAIR --echo ref > $DATA_DIR/val.target\necho $PAIR\nPYTHONPATH="src:examples/seq2seq" python examples/seq2seq/run_eval.py allenai/{model_name} $DATA_DIR/val.source $SAVE_DIR/test_translations.txt --reference_path $DATA_DIR/val.target --score_path $SAVE_DIR/test_bleu.json --bs $BS --task translation --num_beams $NUM_BEAMS\n```\n\n## Data Sources\n\n- [training, etc.](http://www.statmt.org/wmt19/)\n- [test set](http://matrix.statmt.org/test_sets/newstest2019.tgz?1556572561)\n\n\n### BibTeX entry and citation info\n\n```\n@misc{{kasai2020deep,\n    title={{Deep Encoder, Shallow Decoder: Reevaluating the Speed-Quality Tradeoff in Machine Translation}},\n    author={{Jungo Kasai and Nikolaos Pappas and Hao Peng and James Cross and Noah A. Smith}},\n    year={{2020}},\n    eprint={{2006.10369}},\n    archivePrefix={{arXiv}},\n    primaryClass={{cs.CL}}\n}}\n```\n\n'
    model_card_dir.mkdir(parents=True, exist_ok=True)
    path = os.path.join(model_card_dir, 'README.md')
    print(f'Generating {path}')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(readme)
repo_dir = Path(__file__).resolve().parent.parent.parent
model_cards_dir = repo_dir / 'model_cards'
for model_name in ['wmt19-de-en-6-6-base', 'wmt19-de-en-6-6-big']:
    model_card_dir = model_cards_dir / 'allenai' / model_name
    write_model_card(model_card_dir, src_lang='de', tgt_lang='en', model_name=model_name)