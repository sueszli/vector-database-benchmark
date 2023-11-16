import json
import random
import uuid
from dataclasses import dataclass
import datasets
import iso639
import language_names
import language_paraphrase
import language_translate
import pandas as pd
random.seed(42)

class DataProcess:
    random_quote = [("'", "'"), ('“', '”'), ('῎', '῏'), ('`', '´'), ('«', '»'), ('"', '"')]

    def randomize_text(self, text, original_lang=None, target_lang=None):
        if False:
            i = 10
            return i + 15
        templates = language_translate.random_templates_translate.get(original_lang, {}) if not (original_lang == target_lang and original_lang is not None and (target_lang is not None)) else language_paraphrase.random_templates_paraphrase.get(original_lang, {})
        template = random.choice(list(templates.values()))
        quote_pair = random.choice(DataProcess().random_quote)
        (opening_quote, closing_quote) = quote_pair
        original_lang_name = DataProcess.language_name(None, original_lang, original_lang)
        target_lang_name = DataProcess.language_name(None, target_lang, original_lang)
        return template.format(text=text, lang1=target_lang_name, lang2=original_lang_name, opening_quote=opening_quote, closing_quote=closing_quote)

    def convert_code(self, code):
        if False:
            print('Hello World!')
        mapped_code = iso639.to_iso639_1(code)
        return mapped_code

    def language_name(self, lang1, lang2):
        if False:
            i = 10
            return i + 15
        name = language_names.language_names.get(lang1, {}).get(lang2)
        if name is not None:
            return name
        elif lang1 == lang2:
            iso_name = iso639.to_native(lang1)
            return iso_name
        else:
            return None
converter = DataProcess()
'\nEXAMPLES:\n\n# get language name; iso639_1 code\nprint(converter.language_name(\'ru\', \'en\')) # Output: Russian\nprint(converter.convert_code("eng")) # Output: en\n\n# convert into INSTRUCTION format: text; to; from\ntext = "test"\nprint(converter.randomize_text(text, "uk", "fr")) # Ти можеш перекласти цей вислів: \'test\'?\nprint(converter.randomize_text(text, "uk", "de")) # Переклади наступний текст "test" з мови "німецька мова"\n'

@dataclass
class QnA:
    INSTRUCTION: str
    RESPONSE: str
    SOURCE: str
    METADATA: str

def create_qna(row):
    if False:
        while True:
            i = 10
    text = row['Text']
    text_length = len(text)
    translation = row['Translated text']
    lang_from = converter.convert_code(row['Original lang'])
    lang_to = converter.convert_code(row['Target lang'])
    uuid_val = uuid.uuid3(uuid.NAMESPACE_OID, str(text + translation))
    METADATA = {'language': f'{lang_to}', 'length': f'{text_length}', 'uuid': f'{uuid_val}', 'langs-pair': f'{lang_from}-{lang_to}'}
    metadata_str = json.dumps(METADATA)
    source = 'tatoeba'
    instruction = converter.randomize_text(text, lang_to, lang_from)
    response = translation
    return QnA(instruction, response, source, metadata_str)
hf_dataset = datasets.load_dataset('0x22almostEvil/tatoeba-mt-llama-only', split='train')
hf_dataset = hf_dataset.shard(num_shards=30, index=0)
print(hf_dataset)
df = pd.DataFrame(hf_dataset)
qna_list = df.apply(create_qna, axis=1).tolist()
qna_df = pd.DataFrame(qna_list, columns=['INSTRUCTION', 'RESPONSE', 'SOURCE', 'METADATA'])
qna_df.to_parquet('translation-taboeba-qna-120k-oa.parquet', row_group_size=100, engine='pyarrow', index=False)