import json
import pathlib
import requests
lingva = 'https://lingva.ml/api/v1/en'

def format_lang(lang: str) -> str:
    if False:
        while True:
            i = 10
    if 'zh-' in lang:
        if lang == 'zh-TW':
            return 'zh_HANT'
        return 'zh'
    return lang.replace('lang_', '')

def translate(v: str, lang: str) -> str:
    if False:
        i = 10
        return i + 15
    lang = format_lang(lang)
    lingva_req = f'{lingva}/{lang}/{v}'
    response = requests.get(lingva_req).json()
    if 'translation' in response:
        return response['translation']
    return ''
if __name__ == '__main__':
    file_path = pathlib.Path(__file__).parent.resolve()
    tl_path = 'app/static/settings/translations.json'
    with open(f'{file_path}/../{tl_path}', 'r+', encoding='utf-8') as tl_file:
        tl_data = json.load(tl_file)
        en_tl = tl_data['lang_en']
        for (k, v) in en_tl.items():
            for lang in tl_data:
                if lang == 'lang_en' or k in tl_data[lang]:
                    continue
                translation = ''
                if len(k) == 0:
                    translation = v
                else:
                    translation = translate(v, lang)
                if len(translation) == 0:
                    print(f'! Unable to translate {lang}[{k}]')
                    continue
                print(f'{lang}[{k}] = {translation}')
                tl_data[lang][k] = translation
        print(json.dumps(tl_data, indent=4, ensure_ascii=False))