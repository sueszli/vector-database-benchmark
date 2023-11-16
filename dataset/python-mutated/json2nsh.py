import glob
import json
import os.path
import nshutil as nsh

def language_from_filename(path):
    if False:
        print('Hello World!')
    lang = os.path.splitext(os.path.basename(path))[0]
    return (nsh.code_to_language(lang), lang)

def write_langstring(f, language, identifier, text):
    if False:
        i = 10
        return i + 15
    langstring = nsh.make_langstring(language, identifier, text)
    f.write(langstring)

def merge_translations(*translations):
    if False:
        return 10
    merged = {}
    for trans in translations:
        for (k, v) in trans.items():
            if v:
                merged[k] = v
    return merged

def main():
    if False:
        return 10
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    sourcesdir = os.path.join(scriptdir, 'sources')
    outdir = os.path.join(scriptdir, 'out')
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(sourcesdir, 'en.json'), 'r', encoding='utf-8') as infile:
        data_en = json.loads(infile.read())
    for path in glob.glob(os.path.join(sourcesdir, '*.json')):
        (language, language_code) = language_from_filename(path)
        if not language:
            print(f'Unknown language code "{language_code}", skipping')
            continue
        target_file = os.path.join(outdir, f'{language}.nsh')
        print(f'{path} => {target_file}')
        with open(path, 'r', encoding='utf-8') as infile:
            data = json.loads(infile.read())
            data = merge_translations(data_en, data)
            with open(target_file, 'w+', encoding='utf-8') as outfile:
                for (identifier, text) in data.items():
                    write_langstring(outfile, language, identifier, text)
if __name__ == '__main__':
    main()