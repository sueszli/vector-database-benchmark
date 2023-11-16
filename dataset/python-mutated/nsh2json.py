import glob
import json
import os.path
import nshutil as nsh

def language_from_filename(path):
    if False:
        i = 10
        return i + 15
    lang = os.path.splitext(os.path.basename(path))[0]
    return (lang, nsh.language_to_code(lang))

def extract_strings(f):
    if False:
        while True:
            i = 10
    for line in f:
        parsed = nsh.parse_langstring(line)
        if parsed:
            yield parsed

def main():
    if False:
        print('Hello World!')
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    sourcesdir = os.path.join(scriptdir, 'sources')
    outdir = os.path.join(scriptdir, 'out')
    for path in glob.glob(os.path.join(outdir, '*.nsh')):
        (language, language_code) = language_from_filename(path)
        if not language_code:
            print(f'Unknown language "{language}", skipping')
            continue
        target_file = os.path.join(sourcesdir, f'{language_code}.json')
        print(f'{path} => {target_file}')
        with open(path, 'r', encoding='utf-8') as infile:
            output = {}
            for (identifier, text) in extract_strings(infile):
                output[identifier] = text
            with open(target_file, 'w+', encoding='utf-8') as outfile:
                outfile.write(json.dumps(output, ensure_ascii=False, indent=4))
if __name__ == '__main__':
    main()