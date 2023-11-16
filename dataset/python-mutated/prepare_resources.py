"""
Converts a directory of models organized by type into a directory organized by language.

Also produces the resources.json file.

For example, on the cluster, you can do this:

python3 -m stanza.resources.prepare_resources --input_dir /u/nlp/software/stanza/models/current-models-1.5.0 --output_dir /u/nlp/software/stanza/models/1.5.0 > resources.out 2>&1
nlprun -a stanza-1.2 -q john "python3 -m stanza.resources.prepare_resources --input_dir /u/nlp/software/stanza/models/current-models-1.5.0 --output_dir /u/nlp/software/stanza/models/1.5.0" -o resources.out
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import hashlib
import shutil
import zipfile
from stanza import __resources_version__
from stanza.models.common.constant import lcode2lang, two_to_three_letters, three_to_two_letters
from stanza.resources.default_packages import PACKAGES, TRANSFORMERS, TRANSFORMER_NICKNAMES
from stanza.resources.default_packages import default_treebanks, no_pretrain_languages, default_pretrains, pos_pretrains, depparse_pretrains, ner_pretrains, default_charlms, pos_charlms, depparse_charlms, ner_charlms, lemma_charlms, known_nicknames
from stanza.utils.get_tqdm import get_tqdm
tqdm = get_tqdm()

def parse_args():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/u/nlp/software/stanza/models/current-models-%s' % __resources_version__, help='Input dir for various models.  Defaults to the recommended home on the nlp cluster')
    parser.add_argument('--output_dir', type=str, default='/u/nlp/software/stanza/models/%s' % __resources_version__, help='Output dir for various models.')
    parser.add_argument('--packages_only', action='store_true', default=False, help='Only build the package maps instead of rebuilding everything')
    parser.add_argument('--lang', type=str, default=None, help='Only process this language.  If left blank, will prepare all languages.  To use this argument, a previous prepared resources with all of the languages is necessary.')
    args = parser.parse_args()
    args.input_dir = os.path.abspath(args.input_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    return args
default_ners = {'af': 'nchlt', 'ar': 'aqmar_charlm', 'bg': 'bsnlp19', 'da': 'ddt', 'de': 'germeval2014', 'en': 'ontonotes', 'es': 'conll02', 'fa': 'arman', 'fi': 'turku', 'fr': 'wikiner', 'hu': 'combined', 'hy': 'armtdp', 'it': 'fbk', 'ja': 'gsd', 'kk': 'kazNERD', 'mr': 'l3cube', 'my': 'ucsy', 'nb': 'norne', 'nl': 'conll02', 'nn': 'norne', 'pl': 'nkjp', 'ru': 'wikiner', 'sd': 'siner', 'sv': 'suc3shuffle', 'th': 'lst20', 'tr': 'starlang', 'uk': 'languk', 'vi': 'vlsp', 'zh-hans': 'ontonotes'}
default_sentiment = {'en': 'sstplus', 'de': 'sb10k', 'es': 'tass2020', 'mr': 'l3cube', 'vi': 'vsfc', 'zh-hans': 'ren'}
default_constituency = {'da': 'arboretum_charlm', 'en': 'ptb3-revised_charlm', 'es': 'combined_charlm', 'id': 'icon_charlm', 'it': 'vit_charlm', 'ja': 'alt_charlm', 'pt': 'cintil_charlm', 'vi': 'vlsp22_charlm', 'zh-hans': 'ctb-51_charlm'}
default_tokenizer = {'my': 'alt'}
allowed_empty_languages = ['th', 'my', 'sd']
processor_to_ending = {'tokenize': 'tokenizer', 'mwt': 'mwt_expander', 'lemma': 'lemmatizer', 'pos': 'tagger', 'depparse': 'parser', 'pretrain': 'pretrain', 'ner': 'nertagger', 'forward_charlm': 'forward_charlm', 'backward_charlm': 'backward_charlm', 'sentiment': 'sentiment', 'constituency': 'constituency', 'langid': 'langid'}
ending_to_processor = {j: i for (i, j) in processor_to_ending.items()}
PROCESSORS = list(processor_to_ending.keys())

def ensure_dir(dir):
    if False:
        print('Hello World!')
    Path(dir).mkdir(parents=True, exist_ok=True)

def copy_file(src, dst):
    if False:
        for i in range(10):
            print('nop')
    ensure_dir(Path(dst).parent)
    shutil.copy2(src, dst)

def get_md5(path):
    if False:
        print('Hello World!')
    data = open(path, 'rb').read()
    return hashlib.md5(data).hexdigest()

def split_model_name(model):
    if False:
        while True:
            i = 10
    '\n    Split model names by _\n\n    Takes into account packages with _ and processor types with _\n    '
    model = model[:-3].replace('.', '_')
    for processor in sorted(ending_to_processor.keys(), key=lambda x: -len(x)):
        if model.endswith(processor):
            model = model[:-(len(processor) + 1)]
            processor = ending_to_processor[processor]
            break
    else:
        raise AssertionError(f'Could not find a processor type in {model}')
    (lang, package) = model.split('_', 1)
    return (lang, package, processor)

def split_package(package):
    if False:
        while True:
            i = 10
    if package.endswith('_finetuned'):
        package = package[:-10]
    if package.endswith('_nopretrain'):
        package = package[:-11]
        return (package, False, False)
    if package.endswith('_nocharlm'):
        package = package[:-9]
        return (package, True, False)
    if package.endswith('_charlm'):
        package = package[:-7]
        return (package, True, True)
    underscore = package.rfind('_')
    if underscore >= 0:
        nickname = package[underscore + 1:]
        if nickname in known_nicknames():
            return (package[:underscore], True, True)
    return (package, True, True)

def get_pretrain_package(lang, package, model_pretrains, default_pretrains):
    if False:
        for i in range(10):
            print('nop')
    (package, uses_pretrain, _) = split_package(package)
    if not uses_pretrain or lang in no_pretrain_languages:
        return None
    elif model_pretrains is not None and lang in model_pretrains and (package in model_pretrains[lang]):
        return model_pretrains[lang][package]
    elif lang in default_pretrains:
        return default_pretrains[lang]
    raise RuntimeError('pretrain not specified for lang %s package %s' % (lang, package))

def get_charlm_package(lang, package, model_charlms, default_charlms):
    if False:
        print('Hello World!')
    (package, _, uses_charlm) = split_package(package)
    if not uses_charlm:
        return None
    if model_charlms is not None and lang in model_charlms and (package in model_charlms[lang]):
        return model_charlms[lang][package]
    else:
        return default_charlms.get(lang, None)

def get_con_dependencies(lang, package):
    if False:
        return 10
    pretrain_package = get_pretrain_package(lang, package, None, default_pretrains)
    dependencies = [{'model': 'pretrain', 'package': pretrain_package}]
    charlm_package = default_charlms.get(lang, None)
    if charlm_package is not None:
        dependencies.append({'model': 'forward_charlm', 'package': charlm_package})
        dependencies.append({'model': 'backward_charlm', 'package': charlm_package})
    return dependencies

def get_pos_charlm_package(lang, package):
    if False:
        print('Hello World!')
    return get_charlm_package(lang, package, pos_charlms, default_charlms)

def get_pos_dependencies(lang, package):
    if False:
        for i in range(10):
            print('nop')
    dependencies = []
    pretrain_package = get_pretrain_package(lang, package, pos_pretrains, default_pretrains)
    if pretrain_package is not None:
        dependencies.append({'model': 'pretrain', 'package': pretrain_package})
    charlm_package = get_pos_charlm_package(lang, package)
    if charlm_package is not None:
        dependencies.append({'model': 'forward_charlm', 'package': charlm_package})
        dependencies.append({'model': 'backward_charlm', 'package': charlm_package})
    return dependencies

def get_lemma_charlm_package(lang, package):
    if False:
        print('Hello World!')
    return get_charlm_package(lang, package, lemma_charlms, default_charlms)

def get_lemma_dependencies(lang, package):
    if False:
        print('Hello World!')
    dependencies = []
    charlm_package = get_lemma_charlm_package(lang, package)
    if charlm_package is not None:
        dependencies.append({'model': 'forward_charlm', 'package': charlm_package})
        dependencies.append({'model': 'backward_charlm', 'package': charlm_package})
    return dependencies

def get_depparse_charlm_package(lang, package):
    if False:
        return 10
    return get_charlm_package(lang, package, depparse_charlms, default_charlms)

def get_depparse_dependencies(lang, package):
    if False:
        return 10
    dependencies = []
    pretrain_package = get_pretrain_package(lang, package, depparse_pretrains, default_pretrains)
    if pretrain_package is not None:
        dependencies.append({'model': 'pretrain', 'package': pretrain_package})
    charlm_package = get_depparse_charlm_package(lang, package)
    if charlm_package is not None:
        dependencies.append({'model': 'forward_charlm', 'package': charlm_package})
        dependencies.append({'model': 'backward_charlm', 'package': charlm_package})
    return dependencies

def get_ner_charlm_package(lang, package):
    if False:
        for i in range(10):
            print('nop')
    return get_charlm_package(lang, package, ner_charlms, default_charlms)

def get_ner_dependencies(lang, package):
    if False:
        return 10
    dependencies = []
    pretrain_package = get_pretrain_package(lang, package, ner_pretrains, default_pretrains)
    if pretrain_package is not None:
        dependencies.append({'model': 'pretrain', 'package': pretrain_package})
    charlm_package = get_ner_charlm_package(lang, package)
    if charlm_package is not None:
        dependencies.append({'model': 'forward_charlm', 'package': charlm_package})
        dependencies.append({'model': 'backward_charlm', 'package': charlm_package})
    return dependencies

def get_sentiment_dependencies(lang, package):
    if False:
        i = 10
        return i + 15
    '\n    Return a list of dependencies for the sentiment model\n\n    Generally this will be pretrain, forward & backward charlm\n    So far, this invariant is true:\n    sentiment models use the default pretrain for the language\n    also, they all use the default charlm for a language\n    '
    pretrain_package = get_pretrain_package(lang, package, None, default_pretrains)
    dependencies = [{'model': 'pretrain', 'package': pretrain_package}]
    charlm_package = default_charlms.get(lang, None)
    if charlm_package is not None:
        dependencies.append({'model': 'forward_charlm', 'package': charlm_package})
        dependencies.append({'model': 'backward_charlm', 'package': charlm_package})
    return dependencies

def get_dependencies(processor, lang, package):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the dependencies for a particular lang/package based on the package name\n\n    The package can include descriptors such as _nopretrain, _nocharlm, _charlm\n    which inform whether or not this particular model uses charlm or pretrain\n    '
    if processor == 'depparse':
        return get_depparse_dependencies(lang, package)
    elif processor == 'lemma':
        return get_lemma_dependencies(lang, package)
    elif processor == 'pos':
        return get_pos_dependencies(lang, package)
    elif processor == 'ner':
        return get_ner_dependencies(lang, package)
    elif processor == 'sentiment':
        return get_sentiment_dependencies(lang, package)
    elif processor == 'constituency':
        return get_con_dependencies(lang, package)
    return {}

def process_dirs(args):
    if False:
        print('Hello World!')
    dirs = sorted(os.listdir(args.input_dir))
    resources = {}
    if args.lang:
        resources = json.load(open(os.path.join(args.output_dir, 'resources.json')))
        resources[args.lang] = {}
    for model_dir in dirs:
        print(f'Processing models in {model_dir}')
        models = sorted(os.listdir(os.path.join(args.input_dir, model_dir)))
        for model in tqdm(models):
            if not model.endswith('.pt'):
                continue
            (lang, package, processor) = split_model_name(model)
            if args.lang and lang != args.lang:
                continue
            input_path = os.path.join(args.input_dir, model_dir, model)
            output_path = os.path.join(args.output_dir, lang, 'models', processor, package + '.pt')
            copy_file(input_path, output_path)
            md5 = get_md5(output_path)
            dependencies = get_dependencies(processor, lang, package)
            if lang not in resources:
                resources[lang] = {}
            if processor not in resources[lang]:
                resources[lang][processor] = {}
            if dependencies:
                resources[lang][processor][package] = {'md5': md5, 'dependencies': dependencies}
            else:
                resources[lang][processor][package] = {'md5': md5}
    print('Processed initial model directories.  Writing preliminary resources.json')
    json.dump(resources, open(os.path.join(args.output_dir, 'resources.json'), 'w'), indent=2)

def get_default_pos_package(lang, ud_package):
    if False:
        for i in range(10):
            print('nop')
    charlm_package = get_pos_charlm_package(lang, ud_package)
    if charlm_package is not None:
        return ud_package + '_charlm'
    if lang in no_pretrain_languages:
        return ud_package + '_nopretrain'
    return ud_package + '_nocharlm'

def get_default_depparse_package(lang, ud_package):
    if False:
        return 10
    charlm_package = get_depparse_charlm_package(lang, ud_package)
    if charlm_package is not None:
        return ud_package + '_charlm'
    if lang in no_pretrain_languages:
        return ud_package + '_nopretrain'
    return ud_package + '_nocharlm'

def process_default_zips(args):
    if False:
        while True:
            i = 10
    resources = json.load(open(os.path.join(args.output_dir, 'resources.json')))
    for lang in resources:
        if lang == 'url':
            continue
        if 'alias' in resources[lang]:
            continue
        if all((k in ('backward_charlm', 'forward_charlm', 'pretrain', 'lang_name') for k in resources[lang].keys())):
            continue
        if lang not in default_treebanks:
            raise AssertionError(f'{lang} not in default treebanks!!!')
        if args.lang and lang != args.lang:
            continue
        print(f'Preparing default models for language {lang}')
        models_needed = defaultdict(set)
        packages = resources[lang][PACKAGES]['default']
        for (processor, package) in packages.items():
            if processor == 'lemma' and package == 'identity':
                continue
            models_needed[processor].add(package)
            dependencies = get_dependencies(processor, lang, package)
            for dependency in dependencies:
                models_needed[dependency['model']].add(dependency['package'])
        model_files = []
        for processor in PROCESSORS:
            if processor in models_needed:
                for package in sorted(models_needed[processor]):
                    filename = os.path.join(args.output_dir, lang, 'models', processor, package + '.pt')
                    if os.path.exists(filename):
                        print('   Model {} package {}: file {}'.format(processor, package, filename))
                        model_files.append((filename, processor, package))
                    else:
                        raise FileNotFoundError(f'Processor {processor} package {package} needed for {lang} but cannot be found at {filename}')
        with zipfile.ZipFile(os.path.join(args.output_dir, lang, 'models', 'default.zip'), 'w', zipfile.ZIP_DEFLATED) as zipf:
            for (filename, processor, package) in model_files:
                zipf.write(filename=filename, arcname=os.path.join(processor, package + '.pt'))
        default_md5 = get_md5(os.path.join(args.output_dir, lang, 'models', 'default.zip'))
        resources[lang]['default_md5'] = default_md5
    print('Processed default model zips.  Writing resources.json')
    json.dump(resources, open(os.path.join(args.output_dir, 'resources.json'), 'w'), indent=2)

def get_default_processors(resources, lang):
    if False:
        print('Hello World!')
    '\n    Build a default package for this language\n\n    Will add each of pos, lemma, depparse, etc if those are available\n    Uses the existing models scraped from the language directories into resources.json, as relevant\n    '
    if lang == 'multilingual':
        return {'langid': 'ud'}
    default_package = default_treebanks[lang]
    default_processors = {}
    if lang in default_tokenizer:
        default_processors['tokenize'] = default_tokenizer[lang]
    else:
        default_processors['tokenize'] = default_package
    if 'mwt' in resources[lang] and default_processors['tokenize'] in resources[lang]['mwt']:
        default_processors['mwt'] = default_package
    if 'lemma' in resources[lang]:
        expected_lemma = default_package + '_nocharlm'
        if expected_lemma in resources[lang]['lemma']:
            default_processors['lemma'] = expected_lemma
    elif lang not in allowed_empty_languages:
        default_processors['lemma'] = 'identity'
    if 'pos' in resources[lang]:
        default_processors['pos'] = get_default_pos_package(lang, default_package)
        if default_processors['pos'] not in resources[lang]['pos']:
            raise AssertionError('Expected POS model not in resources: %s' % default_processors['pos'])
    elif lang not in allowed_empty_languages:
        raise AssertionError('Expected to find POS models for language %s' % lang)
    if 'depparse' in resources[lang]:
        default_processors['depparse'] = get_default_depparse_package(lang, default_package)
        if default_processors['depparse'] not in resources[lang]['depparse']:
            raise AssertionError('Expected depparse model not in resources: %s' % default_processors['depparse'])
    elif lang not in allowed_empty_languages:
        raise AssertionError('Expected to find depparse models for language %s' % lang)
    if lang in default_ners:
        default_processors['ner'] = default_ners[lang]
    if lang in default_sentiment:
        default_processors['sentiment'] = default_sentiment[lang]
    if lang in default_constituency:
        default_processors['constituency'] = default_constituency[lang]
    return default_processors

def get_default_accurate(resources, lang):
    if False:
        while True:
            i = 10
    '\n    A package that, if available, uses charlm and transformer models for each processor\n    '
    default_processors = get_default_processors(resources, lang)
    if 'lemma' in default_processors and default_processors['lemma'] != 'identity':
        lemma_model = default_processors['lemma']
        lemma_model = lemma_model.replace('_nocharlm', '_charlm')
        charlm_package = get_lemma_charlm_package(lang, lemma_model)
        if charlm_package is not None:
            if lemma_model in resources[lang]['lemma']:
                default_processors['lemma'] = lemma_model
            else:
                print('WARNING: wanted to use %s for %s default_accurate lemma, but that model does not exist' % (lemma_model, lang))
    transformer = TRANSFORMER_NICKNAMES.get(TRANSFORMERS.get(lang, None), None)
    if transformer is not None:
        for processor in ('pos', 'depparse', 'constituency'):
            if processor not in default_processors:
                continue
            new_model = default_processors[processor].replace('_charlm', '_' + transformer).replace('_nocharlm', '_' + transformer)
            if new_model in resources[lang][processor]:
                default_processors[processor] = new_model
            else:
                print('WARNING: wanted to use %s for %s default_accurate %s, but that model does not exist' % (new_model, lang, processor))
    return default_processors

def get_default_fast(resources, lang):
    if False:
        return 10
    "\n    Build a packages entry which only has the nocharlm models\n\n    Will make it easy for people to use the lower tier of models\n\n    We do this by building the same default package as normal,\n    then switching everything out for the lower tier model when possible.\n    We also remove constituency, as it is super slow.\n    Note that in the case of a language which doesn't have a charlm,\n    that means we wind up building the same for default and default_nocharlm\n    "
    default_processors = get_default_processors(resources, lang)
    if 'constituency' in default_processors:
        default_processors.pop('constituency')
    for (processor, model) in default_processors.items():
        if '_charlm' in model:
            nocharlm = model.replace('_charlm', '_nocharlm')
            if nocharlm not in resources[lang][processor]:
                print('WARNING: wanted to use %s for %s default_fast processor %s, but that model does not exist' % (nocharlm, lang, processor))
            else:
                default_processors[processor] = nocharlm
    return default_processors

def process_packages(args):
    if False:
        while True:
            i = 10
    "\n    Build a package for a language's default processors and all of the treebanks specifically used for that language\n    "
    resources = json.load(open(os.path.join(args.output_dir, 'resources.json')))
    for lang in resources:
        if lang == 'url':
            continue
        if 'alias' in resources[lang]:
            continue
        if all((k in ('backward_charlm', 'forward_charlm', 'pretrain', 'lang_name') for k in resources[lang].keys())):
            continue
        if lang not in default_treebanks:
            raise AssertionError(f'{lang} not in default treebanks!!!')
        if args.lang and lang != args.lang:
            continue
        default_processors = get_default_processors(resources, lang)
        resources[lang]['default_processors'] = default_processors
        resources[lang][PACKAGES] = {}
        resources[lang][PACKAGES]['default'] = default_processors
        if lang not in no_pretrain_languages and lang != 'multilingual':
            default_fast = get_default_fast(resources, lang)
            resources[lang][PACKAGES]['default_fast'] = default_fast
            default_accurate = get_default_accurate(resources, lang)
            resources[lang][PACKAGES]['default_accurate'] = default_accurate
        if 'tokenize' in resources[lang]:
            for package in resources[lang]['tokenize']:
                processors = {'tokenize': package}
                if 'mwt' in resources[lang] and package in resources[lang]['mwt']:
                    processors['mwt'] = package
                if 'pos' in resources[lang]:
                    if package + '_charlm' in resources[lang]['pos']:
                        processors['pos'] = package + '_charlm'
                    elif package + '_nocharlm' in resources[lang]['pos']:
                        processors['pos'] = package + '_nocharlm'
                if 'lemma' in resources[lang] and 'pos' in processors:
                    lemma_package = package + '_nocharlm'
                    if lemma_package in resources[lang]['lemma']:
                        processors['lemma'] = lemma_package
                if 'depparse' in resources[lang] and 'pos' in processors:
                    depparse_package = None
                    if package + '_charlm' in resources[lang]['depparse']:
                        depparse_package = package + '_charlm'
                    elif package + '_nocharlm' in resources[lang]['depparse']:
                        depparse_package = package + '_nocharlm'
                    if depparse_package is not None:
                        if 'lemma' not in processors:
                            processors['lemma'] = 'identity'
                        processors['depparse'] = depparse_package
                resources[lang][PACKAGES][package] = processors
    print('Processed packages.  Writing resources.json')
    json.dump(resources, open(os.path.join(args.output_dir, 'resources.json'), 'w'), indent=2)

def process_lcode(args):
    if False:
        i = 10
        return i + 15
    resources = json.load(open(os.path.join(args.output_dir, 'resources.json')))
    resources_new = {}
    resources_new['multilingual'] = resources['multilingual']
    for lang in resources:
        if lang == 'multilingual':
            continue
        if 'alias' in resources[lang]:
            continue
        if lang not in lcode2lang:
            print(lang + ' not found in lcode2lang!')
            continue
        lang_name = lcode2lang[lang]
        resources[lang]['lang_name'] = lang_name
        resources_new[lang.lower()] = resources[lang.lower()]
        resources_new[lang_name.lower()] = {'alias': lang.lower()}
        if lang.lower() in two_to_three_letters:
            resources_new[two_to_three_letters[lang.lower()]] = {'alias': lang.lower()}
        elif lang.lower() in three_to_two_letters:
            resources_new[three_to_two_letters[lang.lower()]] = {'alias': lang.lower()}
    print('Processed lcode aliases.  Writing resources.json')
    json.dump(resources_new, open(os.path.join(args.output_dir, 'resources.json'), 'w'), indent=2)

def process_misc(args):
    if False:
        print('Hello World!')
    resources = json.load(open(os.path.join(args.output_dir, 'resources.json')))
    resources['no'] = {'alias': 'nb'}
    resources['zh'] = {'alias': 'zh-hans'}
    resources['url'] = 'https://huggingface.co/stanfordnlp/stanza-{lang}/resolve/v{resources_version}/models/{filename}'
    print('Finalized misc attributes.  Writing resources.json')
    json.dump(resources, open(os.path.join(args.output_dir, 'resources.json'), 'w'), indent=2)

def main():
    if False:
        i = 10
        return i + 15
    args = parse_args()
    print('Converting models from %s to %s' % (args.input_dir, args.output_dir))
    if not args.packages_only:
        process_dirs(args)
    process_packages(args)
    if not args.packages_only:
        process_default_zips(args)
        process_lcode(args)
        process_misc(args)
if __name__ == '__main__':
    main()