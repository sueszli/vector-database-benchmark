import os
from pathlib import Path
import unasync

def main():
    if False:
        i = 10
        return i + 15
    additional_replacements = {'AsyncTransport': 'Transport', 'AsyncElasticsearch': 'Elasticsearch', 'AsyncSearchClient': 'AsyncSearchClient', '_TYPE_ASYNC_SNIFF_CALLBACK': '_TYPE_SYNC_SNIFF_CALLBACK'}
    rules = [unasync.Rule(fromdir='/elasticsearch/_async/client/', todir='/elasticsearch/_sync/client/', additional_replacements=additional_replacements)]
    filepaths = []
    for (root, _, filenames) in os.walk(Path(__file__).absolute().parent.parent / 'elasticsearch/_async'):
        for filename in filenames:
            if filename.rpartition('.')[-1] in ('py', 'pyi') and (not filename.startswith('utils.py')):
                filepaths.append(os.path.join(root, filename))
    unasync.unasync_files(filepaths, rules)
if __name__ == '__main__':
    main()