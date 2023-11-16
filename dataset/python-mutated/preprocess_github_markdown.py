import argparse
import pathlib
import re
from typing import Optional

def preprocess_github_markdown_file(source_path: str, dest_path: Optional[str]=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Preprocesses GitHub Markdown files by:\n        - Uncommenting all ``<!-- -->`` comments in which opening tag is immediately\n          succeeded by ``$UNCOMMENT``(eg. ``<!--$UNCOMMENTthis will be uncommented-->``)\n        - Removing text between ``<!--$REMOVE-->`` and ``<!--$END_REMOVE-->``\n\n    This is to enable translation between GitHub Markdown and MyST Markdown used\n    in docs. For more details, see ``doc/README.md``.\n\n    Args:\n        source_path: The path to the locally saved markdown file to preprocess.\n        dest_path: The destination path to save the preprocessed markdown file.\n            If not provided, save to the same location as source_path.\n    '
    dest_path = dest_path if dest_path else source_path
    with open(source_path, 'r') as f:
        text = f.read()
    text = re.sub('<!--\\s*\\$UNCOMMENT(.*?)(-->)', '\\1', text, flags=re.DOTALL)
    text = re.sub('(<!--\\s*\\$REMOVE\\s*-->)(.*?)(<!--\\s*\\$END_REMOVE\\s*-->)', '', text, flags=re.DOTALL)
    with open(dest_path, 'w') as f:
        f.write(text)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess github markdown file to Ray Docs MyST markdown')
    parser.add_argument('source_path', type=pathlib.Path, help='Path to github markdown file to preprocess')
    parser.add_argument('dest_path', type=pathlib.Path, help='Path to save preprocessed markdown file.')
    (args, _) = parser.parse_known_args()
    preprocess_github_markdown_file(args.source_path.expanduser(), args.dest_path.expanduser())