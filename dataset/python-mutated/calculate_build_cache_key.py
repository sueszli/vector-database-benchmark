from hashlib import sha256
from pathlib import Path

def hash_file(filename):
    if False:
        for i in range(10):
            print('nop')
    with open(filename, 'rb') as f:
        return sha256(f.read()).hexdigest()

def get_ignore_pattern(root: Path) -> list[str]:
    if False:
        while True:
            i = 10
    ignorefile = root / '.gitignore'
    if not ignorefile.exists():
        raise RuntimeError('No .gitignore file found in root directory')
    return ignorefile.read_text().splitlines()

def main():
    if False:
        print('Hello World!')
    import pathspec
    root: Path = Path(__file__).parent.parent
    targets: list[Path] = [root / 'Makefile', root / 'Makefile.envs', root / 'packages' / 'Makefile', root / 'pyodide-build' / 'setup.cfg', root / 'pyodide-build' / 'pyodide_build' / '__init__.py', root / 'pyodide-build' / 'pyodide_build' / 'pywasmcross.py', root / 'tools']
    ignore_pattern = get_ignore_pattern(root)
    ignore_spec = pathspec.PathSpec.from_lines('gitwildmatch', ignore_pattern)
    hash_candidates: list[Path] = []
    for target in targets:
        if target.is_file():
            hash_candidates.append(target)
        else:
            hash_candidates.extend(list(target.glob('**/*')))
    hash_candidates_filtered = sorted(list(filter(lambda file: file.is_file() and (not ignore_spec.match_file(str(file))), hash_candidates)))
    hashes = []
    for file in hash_candidates_filtered:
        hashes.append(hash_file(file))
    print(''.join(hashes))
if __name__ == '__main__':
    main()