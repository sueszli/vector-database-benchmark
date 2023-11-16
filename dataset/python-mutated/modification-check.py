"""Ensures there have been no changes to important certbot-auto files."""
import hashlib
import os
EXPECTED_FILES = {os.path.join('letsencrypt-auto-source', 'letsencrypt-auto'): 'b997e3608526650a08e36e682fc3bf0c29903c06fa5ba4cc49308c43832450c2', os.path.join('letsencrypt-auto-source', 'letsencrypt-auto.sig'): '61c036aabf75da350b0633da1b2bef0260303921ecda993455ea5e6d3af3b2fe'}

def find_repo_root():
    if False:
        i = 10
        return i + 15
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def sha256_hash(filename):
    if False:
        print('Hello World!')
    hash_object = hashlib.sha256()
    with open(filename, 'rb') as f:
        hash_object.update(f.read())
    return hash_object.hexdigest()

def main():
    if False:
        return 10
    repo_root = find_repo_root()
    for (filename, expected_hash) in EXPECTED_FILES.items():
        filepath = os.path.join(repo_root, filename)
        assert sha256_hash(filepath) == expected_hash, f'unexpected changes to {filepath}'
    print('All certbot-auto files have correct hashes.')
if __name__ == '__main__':
    main()