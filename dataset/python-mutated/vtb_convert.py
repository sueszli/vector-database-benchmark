"""
Script for processing the VTB files and turning their trees into the desired tree syntax

The VTB original trees are stored in the directory:
VietTreebank_VLSP_SP73/Kho ngu lieu 10000 cay cu phap
The script requires two arguments:
1. Original directory storing the original trees
2. New directory storing the converted trees
"""
import argparse
import os
from collections import defaultdict
from stanza.models.constituency.tree_reader import read_trees, MixedTreeError, UnlabeledTreeError
REMAPPING = {'(ADV-MDP': '(RP-MDP', '(MPD': '(MDP', '(MP ': '(NP ', '(MP(': '(NP(', '(Np(': '(NP(', '(Np (': '(NP (', '(NLOC': '(NP-LOC', '(N-P-LOC': '(NP-LOC', '(N-p-loc': '(NP-LOC', '(NPDOB': '(NP-DOB', '(NPSUB': '(NP-SUB', '(NPTMP': '(NP-TMP', '(PPLOC': '(PP-LOC', '(SBA ': '(SBAR ', '(SBA-': '(SBAR-', '(SBA(': '(SBAR(', '(SBAS': '(SBAR', '(SABR': '(SBAR', '(SE-SPL': '(S-SPL', '(SBARR': '(SBAR', 'PPADV': 'PP-ADV', '(PR (': '(PP (', '(PPP': '(PP', 'VP0ADV': 'VP-ADV', '(S1': '(S', '(S2': '(S', '(S3': '(S', 'BP-SUB': 'NP-SUB', 'APPPD': 'AP-PPD', 'APPRD': 'AP-PPD', 'Np--H': 'Np-H', '(WPNP': '(WHNP', '(WHRPP': '(WHRP', '(PV': '(PP', '(Mpd': '(MDP', '(Whadv ': '(WHRP ', '(Whpr (Pro-h nào))': '(WHPP (Pro-h nào))', '(Whpr (Pro-h Sao))': '(WHRP (Pro-h Sao))', '(Tp-tmp': '(NP-TMP', '(Yp': '(NP'}

def unify_label(tree):
    if False:
        return 10
    for (old, new) in REMAPPING.items():
        tree = tree.replace(old, new)
    return tree

def count_paren_parity(tree):
    if False:
        return 10
    '\n    Checks if the tree is properly closed\n    :param tree: tree as a string\n    :return: True if closed otherwise False\n    '
    count = 0
    for char in tree:
        if char == '(':
            count += 1
        elif char == ')':
            count -= 1
    return count

def is_valid_line(line):
    if False:
        print('Hello World!')
    '\n    Check if a line being read is a valid constituent\n\n    The idea is that some "trees" are just a long list of words with\n    no tree structure and need to be eliminated.\n\n    :param line: constituent being read\n    :return: True if it has open OR closing parenthesis.\n    '
    if line.startswith('(') or line.endswith(')'):
        return True
    return False
WEIRD_LABELS = sorted(set(['WP', 'YP', 'SNP', 'STC', 'UPC', '(TP', 'Xp', 'XP', 'WHVP', 'WHPR', 'NO', 'WHADV', '(SC (', '(VOC (', '(Adv (', '(SP (', 'ADV-MDP', '(SPL', '(ADV (', '(V-MWE ('] + list(REMAPPING.keys())))

def convert_file(orig_file, new_file, fix_errors=True, convert_brackets=False):
    if False:
        while True:
            i = 10
    '\n    :param orig_file: original directory storing original trees\n    :param new_file: new directory storing formatted constituency trees\n    This function writes new trees to the corresponding files in new_file\n    '
    errors = defaultdict(list)
    with open(orig_file, 'r', encoding='utf-8') as reader, open(new_file, 'w', encoding='utf-8') as writer:
        content = reader.readlines()
        tree = ''
        reading_tree = False
        for (line_idx, line) in enumerate(content):
            line = ' '.join(line.split())
            if line == '':
                continue
            elif line == '<s>' or line.startswith('<s id='):
                tree = ''
                tree += '(ROOT '
                reading_tree = True
            elif line == '</s>' and reading_tree:
                if tree.strip() == '(ROOT':
                    errors['empty'].append('Empty tree in {} line {}'.format(orig_file, line_idx))
                    continue
                tree += ')\n'
                parity = count_paren_parity(tree)
                if parity > 0:
                    errors['unclosed'].append('Unclosed tree from {} line {}: |{}|'.format(orig_file, line_idx, tree))
                    continue
                if parity < 0:
                    errors['extra_parens'].append('Extra parens at end of tree from {} line {} for having extra parens: {}'.format(orig_file, line_idx, tree))
                    continue
                if convert_brackets:
                    tree = tree.replace('RBKT', '-RRB-').replace('LBKT', '-LRB-')
                try:
                    processed_trees = read_trees(tree)
                    if len(processed_trees) > 1:
                        errors['multiple'].append('Multiple trees in one xml annotation from {} line {}'.format(orig_file, line_idx))
                        continue
                    if len(processed_trees) == 0:
                        errors['empty'].append('Empty tree in {} line {}'.format(orig_file, line_idx))
                        continue
                    if not processed_trees[0].all_leaves_are_preterminals():
                        errors['untagged_leaf'].append('Tree with non-preterminal leaves in {} line {}: {}'.format(orig_file, line_idx, tree))
                        continue
                    if fix_errors:
                        tree = unify_label(tree)
                    bad_label = False
                    for weird_label in WEIRD_LABELS:
                        if tree.find(weird_label) >= 0:
                            bad_label = True
                            errors[weird_label].append('Weird label {} from {} line {}: {}'.format(weird_label, orig_file, line_idx, tree))
                            break
                    if bad_label:
                        continue
                    writer.write(tree)
                    reading_tree = False
                    tree = ''
                except MixedTreeError:
                    errors['mixed'].append('Mixed leaves and constituents from {} line {}: {}'.format(orig_file, line_idx, tree))
                except UnlabeledTreeError:
                    errors['unlabeled'].append('Unlabeled nodes in tree from {} line {}: {}'.format(orig_file, line_idx, tree))
            elif is_valid_line(line) and reading_tree:
                tree += line
            elif reading_tree:
                errors['invalid'].append('Invalid tree error in {} line {}: |{}|, rejected because of line |{}|'.format(orig_file, line_idx, tree, line))
                reading_tree = False
    return errors

def convert_files(file_list, new_dir, verbose=False, fix_errors=True, convert_brackets=False):
    if False:
        while True:
            i = 10
    errors = defaultdict(list)
    for filename in file_list:
        (base_name, _) = os.path.splitext(os.path.split(filename)[-1])
        new_path = os.path.join(new_dir, base_name)
        new_file_path = f'{new_path}.mrg'
        new_errors = convert_file(filename, new_file_path, fix_errors, convert_brackets)
        for e in new_errors:
            errors[e].extend(new_errors[e])
    if len(errors.keys()) == 0:
        print('All errors were fixed!')
    else:
        print('Found the following errors:')
        keys = sorted(errors.keys())
        if verbose:
            for e in keys:
                print('--------- %10s -------------' % e)
                print('\n\n'.join(errors[e]))
                print()
            print()
        for e in keys:
            print('%s: %d' % (e, len(errors[e])))

def convert_dir(orig_dir, new_dir):
    if False:
        i = 10
        return i + 15
    file_list = os.listdir(orig_dir)
    file_list = [os.path.join(orig_dir, f) for f in file_list if os.path.splitext(f)[1] != '.raw']
    convert_files(file_list, new_dir)

def main():
    if False:
        return 10
    '\n    Converts files from the 2009 version of VLSP to .mrg files\n    \n    Process args, loop through each file in the directory and convert\n    to the desired tree format\n    '
    parser = argparse.ArgumentParser(description='Script that converts a VTB Tree into the desired format')
    parser.add_argument('orig_dir', help='The location of the original directory storing original trees ')
    parser.add_argument('new_dir', help='The location of new directory storing the new formatted trees')
    args = parser.parse_args()
    org_dir = args.org_dir
    new_dir = args.new_dir
    convert_dir(org_dir, new_dir)
if __name__ == '__main__':
    main()