import pytest
from spacy.matcher import levenshtein
from spacy.matcher.levenshtein import levenshtein_compare

@pytest.mark.parametrize('dist,a,b', [(0, '', ''), (4, 'bbcb', 'caba'), (3, 'abcb', 'cacc'), (3, 'aa', 'ccc'), (1, 'cca', 'ccac'), (1, 'aba', 'aa'), (4, 'bcbb', 'abac'), (3, 'acbc', 'bba'), (3, 'cbba', 'a'), (2, 'bcc', 'ba'), (4, 'aaa', 'ccbb'), (3, 'うあい', 'いいうい'), (2, 'あううい', 'うあい'), (3, 'いういい', 'うううあ'), (2, 'うい', 'あいあ'), (2, 'いあい', 'いう'), (1, 'いい', 'あいい'), (3, 'あうあ', 'いいああ'), (4, 'いあうう', 'ううああ'), (3, 'いあいい', 'ういああ'), (3, 'いいああ', 'ううあう'), (166, 'TCTGGGCACGGATTCGTCAGATTCCATGTCCATATTTGAGGCTCTTGCAGGCAAAATTTGGGCATGTGAACTCCTTATAGTCCCCGTGC', 'ATATGGATTGGGGGCATTCAAAGATACGGTTTCCCTTTCTTCAGTTTCGCGCGGCGCACGTCCGGGTGCGAGCCAGTTCGTCTTACTCACATTGTCGACTTCACGAATCGCGCATGATGTGCTTAGCCTGTACTTACGAACGAACTTTCGGTCCAAATACATTCTATCAACACCGAGGTATCCGTGCCACACGCCGAAGCTCGACCGTGTTCGTTGAGAGGTGGAAATGGTAAAAGATGAACATAGTC'), (111, 'GGTTCGGCCGAATTCATAGAGCGTGGTAGTCGACGGTATCCCGCCTGGTAGGGGCCCCTTCTACCTAGCGGAAGTTTGTCAGTACTCTATAACACGAGGGCCTCTCACACCCTAGATCGTCCAGCCACTCGAAGATCGCAGCACCCTTACAGAAAGGCATTAATGTTTCTCCTAGCACTTGTGCAATGGTGAAGGAGTGATG', 'CGTAACACTTCGCGCTACTGGGCTGCAACGTCTTGGGCATACATGCAAGATTATCTAATGCAAGCTTGAGCCCCGCTTGCGGAATTTCCCTAATCGGGGTCCCTTCCTGTTACGATAAGGACGCGTGCACT')])
def test_levenshtein(dist, a, b):
    if False:
        for i in range(10):
            print('nop')
    assert levenshtein(a, b) == dist

@pytest.mark.parametrize('a,b,fuzzy,expected', [('a', 'a', 1, True), ('a', 'a', 0, True), ('a', 'a', -1, True), ('a', 'ab', 1, True), ('a', 'ab', 0, False), ('a', 'ab', -1, True), ('ab', 'ac', 1, True), ('ab', 'ac', -1, True), ('abc', 'cde', 4, True), ('abc', 'cde', -1, False), ('abcdef', 'cdefgh', 4, True), ('abcdef', 'cdefgh', 3, False), ('abcdef', 'cdefgh', -1, False), ('abcdefgh', 'cdefghijk', 5, True), ('abcdefgh', 'cdefghijk', 4, False), ('abcdefgh', 'cdefghijk', -1, False), ('abcdefgh', 'cdefghijkl', 6, True), ('abcdefgh', 'cdefghijkl', 5, False), ('abcdefgh', 'cdefghijkl', -1, False)])
def test_levenshtein_compare(a, b, fuzzy, expected):
    if False:
        print('Hello World!')
    assert levenshtein_compare(a, b, fuzzy) == expected