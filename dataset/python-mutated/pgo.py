PGO_TESTS = ['test_array', 'test_base64', 'test_binascii', 'test_binop', 'test_bisect', 'test_bytes', 'test_bz2', 'test_cmath', 'test_codecs', 'test_collections', 'test_complex', 'test_dataclasses', 'test_datetime', 'test_decimal', 'test_difflib', 'test_embed', 'test_float', 'test_fstring', 'test_functools', 'test_generators', 'test_hashlib', 'test_heapq', 'test_int', 'test_itertools', 'test_json', 'test_long', 'test_lzma', 'test_math', 'test_memoryview', 'test_operator', 'test_ordered_dict', 'test_patma', 'test_pickle', 'test_pprint', 'test_re', 'test_set', 'test_sqlite', 'test_statistics', 'test_struct', 'test_tabnanny', 'test_time', 'test_unicode', 'test_xml_etree', 'test_xml_etree_c']

def setup_pgo_tests(ns):
    if False:
        return 10
    if not ns.args and (not ns.pgo_extended):
        ns.args = PGO_TESTS[:]