"""Tests for SearchIO blast-tab indexing."""
import unittest
from search_tests_common import CheckRaw, CheckIndex

class BlastTabRawCases(CheckRaw):
    """Check BLAST tabular get_raw method."""
    fmt = 'blast-tab'

    def test_blasttab_2226_multiple_first(self):
        if False:
            return 10
        'Test blast-tab raw string retrieval, BLAST 2.2.26+, multiple queries, first (tab_2226_tblastn_001.txt).'
        filename = 'Blast/tab_2226_tblastn_001.txt'
        raw = 'gi|16080617|ref|NP_391444.1|\tgi|145479850|ref|XM_001425911.1|\t34.88\t43\t28\t0\t31\t73\t1744\t1872\t1e-05\t34.7\ngi|16080617|ref|NP_391444.1|\tgi|72012412|ref|XM_777959.1|\t33.90\t59\t31\t1\t44\t94\t1057\t1233\t1e-04\t31.6\ngi|16080617|ref|NP_391444.1|\tgi|115975252|ref|XM_001180111.1|\t33.90\t59\t31\t1\t44\t94\t1057\t1233\t1e-04\t31.6\n'
        self.check_raw(filename, 'gi|16080617|ref|NP_391444.1|', raw)

    def test_blasttab_2226_multiple_last(self):
        if False:
            while True:
                i = 10
        'Test blast-tab raw string retrieval, BLAST 2.2.26+, multiple queries, last (tab_2226_tblastn_001.txt).'
        filename = 'Blast/tab_2226_tblastn_001.txt'
        raw = 'gi|11464971:4-101\tgi|350596019|ref|XM_003360601.2|\t95.92\t98\t4\t0\t1\t98\t95\t388\t2e-67\t 199\ngi|11464971:4-101\tgi|350596019|ref|XM_003360601.2|\t29.58\t71\t46\t2\t30\t96\t542\t754\t4e-05\t32.7\ngi|11464971:4-101\tgi|301779869|ref|XM_002925302.1|\t97.96\t98\t2\t0\t1\t98\t78\t371\t2e-67\t 202\ngi|11464971:4-101\tgi|301779869|ref|XM_002925302.1|\t30.00\t100\t64\t2\t3\t96\t804\t1103\t3e-09\t45.1\ngi|11464971:4-101\tgi|296223671|ref|XM_002757683.1|\t97.96\t98\t2\t0\t1\t98\t161\t454\t4e-67\t 202\ngi|11464971:4-101\tgi|296223671|ref|XM_002757683.1|\t30.00\t100\t64\t2\t3\t96\t866\t1165\t3e-09\t45.1\ngi|11464971:4-101\tgi|338714227|ref|XM_001492113.3|\t97.96\t98\t2\t0\t1\t98\t173\t466\t2e-66\t 202\ngi|11464971:4-101\tgi|338714227|ref|XM_001492113.3|\t31.00\t100\t63\t2\t3\t96\t899\t1198\t1e-09\t46.6\ngi|11464971:4-101\tgi|365982352|ref|XM_003667962.1|\t30.77\t52\t27\t1\t12\t54\t3181\t3336\t1.7\t19.6\n'
        self.check_raw(filename, 'gi|11464971:4-101', raw)

    def test_blasttab_2226_single(self):
        if False:
            while True:
                i = 10
        'Test blast-tab raw string retrieval, BLAST 2.2.26+, single query (tab_2226_tblastn_004.txt).'
        filename = 'Blast/tab_2226_tblastn_004.txt'
        raw = 'gi|11464971:4-101\tgi|350596019|ref|XM_003360601.2|\t95.92\t98\t4\t0\t1\t98\t95\t388\t2e-67\t 199\ngi|11464971:4-101\tgi|350596019|ref|XM_003360601.2|\t29.58\t71\t46\t2\t30\t96\t542\t754\t4e-05\t32.7\ngi|11464971:4-101\tgi|301779869|ref|XM_002925302.1|\t97.96\t98\t2\t0\t1\t98\t78\t371\t2e-67\t 202\ngi|11464971:4-101\tgi|301779869|ref|XM_002925302.1|\t30.00\t100\t64\t2\t3\t96\t804\t1103\t3e-09\t45.1\ngi|11464971:4-101\tgi|296223671|ref|XM_002757683.1|\t97.96\t98\t2\t0\t1\t98\t161\t454\t4e-67\t 202\ngi|11464971:4-101\tgi|296223671|ref|XM_002757683.1|\t30.00\t100\t64\t2\t3\t96\t866\t1165\t3e-09\t45.1\ngi|11464971:4-101\tgi|338714227|ref|XM_001492113.3|\t97.96\t98\t2\t0\t1\t98\t173\t466\t2e-66\t 202\ngi|11464971:4-101\tgi|338714227|ref|XM_001492113.3|\t31.00\t100\t63\t2\t3\t96\t899\t1198\t1e-09\t46.6\ngi|11464971:4-101\tgi|365982352|ref|XM_003667962.1|\t30.77\t52\t27\t1\t12\t54\t3181\t3336\t1.7\t19.6\n'
        self.check_raw(filename, 'gi|11464971:4-101', raw)

    def test_blasttab_2226_multiple_first_commented(self):
        if False:
            return 10
        'Test blast-tab raw string retrieval, BLAST 2.2.26+, multiple queries, first, commented (tab_2226_tblastn_005.txt).'
        filename = 'Blast/tab_2226_tblastn_005.txt'
        raw = '# TBLASTN 2.2.26+\n# Query: random_s00\n# Database: db/minirefseq_mrna\n# 0 hits found\n'
        self.check_raw(filename, 'random_s00', raw, comments=True)

    def test_blasttab_2226_multiple_middle_commented(self):
        if False:
            for i in range(10):
                print('nop')
        'Test blast-tab raw string retrieval, BLAST 2.2.26+, multiple queries, middle, commented (tab_2226_tblastn_005.txt).'
        filename = 'Blast/tab_2226_tblastn_005.txt'
        raw = '# TBLASTN 2.2.26+\n# Query: gi|16080617|ref|NP_391444.1| membrane bound lipoprotein [Bacillus subtilis subsp. subtilis str. 168]\n# Database: db/minirefseq_mrna\n# Fields: query id, subject id, % identity, alignment length, mismatches, gap opens, q. start, q. end, s. start, s. end, evalue, bit score\n# 3 hits found\ngi|16080617|ref|NP_391444.1|\tgi|145479850|ref|XM_001425911.1|\t34.88\t43\t28\t0\t31\t73\t1744\t1872\t1e-05\t34.7\ngi|16080617|ref|NP_391444.1|\tgi|72012412|ref|XM_777959.1|\t33.90\t59\t31\t1\t44\t94\t1057\t1233\t1e-04\t31.6\ngi|16080617|ref|NP_391444.1|\tgi|115975252|ref|XM_001180111.1|\t33.90\t59\t31\t1\t44\t94\t1057\t1233\t1e-04\t31.6\n'
        self.check_raw(filename, 'gi|16080617|ref|NP_391444.1|', raw, comments=True)

    def test_blasttab_2226_multiple_last_commented(self):
        if False:
            return 10
        'Test blast-tab raw string retrieval, BLAST 2.2.26+, multiple queries, last, commented (tab_2226_tblastn_005.txt).'
        filename = 'Blast/tab_2226_tblastn_005.txt'
        raw = '# TBLASTN 2.2.26+\n# Query: gi|11464971:4-101 pleckstrin [Mus musculus]\n# Database: db/minirefseq_mrna\n# Fields: query id, subject id, % identity, alignment length, mismatches, gap opens, q. start, q. end, s. start, s. end, evalue, bit score\n# 9 hits found\ngi|11464971:4-101\tgi|350596019|ref|XM_003360601.2|\t95.92\t98\t4\t0\t1\t98\t95\t388\t2e-67\t 199\ngi|11464971:4-101\tgi|350596019|ref|XM_003360601.2|\t29.58\t71\t46\t2\t30\t96\t542\t754\t4e-05\t32.7\ngi|11464971:4-101\tgi|301779869|ref|XM_002925302.1|\t97.96\t98\t2\t0\t1\t98\t78\t371\t2e-67\t 202\ngi|11464971:4-101\tgi|301779869|ref|XM_002925302.1|\t30.00\t100\t64\t2\t3\t96\t804\t1103\t3e-09\t45.1\ngi|11464971:4-101\tgi|296223671|ref|XM_002757683.1|\t97.96\t98\t2\t0\t1\t98\t161\t454\t4e-67\t 202\ngi|11464971:4-101\tgi|296223671|ref|XM_002757683.1|\t30.00\t100\t64\t2\t3\t96\t866\t1165\t3e-09\t45.1\ngi|11464971:4-101\tgi|338714227|ref|XM_001492113.3|\t97.96\t98\t2\t0\t1\t98\t173\t466\t2e-66\t 202\ngi|11464971:4-101\tgi|338714227|ref|XM_001492113.3|\t31.00\t100\t63\t2\t3\t96\t899\t1198\t1e-09\t46.6\ngi|11464971:4-101\tgi|365982352|ref|XM_003667962.1|\t30.77\t52\t27\t1\t12\t54\t3181\t3336\t1.7\t19.6\n'
        self.check_raw(filename, 'gi|11464971:4-101', raw, comments=True)

    def test_blasttab_2226_single_commented(self):
        if False:
            print('Hello World!')
        'Test blast-tab raw string retrieval, BLAST 2.2.26+, single query, commented (tab_2226_tblastn_008.txt).'
        filename = 'Blast/tab_2226_tblastn_008.txt'
        raw = '# TBLASTN 2.2.26+\n# Query: gi|11464971:4-101 pleckstrin [Mus musculus]\n# Database: db/minirefseq_mrna\n# Fields: query id, subject id, % identity, alignment length, mismatches, gap opens, q. start, q. end, s. start, s. end, evalue, bit score\n# 9 hits found\ngi|11464971:4-101\tgi|350596019|ref|XM_003360601.2|\t95.92\t98\t4\t0\t1\t98\t95\t388\t2e-67\t 199\ngi|11464971:4-101\tgi|350596019|ref|XM_003360601.2|\t29.58\t71\t46\t2\t30\t96\t542\t754\t4e-05\t32.7\ngi|11464971:4-101\tgi|301779869|ref|XM_002925302.1|\t97.96\t98\t2\t0\t1\t98\t78\t371\t2e-67\t 202\ngi|11464971:4-101\tgi|301779869|ref|XM_002925302.1|\t30.00\t100\t64\t2\t3\t96\t804\t1103\t3e-09\t45.1\ngi|11464971:4-101\tgi|296223671|ref|XM_002757683.1|\t97.96\t98\t2\t0\t1\t98\t161\t454\t4e-67\t 202\ngi|11464971:4-101\tgi|296223671|ref|XM_002757683.1|\t30.00\t100\t64\t2\t3\t96\t866\t1165\t3e-09\t45.1\ngi|11464971:4-101\tgi|338714227|ref|XM_001492113.3|\t97.96\t98\t2\t0\t1\t98\t173\t466\t2e-66\t 202\ngi|11464971:4-101\tgi|338714227|ref|XM_001492113.3|\t31.00\t100\t63\t2\t3\t96\t899\t1198\t1e-09\t46.6\ngi|11464971:4-101\tgi|365982352|ref|XM_003667962.1|\t30.77\t52\t27\t1\t12\t54\t3181\t3336\t1.7\t19.6\n'
        self.check_raw(filename, 'gi|11464971:4-101', raw, comments=True)

class BlastTabIndexCases(CheckIndex):
    fmt = 'blast-tab'

    def test_blasttab_2226_tblastn_001(self):
        if False:
            for i in range(10):
                print('nop')
        'Test blast-tab indexing, BLAST 2.2.26+, multiple queries.'
        filename = 'Blast/tab_2226_tblastn_001.txt'
        self.check_index(filename, self.fmt)

    def test_blasttab_2226_tblastn_002(self):
        if False:
            return 10
        'Test blast-tab indexing, BLAST 2.2.26+, single query, no hits.'
        filename = 'Blast/tab_2226_tblastn_002.txt'
        self.check_index(filename, self.fmt)

    def test_blasttab_2226_tblastn_004(self):
        if False:
            print('Hello World!')
        'Test blast-tab indexing, BLAST 2.2.26+, single query, multiple hits.'
        filename = 'Blast/tab_2226_tblastn_004.txt'
        self.check_index(filename, self.fmt)

    def test_blasttab_2226_tblastn_005(self):
        if False:
            while True:
                i = 10
        'Test blast-tab indexing, BLAST 2.2.26+, multiple queries, commented.'
        filename = 'Blast/tab_2226_tblastn_005.txt'
        self.check_index(filename, self.fmt, comments=True)

    def test_blasttab_2226_tblastn_006(self):
        if False:
            while True:
                i = 10
        'Test blast-tab indexing, BLAST 2.2.26+, single query, no hits, commented.'
        filename = 'Blast/tab_2226_tblastn_006.txt'
        self.check_index(filename, self.fmt, comments=True)

    def test_blasttab_comment_sing(self):
        if False:
            while True:
                i = 10
        'Test blast-tab indexing, BLAST 2.2.26+, single query, multiple hits, commented.'
        filename = 'Blast/tab_2226_tblastn_008.txt'
        self.check_index(filename, self.fmt, comments=True)

    def test_blasttab_2226_tblastn_009(self):
        if False:
            return 10
        'Test blast-tab indexing, BLAST 2.2.26+, custom columns.'
        filename = 'Blast/tab_2226_tblastn_009.txt'
        self.check_index(filename, self.fmt, fields=['qseqid', 'sseqid'])

    def test_blasttab_2226_tblastn_010(self):
        if False:
            i = 10
            return i + 15
        'Test blast-tab indexing, BLAST 2.2.26+, custom columns, commented.'
        filename = 'Blast/tab_2226_tblastn_010.txt'
        self.check_index(filename, self.fmt, comments=True)

    def test_blasttab_2226_tblastn_011(self):
        if False:
            print('Hello World!')
        'Test blast-tab indexing, BLAST 2.2.26+, all columns, commented.'
        filename = 'Blast/tab_2226_tblastn_011.txt'
        self.check_index(filename, self.fmt, comments=True)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)