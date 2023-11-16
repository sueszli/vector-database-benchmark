"""Unit test for Scop."""
import unittest
from io import StringIO
from Bio.SCOP import Scop, cmp_sccs, parse_domain

class ScopTests(unittest.TestCase):

    def _compare_cla_lines(self, cla_line_1, cla_line_2):
        if False:
            print('Hello World!')
        'Compare the two specified Cla lines for equality.\n\n        The order of the key-value pairs in the sixth field of the lines does\n        not matter. For more information, see\n        http://scop.mrc-lmb.cam.ac.uk/scop/release-notes.html.\n        '
        fields1 = cla_line_1.rstrip().split('\t')
        fields2 = cla_line_2.rstrip().split('\t')
        self.assertEqual(fields1[:5], fields2[:5])
        self.assertCountEqual(fields1[5].split(','), fields2[5].split(','))

    def testParse(self):
        if False:
            print('Hello World!')
        with open('./SCOP/dir.cla.scop.txt_test') as f:
            cla = f.read()
        with open('./SCOP/dir.des.scop.txt_test') as f:
            des = f.read()
        with open('./SCOP/dir.hie.scop.txt_test') as f:
            hie = f.read()
        scop = Scop(StringIO(cla), StringIO(des), StringIO(hie))
        cla_out = StringIO()
        scop.write_cla(cla_out)
        lines = zip(cla.rstrip().split('\n'), cla_out.getvalue().rstrip().split('\n'))
        for (expected_line, line) in lines:
            self._compare_cla_lines(expected_line, line)
        des_out = StringIO()
        scop.write_des(des_out)
        self.assertEqual(des_out.getvalue(), des)
        hie_out = StringIO()
        scop.write_hie(hie_out)
        self.assertEqual(hie_out.getvalue(), hie)
        domain = scop.getDomainBySid('d1hbia_')
        self.assertEqual(domain.sunid, 14996)
        domains = scop.getDomains()
        self.assertEqual(len(domains), 14)
        self.assertEqual(domains[4].sunid, 14988)
        dom = scop.getNodeBySunid(-111)
        self.assertIsNone(dom)
        dom = scop.getDomainBySid('no such domain')
        self.assertIsNone(dom)

    def testSccsOrder(self):
        if False:
            print('Hello World!')
        self.assertEqual(cmp_sccs('a.1.1.1', 'a.1.1.1'), 0)
        self.assertEqual(cmp_sccs('a.1.1.2', 'a.1.1.1'), 1)
        self.assertEqual(cmp_sccs('a.1.1.2', 'a.1.1.11'), -1)
        self.assertEqual(cmp_sccs('a.1.2.2', 'a.1.1.11'), 1)
        self.assertEqual(cmp_sccs('a.1.2.2', 'a.5.1.11'), -1)
        self.assertEqual(cmp_sccs('b.1.2.2', 'a.5.1.11'), 1)
        self.assertEqual(cmp_sccs('b.1.2.2', 'b.1.2'), 1)

    def testParseDomain(self):
        if False:
            for i in range(10):
                print('nop')
        s = '>d1tpt_1 a.46.2.1 (1-70) Thymidine phosphorylase {Escherichia coli}'
        dom = parse_domain(s)
        self.assertEqual(dom.sid, 'd1tpt_1')
        self.assertEqual(dom.sccs, 'a.46.2.1')
        self.assertEqual(dom.residues.pdbid, '1tpt')
        self.assertEqual(dom.description, 'Thymidine phosphorylase {Escherichia coli}')
        s2 = 'd1tpt_1 a.46.2.1 (1tpt 1-70) Thymidine phosphorylase {E. coli}'
        self.assertEqual(s2, str(parse_domain(s2)))
        s3 = 'g1cph.1 g.1.1.1 (1cph B:,A:) Insulin {Cow (Bos taurus)}'
        self.assertEqual(s3, str(parse_domain(s3)))
        s4 = 'e1cph.1a g.1.1.1 (1cph A:) Insulin {Cow (Bos taurus)}'
        self.assertEqual(s4, str(parse_domain(s4)))
        s5 = '>e1cph.1a g.1.1.1 (A:) Insulin {Cow (Bos taurus)}'
        self.assertEqual(s4, str(parse_domain(s5)))
        self.assertRaises(ValueError, parse_domain, 'Totally wrong')

    def testConstructFromDirectory(self):
        if False:
            i = 10
            return i + 15
        scop = Scop(dir_path='SCOP', version='test')
        self.assertIsInstance(scop, Scop)
        domain = scop.getDomainBySid('d1hbia_')
        self.assertEqual(domain.sunid, 14996)

    def testGetAscendent(self):
        if False:
            for i in range(10):
                print('nop')
        scop = Scop(dir_path='SCOP', version='test')
        domain = scop.getDomainBySid('d1hbia_')
        fold = domain.getAscendent('cf')
        self.assertEqual(fold.sunid, 46457)
        sf = domain.getAscendent('superfamily')
        self.assertEqual(sf.sunid, 46458)
        px = domain.getAscendent('px')
        self.assertIsNone(px)
        px2 = sf.getAscendent('px')
        self.assertIsNone(px2)

    def test_get_descendents(self):
        if False:
            return 10
        'Test getDescendents method.'
        scop = Scop(dir_path='SCOP', version='test')
        fold = scop.getNodeBySunid(46457)
        domains = fold.getDescendents('px')
        self.assertEqual(len(domains), 14)
        for d in domains:
            self.assertEqual(d.type, 'px')
        sfs = fold.getDescendents('superfamily')
        self.assertEqual(len(sfs), 1)
        for d in sfs:
            self.assertEqual(d.type, 'sf')
        cl = fold.getDescendents('cl')
        self.assertEqual(cl, [])
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)