"""Tests for Bio.AlignIO.ClustalIO module."""
import unittest
from io import StringIO
from Bio.AlignIO.ClustalIO import ClustalIterator
from Bio.AlignIO.ClustalIO import ClustalWriter
aln_example1 = 'CLUSTAL W (1.81) multiple sequence alignment\n\n\ngi|4959044|gb|AAD34209.1|AF069      MENSDSNDKGSDQSAAQRRSQMDRLDREEAFYQFVNNLSEEDYRLMRDNN 50\ngi|671626|emb|CAA85685.1|           ---------MSPQTETKASVGFKAGVKEYKLTYYTPEYETKDTDILAAFR 41\n                                              * *: ::    :.   :*  :  :. : . :*  ::   .\n\ngi|4959044|gb|AAD34209.1|AF069      LLGTPGESTEEELLRRLQQIKEGPPPQSPDENRAGESSDDVTNSDSIIDW 100\ngi|671626|emb|CAA85685.1|           VTPQPG-----------------VPPEEAGAAVAAESSTGT--------- 65\n                                    :   **                  **:...   *.*** ..         \n\ngi|4959044|gb|AAD34209.1|AF069      LNSVRQTGNTTRSRQRGNQSWRAVSRTNPNSGDFRFSLEINVNRNNGSQT 150\ngi|671626|emb|CAA85685.1|           WTTVWTDGLTSLDRYKG-----RCYHIEPVPG------------------ 92\n                                     .:*   * *: .* :*        : :* .*                  \n\ngi|4959044|gb|AAD34209.1|AF069      SENESEPSTRRLSVENMESSSQRQMENSASESASARPSRAERNSTEAVTE 200\ngi|671626|emb|CAA85685.1|           -EKDQCICYVAYPLDLFEEGSVTNMFTSIVGNVFGFKALRALRLEDLRIP 141\n                                     *::.  .    .:: :*..*  :* .*   .. .  :    .  :    \n\ngi|4959044|gb|AAD34209.1|AF069      VPTTRAQRRA 210\ngi|671626|emb|CAA85685.1|           VAYVKTFQGP 151\n                                    *. .:: : .\n'
aln_example2 = 'CLUSTAL X (1.83) multiple sequence alignment\n\n\nV_Harveyi_PATH                 --MKNWIKVAVAAIA--LSAA------------------TVQAATEVKVG\nB_subtilis_YXEM                MKMKKWTVLVVAALLAVLSACG------------NGNSSSKEDDNVLHVG\nB_subtilis_GlnH_homo_YCKK      MKKALLALFMVVSIAALAACGAGNDNQSKDNAKDGDLWASIKKKGVLTVG\nYA80_HAEIN                     MKKLLFTTALLTGAIAFSTF-----------SHAGEIADRVEKTKTLLVG\nFLIY_ECOLI                     MKLAHLGRQALMGVMAVALVAG---MSVKSFADEG-LLNKVKERGTLLVG\nE_coli_GlnH                    --MKSVLKVSLAALTLAFAVS------------------SHAADKKLVVA\nDeinococcus_radiodurans        -MKKSLLSLKLSGLLVPSVLALS--------LSACSSPSSTLNQGTLKIA\nHISJ_E_COLI                    MKKLVLSLSLVLAFSSATAAF-------------------AAIPQNIRIG\nHISJ_E_COLI                    MKKLVLSLSLVLAFSSATAAF-------------------AAIPQNIRIG\n                                         : .                                 : :.\n\nV_Harveyi_PATH                 MSGRYFPFTFVKQ--DKLQGFEVDMWDEIGKRNDYKIEYVTANFSGLFGL\nB_subtilis_YXEM                ATGQSYPFAYKEN--GKLTGFDVEVMEAVAKKIDMKLDWKLLEFSGLMGE\nB_subtilis_GlnH_homo_YCKK      TEGTYEPFTYHDKDTDKLTGYDVEVITEVAKRLGLKVDFKETQWGSMFAG\nYA80_HAEIN                     TEGTYAPFTFHDK-SGKLTGFDVEVIRKVAEKLGLKVEFKETQWDAMYAG\nFLIY_ECOLI                     LEGTYPPFSFQGD-DGKLTGFEVEFAQQLAKHLGVEASLKPTKWDGMLAS\nE_coli_GlnH                    TDTAFVPFEFKQG--DKYVGFDVDLWAAIAKELKLDYELKPMDFSGIIPA\nDeinococcus_radiodurans        MEGTYPPFTSKNE-QGELVGFDVDIAKAVAQKLNLKPEFVLTEWSGILAG\nHISJ_E_COLI                    TDPTYAPFESKNS-QGELVGFDIDLAKELCKRINTQCTFVENPLDALIPS\nHISJ_E_COLI                    TDPTYAPFESKNS-QGELVGFDIDLAKELCKRINTQCTFVENPLDALIPS\n                                     **       .:  *::::.   : :.   .        ..:   \n\nV_Harveyi_PATH                 LETGRIDTISNQITMTDARKAKYLFADPYVVDG-AQI\nB_subtilis_YXEM                LQTGKLDTISNQVAVTDERKETYNFTKPYAYAG-TQI\nB_subtilis_GlnH_homo_YCKK      LNSKRFDVVANQVG-KTDREDKYDFSDKYTTSR-AVV\nYA80_HAEIN                     LNAKRFDVIANQTNPSPERLKKYSFTTPYNYSG-GVI\nFLIY_ECOLI                     LDSKRIDVVINQVTISDERKKKYDFSTPYTISGIQAL\nE_coli_GlnH                    LQTKNVDLALAGITITDERKKAIDFSDGYYKSG-LLV\nDeinococcus_radiodurans        LQANKYDVIVNQVGITPERQNSIGFSQPYAYSRPEII\nHISJ_E_COLI                    LKAKKIDAIMSSLSITEKRQQEIAFTDKLYAADSRLV\nHISJ_E_COLI                    LKAKKIDAIMSSLSITEKRQQEIAFTDKLYAADSRLV\n                               *.: . *        .  *     *:          :\n\n'
aln_example3 = 'CLUSTAL 2.0.9 multiple sequence alignment\n\n\nTest1seq             ------------------------------------------------------------\nAT3G20900.1-SEQ      ATGAACAAAGTAGCGAGGAAGAACAAAACATCAGGTGAACAAAAAAAAAACTCAATCCAC\nAT3G20900.1-CDS      ------------------------------------------------------------\n                                                                                 \n\nTest1seq             -----AGTTACAATAACTGACGAAGCTAAGTAGGCTACTAATTAACGTCATCAACCTAAT\nAT3G20900.1-SEQ      ATCAAAGTTACAATAACTGACGAAGCTAAGTAGGCTAGAAATTAAAGTCATCAACCTAAT\nAT3G20900.1-CDS      ------------------------------------------------------------\n                                                                                 \n\nTest1seq             ACATAGCACTTAGAAAAAAGTGAAGTAAGAAAATATAAAATAATAAAAGGGTGGGTTATC\nAT3G20900.1-SEQ      ACATAGCACTTAGAAAAAAGTGAAGCAAGAAAATATAAAATAATAAAAGGGTGGGTTATC\nAT3G20900.1-CDS      ------------------------------------------------------------\n                                                                                 \n\nTest1seq             AATTGATAGTGTAAATCATCGTATTCCGGTGATATACCCTACCACAAAAACTCAAACCGA\nAT3G20900.1-SEQ      AATTGATAGTGTAAATCATAGTTGATTTTTGATATACCCTACCACAAAAACTCAAACCGA\nAT3G20900.1-CDS      ------------------------------------------------------------\n                                                                                 \n\nTest1seq             CTTGATTCAAATCATCTCAATAAATTAGCGCCAAAATAATGAAAAAAATAATAACAAACA\nAT3G20900.1-SEQ      CTTGATTCAAATCATCTCAAAAAACAAGCGCCAAAATAATGAAAAAAATAATAACAAAAA\nAT3G20900.1-CDS      ------------------------------------------------------------\n                                                                                 \n\nTest1seq             AAAACAAACCAAAATAAGAAAAAACATTACGCAAAACATAATAATTTACTCTTCGTTATT\nAT3G20900.1-SEQ      CAAACAAACCAAAATAAGAAAAAACATTACGCAAAACATAATAATTTACTCTTCGTTATT\nAT3G20900.1-CDS      ------------------------------------------------------------\n                                                                                 \n\nTest1seq             GTATTAACAAATCAAAGAGCTGAATTTTGATCACCTGCTAATACTACTTTCTGTATTGAT\nAT3G20900.1-SEQ      GTATTAACAAATCAAAGAGATGAATTTTGATCACCTGCTAATACTACTTTCTGTATTGAT\nAT3G20900.1-CDS      ------------------------------------------------------------\n                                                                                 \n\nTest1seq             CCTATATCAACGTAAACAAAGATACTAATAATTAACTAAAAGTACGTTCATCGATCGTGT\nAT3G20900.1-SEQ      CCTATATCAAAAAAAAAAAAGATACTAATAATTAACTAAAAGTACGTTCATCGATCGTGT\nAT3G20900.1-CDS      ------------------------------------------------------ATGAAC\n                                                                             *   \n\nTest1seq             TCGTTGACGAAGAAGAGCTCTATCTCCGGCGGAGCAAAGAAAACGATCTGTCTCCGTCGT\nAT3G20900.1-SEQ      GCGTTGACGAAGAAGAGCTCTATCTCCGGCGGAGCAAAGAAAACGATCTGTCTCCGTCGT\nAT3G20900.1-CDS      AAAGTAGCGAGGAAGAACAAAACATC------AGCAAAGAAAACGATCTGTCTCCGTCGT\n                         *  *** ***** *   *  **      ****************************\n\nTest1seq             AACACACGGTCGCTAGAGAAACTTTGCTTCTTCGGCGCCGGTGGACACGTCAGCATCTCC\nAT3G20900.1-SEQ      AACACACAGTTTTTCGAGACCCTTTGCTTCTTCGGCGCCGGTGGACACGTCAGCATCTCC\nAT3G20900.1-CDS      AACACACAGTTTTTCGAGACCCTTTGCTTCTTCGGCGCCGGTGGACACGTCAGCATCTCC\n                     ******* **   * ****  ***************************************\n\nTest1seq             GGTATCCTAGACTTCTTGGCTTTCGGGGTACAACAACCGCGTGGTGACGTCAGCACCGCT\nAT3G20900.1-SEQ      GGTATCCTAGACTTCTTGGCTTTCGGGGTACAACAACCGCCTGGTGACGTCAGCACCGCT\nAT3G20900.1-CDS      GGTATCCTAGACTTCTTGGCTTTCGGGGTACAACAACCGCCTGGTGACGTCAGCACCGCT\n                     **************************************** *******************\n\nTest1seq             GCTGGGGATGGAGAGGGAACAGAGTT-\nAT3G20900.1-SEQ      GCTGGGGATGGAGAGGGAACAGAGTAG\nAT3G20900.1-CDS      GCTGGGGATGGAGAGGGAACAGAGTAG\n                     *************************  \n'
aln_example4 = 'Kalign (2.0) alignment in ClustalW format\n\nTest1seq             GCTGGGGATGGAGAGGGAACAGAGTT-\nAT3G20900.1-SEQ      GCTGGGGATGGAGAGGGAACAGAGTAG\n\n'
aln_example5 = 'Biopython 1.80.dev0 multiple sequence alignment\n\n\ngi|4959044|gb|AAD34209.1|AF069      ------------MENSDSNDKGSDQSAAQRRSQMDRLDREEAFYQFVNNL\ngi|671626|emb|CAA85685.1|           MSPQTETKASVGFKAGVKEYKLTYYTPEYETKDTDILAAFRVTPQPGVPP\n\ngi|4959044|gb|AAD34209.1|AF069      SEEDYRLMRDNNLLGTPGESTEEELLRRLQQIKEGPPPQSPDENRAGESS\ngi|671626|emb|CAA85685.1|           -EEAGAAVAAESSTGTWTTVWTDGLTS-LDRYK-GRCYHI--EPVPGEKD\n\ngi|4959044|gb|AAD34209.1|AF069      DDVTNSDSIIDWLNSVRQTGNTTRSRQRGNQSWRAVSRTNPNSGDFRFSL\ngi|671626|emb|CAA85685.1|           QCICYVAYPLDLFEEGSVTNMFT-SIV-GNVFGFKALRALRLE-DLRIPV\n\ngi|4959044|gb|AAD34209.1|AF069      EINVNRNNGSQTSENESEPSTRRLSVENMESSSQRQMENSASESASARPS\ngi|671626|emb|CAA85685.1|           AY-VKTFQGPPHGIQVERDKLNKYGRPLLGCTIKPKLGLSAKNYGRAVYE\n\ngi|4959044|gb|AAD34209.1|AF069      RAERNSTEAVTEVPTTRAQRRARSRSPEHRRTRARAERSMSPLQPTSEIP\ngi|671626|emb|CAA85685.1|           CL-RGGLDFTKDDENVNSQPFMRWRD---RFLFC-AEAIYKAQAETGEIK\n\ngi|4959044|gb|AAD34209.1|AF069      RRAPTLEQSSENEPEGSSRTRHHVTLRQQISGPELLGRGLFAASGSRNPS\ngi|671626|emb|CAA85685.1|           GHYLNATAGTC-E-EMIKRAIFARELGVPIVMHDYLTGG-FTANTSLAHY\n\ngi|4959044|gb|AAD34209.1|AF069      QGTSSSDTGSNSESSGSGQRPPTIVLDLQVRRVRPGEYRQRDSIASRTRS\ngi|671626|emb|CAA85685.1|           CRDNGLLLHIHRAMHAVIDRQKNHGMHFRVLAKALRLSGG-DHIHSGTVV\n\ngi|4959044|gb|AAD34209.1|AF069      RSQAPNNTVTYESERGGFRRTFSRSERAGVRTYVSTIRIPIRRILNTGLS\ngi|671626|emb|CAA85685.1|           GKLEGERDITLGFVDLL-RDDFIEKDRSRGI-YF-TQDWVSLPGVIPVAS\n\ngi|4959044|gb|AAD34209.1|AF069      ETTSVAIQTMLRQIMTGFGELSYFMYSDSDSEPSAS--VSSRN-VER-VE\ngi|671626|emb|CAA85685.1|           GGIHVWHMPALTEIFGDDSVLQFGGGTLGHPWGNAPGAVANRVAVEACVK\n\ngi|4959044|gb|AAD34209.1|AF069      SRN-GRGSSGGGNSSGSSSSS-SPSPSSSGESSESSSKMFEGSSEGGSSG\ngi|671626|emb|CAA85685.1|           ARNEGRDLAAEGNAIIREACKWSPELAAACEVWKEIKFEFPAMD------\n\ngi|4959044|gb|AAD34209.1|AF069      PSRKDGRHRAPVTFDESGSLPFFSLAQFFLLNEDDEDQPRGLTKEQIDNL\ngi|671626|emb|CAA85685.1|           --------------------------------------------------\n\ngi|4959044|gb|AAD34209.1|AF069      AMRSFGENDALKTCSVCITEYTEGDKLRKLPCSHEFHVHCIDRWLSENST\ngi|671626|emb|CAA85685.1|           --------------------------------------------------\n\ngi|4959044|gb|AAD34209.1|AF069      CPICRRAVLSSGNRESVV\ngi|671626|emb|CAA85685.1|           ------------------\n\n'

class TestClustalIO(unittest.TestCase):

    def test_one(self):
        if False:
            for i in range(10):
                print('nop')
        alignments = list(ClustalIterator(StringIO(aln_example1)))
        self.assertEqual(1, len(alignments))
        self.assertEqual(alignments[0]._version, '1.81')
        alignment = alignments[0]
        self.assertEqual(2, len(alignment))
        self.assertEqual(alignment[0].id, 'gi|4959044|gb|AAD34209.1|AF069')
        self.assertEqual(alignment[1].id, 'gi|671626|emb|CAA85685.1|')
        self.assertEqual(alignment[0].seq, 'MENSDSNDKGSDQSAAQRRSQMDRLDREEAFYQFVNNLSEEDYRLMRDNNLLGTPGESTEEELLRRLQQIKEGPPPQSPDENRAGESSDDVTNSDSIIDWLNSVRQTGNTTRSRQRGNQSWRAVSRTNPNSGDFRFSLEINVNRNNGSQTSENESEPSTRRLSVENMESSSQRQMENSASESASARPSRAERNSTEAVTEVPTTRAQRRA')

    def test_two(self):
        if False:
            i = 10
            return i + 15
        alignments = list(ClustalIterator(StringIO(aln_example2)))
        self.assertEqual(1, len(alignments))
        self.assertEqual(alignments[0]._version, '1.83')
        alignment = alignments[0]
        self.assertEqual(9, len(alignment))
        self.assertEqual(alignment[-1].id, 'HISJ_E_COLI')
        self.assertEqual(alignment[-1].seq, 'MKKLVLSLSLVLAFSSATAAF-------------------AAIPQNIRIGTDPTYAPFESKNS-QGELVGFDIDLAKELCKRINTQCTFVENPLDALIPSLKAKKIDAIMSSLSITEKRQQEIAFTDKLYAADSRLV')

    def test_cat_one_two(self):
        if False:
            for i in range(10):
                print('nop')
        alignments = list(ClustalIterator(StringIO(aln_example2 + aln_example1)))
        self.assertEqual(2, len(alignments))
        self.assertEqual(9, len(alignments[0]))
        self.assertEqual(137, alignments[0].get_alignment_length())
        self.assertEqual(2, len(alignments[1]))
        self.assertEqual(210, alignments[1].get_alignment_length())

    def test_empty(self):
        if False:
            i = 10
            return i + 15
        'Checking empty file.'
        self.assertEqual(0, len(list(ClustalIterator(StringIO('')))))

    def test_write_read(self):
        if False:
            while True:
                i = 10
        'Checking write/read.'
        alignments = list(ClustalIterator(StringIO(aln_example1))) + list(ClustalIterator(StringIO(aln_example2))) * 2
        handle = StringIO()
        self.assertEqual(3, ClustalWriter(handle).write_file(alignments))
        handle.seek(0)
        for (i, a) in enumerate(ClustalIterator(handle)):
            self.assertEqual(a.get_alignment_length(), alignments[i].get_alignment_length())

    def test_write_read_single(self):
        if False:
            return 10
        'Testing write/read when there is only one sequence.'
        alignment = next(ClustalIterator(StringIO(aln_example1)))
        alignment = alignment[0:1]
        handle = StringIO()
        ClustalWriter(handle).write_file([alignment])
        handle.seek(0)
        for (i, a) in enumerate(ClustalIterator(handle)):
            self.assertEqual(a.get_alignment_length(), alignment.get_alignment_length())
            self.assertEqual(len(a), 1)

    def test_three(self):
        if False:
            for i in range(10):
                print('nop')
        alignments = list(ClustalIterator(StringIO(aln_example3)))
        self.assertEqual(1, len(alignments))
        self.assertEqual(alignments[0]._version, '2.0.9')

    def test_kalign_header(self):
        if False:
            while True:
                i = 10
        'Make sure we can parse the Kalign header.'
        alignment = next(ClustalIterator(StringIO(aln_example4)))
        self.assertEqual(2, len(alignment))

    def test_biopython_header(self):
        if False:
            print('Hello World!')
        'Make sure we can parse the Biopython header.'
        alignment = next(ClustalIterator(StringIO(aln_example5)))
        self.assertEqual(2, len(alignment))
        self.assertEqual(alignment._version, '1.80.dev0')
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)