"""Tests for Bio.ExPASy.Prodoc module."""
import os
import unittest
from Bio.ExPASy import Prodoc

class TestProdocRead(unittest.TestCase):
    """Tests for the Prodoc read function."""

    def test_read_pdoc00100(self):
        if False:
            i = 10
            return i + 15
        'Reading Prodoc record PDOC00100.'
        filename = os.path.join('Prosite', 'Doc', 'pdoc00100.txt')
        with open(filename) as handle:
            record = Prodoc.read(handle)
        self.assertEqual(record.accession, 'PDOC00100')
        self.assertEqual(len(record.prosite_refs), 4)
        self.assertEqual(record.prosite_refs[0], ('PS00107', 'PROTEIN_KINASE_ATP'))
        self.assertEqual(record.prosite_refs[1], ('PS00108', 'PROTEIN_KINASE_ST'))
        self.assertEqual(record.prosite_refs[2], ('PS00109', 'PROTEIN_KINASE_TYR'))
        self.assertEqual(record.prosite_refs[3], ('PS50011', 'PROTEIN_KINASE_DOM'))
        self.assertEqual(record.text, '******************************************\n* Protein kinases signatures and profile *\n******************************************\n\nEukaryotic  protein kinases [1 to 5]  are  enzymes  that   belong  to  a  very\nextensive family of  proteins which share a conserved catalytic core common to\nboth serine/threonine and  tyrosine protein kinases.  There  are  a  number of\nconserved regions in the catalytic domain of protein kinases. We have selected\ntwo of these regions to build signature patterns.  The  first region, which is\nlocated in the N-terminal extremity of the catalytic domain, is a glycine-rich\nstretch of residues in the vicinity  of a lysine residue, which has been shown\nto be involved in ATP binding.   The second  region,  which is  located in the\ncentral part of the  catalytic  domain,  contains  a  conserved  aspartic acid\nresidue  which is important for the catalytic activity  of  the enzyme [6]; we\nhave derived  two signature patterns for that region: one specific for serine/\nthreonine kinases  and  the  other  for  tyrosine kinases. We also developed a\nprofile which is based on the alignment in [1] and covers the entire catalytic\ndomain.\n\n-Consensus pattern: [LIV]-G-{P}-G-{P}-[FYWMGSTNH]-[SGA]-{PW}-[LIVCAT]-{PD}-x-\n                    [GSTACLIVMFY]-x(5,18)-[LIVMFYWCSTAR]-[AIVP]-[LIVMFAGCKR]-K\n                    [K binds ATP]\n-Sequences known to belong to this class detected by the pattern: the majority\n of known  protein  kinases  but it fails to find a number of them, especially\n viral kinases  which  are  quite  divergent in this region and are completely\n missed by this pattern.\n-Other sequence(s) detected in Swiss-Prot: 42.\n\n-Consensus pattern: [LIVMFYC]-x-[HY]-x-D-[LIVMFY]-K-x(2)-N-[LIVMFYCT](3)\n                    [D is an active site residue]\n-Sequences known to belong to this class detected by the pattern: Most serine/\n threonine  specific protein  kinases  with  10 exceptions (half of them viral\n kinases) and  also  Epstein-Barr  virus BGLF4 and Drosophila ninaC which have\n respectively Ser and Arg instead of the conserved Lys and which are therefore\n detected by the tyrosine kinase specific pattern described below.\n-Other sequence(s) detected in Swiss-Prot: 1.\n\n-Consensus pattern: [LIVMFYC]-{A}-[HY]-x-D-[LIVMFY]-[RSTAC]-{D}-{PF}-N-\n                    [LIVMFYC](3)\n                    [D is an active site residue]\n-Sequences known to belong to this class detected by the pattern: ALL tyrosine\n specific protein  kinases  with  the  exception of human ERBB3 and mouse blk.\n This pattern    will    also    detect    most    bacterial    aminoglycoside\n phosphotransferases [8,9]  and  herpesviruses ganciclovir kinases [10]; which\n are proteins structurally and evolutionary related to protein kinases.\n-Other sequence(s) detected in Swiss-Prot: 17.\n\n-Sequences known to belong to this class detected by the profile: ALL,  except\n for three  viral  kinases.  This  profile  also  detects  receptor  guanylate\n cyclases (see   <PDOC00430>)  and  2-5A-dependent  ribonucleases.    Sequence\n similarities between  these  two  families  and the eukaryotic protein kinase\n family have been noticed before. It also detects Arabidopsis thaliana kinase-\n like protein TMKL1 which seems to have lost its catalytic activity.\n-Other sequence(s) detected in Swiss-Prot: 4.\n\n-Note: If a protein  analyzed  includes the two protein kinase signatures, the\n probability of it being a protein kinase is close to 100%\n-Note: Eukaryotic-type protein  kinases  have  also  been found in prokaryotes\n such as Myxococcus xanthus [11] and Yersinia pseudotuberculosis.\n-Note: The  patterns  shown  above has been updated since their publication in\n [7].\n\n-Expert(s) to contact by email:\n           Hunter T.; hunter@salk-sc2.sdsc.edu\n           Quinn A.M.; quinn@biomed.med.yale.edu\n\n-Last update: April 2006 / Pattern revised.\n\n')
        self.assertEqual(len(record.references), 11)
        self.assertEqual(record.references[0].number, '1')
        self.assertEqual(record.references[0].authors, 'Hanks S.K., Hunter T.')
        self.assertEqual(record.references[0].citation, '"Protein kinases 6. The eukaryotic protein kinase superfamily: kinase\n(catalytic) domain structure and classification."\nFASEB J. 9:576-596(1995).\nPubMed=7768349')
        self.assertEqual(record.references[1].number, '2')
        self.assertEqual(record.references[1].authors, 'Hunter T.')
        self.assertEqual(record.references[1].citation, '"Protein kinase classification."\nMethods Enzymol. 200:3-37(1991).\nPubMed=1835513')
        self.assertEqual(record.references[2].number, '3')
        self.assertEqual(record.references[2].authors, 'Hanks S.K., Quinn A.M.')
        self.assertEqual(record.references[2].citation, '"Protein kinase catalytic domain sequence database: identification of\nconserved features of primary structure and classification of family\nmembers."\nMethods Enzymol. 200:38-62(1991).\nPubMed=1956325')
        self.assertEqual(record.references[3].number, '4')
        self.assertEqual(record.references[3].authors, 'Hanks S.K.')
        self.assertEqual(record.references[3].citation, 'Curr. Opin. Struct. Biol. 1:369-383(1991).')
        self.assertEqual(record.references[4].number, '5')
        self.assertEqual(record.references[4].authors, 'Hanks S.K., Quinn A.M., Hunter T.')
        self.assertEqual(record.references[4].citation, '"The protein kinase family: conserved features and deduced phylogeny\nof the catalytic domains."\nScience 241:42-52(1988).\nPubMed=3291115')
        self.assertEqual(record.references[5].number, '6')
        self.assertEqual(record.references[5].authors, 'Knighton D.R., Zheng J.H., Ten Eyck L.F., Ashford V.A., Xuong N.-H., Taylor S.S., Sowadski J.M.')
        self.assertEqual(record.references[5].citation, '"Crystal structure of the catalytic subunit of cyclic adenosine\nmonophosphate-dependent protein kinase."\nScience 253:407-414(1991).\nPubMed=1862342')
        self.assertEqual(record.references[6].number, '7')
        self.assertEqual(record.references[6].authors, 'Bairoch A., Claverie J.-M.')
        self.assertEqual(record.references[6].citation, '"Sequence patterns in protein kinases."\nNature 331:22-22(1988).\nPubMed=3340146; DOI=10.1038/331022a0')
        self.assertEqual(record.references[7].number, '8')
        self.assertEqual(record.references[7].authors, 'Benner S.')
        self.assertEqual(record.references[7].citation, 'Nature 329:21-21(1987).')
        self.assertEqual(record.references[8].number, '9')
        self.assertEqual(record.references[8].authors, 'Kirby R.')
        self.assertEqual(record.references[8].citation, '"Evolutionary origin of aminoglycoside phosphotransferase resistance\ngenes."\nJ. Mol. Evol. 30:489-492(1990).\nPubMed=2165531')
        self.assertEqual(record.references[9].number, '10')
        self.assertEqual(record.references[9].authors, 'Littler E., Stuart A.D., Chee M.S.')
        self.assertEqual(record.references[9].citation, 'Nature 358:160-162(1992).')
        self.assertEqual(record.references[10].number, '11')
        self.assertEqual(record.references[10].authors, 'Munoz-Dorado J., Inouye S., Inouye M.')
        self.assertEqual(record.references[10].citation, 'Cell 67:995-1006(1991).')

    def test_read_pdoc00113(self):
        if False:
            for i in range(10):
                print('nop')
        'Reading Prodoc record PDOC00113.'
        filename = os.path.join('Prosite', 'Doc', 'pdoc00113.txt')
        with open(filename) as handle:
            record = Prodoc.read(handle)
        self.assertEqual(record.accession, 'PDOC00113')
        self.assertEqual(len(record.prosite_refs), 1)
        self.assertEqual(record.prosite_refs[0], ('PS00123', 'ALKALINE_PHOSPHATASE'))
        self.assertEqual(record.text, '************************************\n* Alkaline phosphatase active site *\n************************************\n\nAlkaline phosphatase (EC 3.1.3.1) (ALP) [1] is a zinc and magnesium-containing\nmetalloenzyme  which hydrolyzes phosphate esters, optimally at high pH.  It is\nfound in nearly  all living organisms,  with the exception of some plants.  In\nEscherichia coli, ALP (gene phoA) is found in the periplasmic space.  In yeast\nit (gene  PHO8)  is  found  in  lysosome-like vacuoles and in mammals, it is a\nglycoprotein attached to the membrane by a GPI-anchor.\n\nIn mammals, four different isozymes are currently known [2]. Three of them are\ntissue-specific:  the  placental,  placental-like (germ cell)   and intestinal\nisozymes.  The fourth form is  tissue non-specific and was previously known as\nthe liver/bone/kidney isozyme.\n\nStreptomyces\' species  involved  in  the  synthesis  of  streptomycin (SM), an\nantibiotic, express  a  phosphatase (EC 3.1.3.39) (gene strK) which is  highly\nrelated to ALP.   It specifically cleaves  both  streptomycin-6-phosphate and,\nmore slowly, streptomycin-3"-phosphate.\n\nA serine is involved   in the catalytic activity of ALP. The region around the\nactive site serine is relatively well conserved and can be used as a signature\npattern.\n\n-Consensus pattern: [IV]-x-D-S-[GAS]-[GASC]-[GAST]-[GA]-T\n                    [S is the active site residue]\n-Sequences known to belong to this class detected by the pattern: ALL.\n-Other sequence(s) detected in Swiss-Prot: 3.\n-Last update: June 1994 / Text revised.\n\n')
        self.assertEqual(len(record.references), 3)
        self.assertEqual(record.references[0].number, '1')
        self.assertEqual(record.references[0].authors, 'Trowsdale J., Martin D., Bicknell D., Campbell I.')
        self.assertEqual(record.references[0].citation, '"Alkaline phosphatases."\nBiochem. Soc. Trans. 18:178-180(1990).\nPubMed=2379681')
        self.assertEqual(record.references[1].number, '2')
        self.assertEqual(record.references[1].authors, 'Manes T., Glade K., Ziomek C.A., Millan J.L.')
        self.assertEqual(record.references[1].citation, '"Genomic structure and comparison of mouse tissue-specific alkaline\nphosphatase genes."\nGenomics 8:541-554(1990).\nPubMed=2286375')
        self.assertEqual(record.references[2].number, '3')
        self.assertEqual(record.references[2].authors, 'Mansouri K., Piepersberg W.')
        self.assertEqual(record.references[2].citation, '"Genetics of streptomycin production in Streptomyces griseus:\nnucleotide sequence of five genes, strFGHIK, including a phosphatase\ngene."\nMol. Gen. Genet. 228:459-469(1991).\nPubMed=1654502')

    def test_read_pdoc00144(self):
        if False:
            for i in range(10):
                print('nop')
        'Reading Prodoc record PDOC00144.'
        filename = os.path.join('Prosite', 'Doc', 'pdoc00144.txt')
        with open(filename) as handle:
            record = Prodoc.read(handle)
        self.assertEqual(record.accession, 'PDOC00144')
        self.assertEqual(len(record.prosite_refs), 2)
        self.assertEqual(record.prosite_refs[0], ('PS00159', 'ALDOLASE_KDPG_KHG_1'))
        self.assertEqual(record.prosite_refs[1], ('PS00160', 'ALDOLASE_KDPG_KHG_2'))
        self.assertEqual(record.text, '*************************************************\n* KDPG and KHG aldolases active site signatures *\n*************************************************\n\n4-hydroxy-2-oxoglutarate aldolase (EC 4.1.3.16)  (KHG-aldolase)  catalyzes the\ninterconversion of  4-hydroxy-2-oxoglutarate  into  pyruvate  and  glyoxylate.\nPhospho-2-dehydro-3-deoxygluconate  aldolase   (EC 4.1.2.14)   (KDPG-aldolase)\ncatalyzes the interconversion of  6-phospho-2-dehydro-3-deoxy-D-gluconate into\npyruvate and glyceraldehyde 3-phosphate.\n\nThese two enzymes are structurally and functionally related [1]. They are both\nhomotrimeric proteins of approximately 220 amino-acid residues. They are class\nI aldolases whose catalytic mechanism involves  the formation of a Schiff-base\nintermediate  between  the  substrate  and the epsilon-amino group of a lysine\nresidue. In both enzymes, an arginine is required for catalytic activity.\n\nWe developed  two signature patterns for these enzymes. The first one contains\nthe active  site  arginine  and the second, the lysine involved in the Schiff-\nbase formation.\n\n-Consensus pattern: G-[LIVM]-x(3)-E-[LIV]-T-[LF]-R\n                    [R is the active site residue]\n-Sequences known to belong to this class detected by the pattern: ALL,  except\n for Bacillus  subtilis  KDPG-aldolase  which  has  Thr  instead of Arg in the\n active site.\n-Other sequence(s) detected in Swiss-Prot: NONE.\n\n-Consensus pattern: G-x(3)-[LIVMF]-K-[LF]-F-P-[SA]-x(3)-G\n                    [K is involved in Schiff-base formation]\n-Sequences known to belong to this class detected by the pattern: ALL.\n-Other sequence(s) detected in Swiss-Prot: NONE.\n\n-Last update: November 1997 / Patterns and text revised.\n\n')
        self.assertEqual(len(record.references), 1)
        self.assertEqual(record.references[0].number, '1')
        self.assertEqual(record.references[0].authors, 'Vlahos C.J., Dekker E.E.')
        self.assertEqual(record.references[0].citation, '"The complete amino acid sequence and identification of the\nactive-site arginine peptide of Escherichia coli\n2-keto-4-hydroxyglutarate aldolase."\nJ. Biol. Chem. 263:11683-11691(1988).\nPubMed=3136164')

    def test_read_pdoc00149(self):
        if False:
            return 10
        'Reading Prodoc record PDOC00149.'
        filename = os.path.join('Prosite', 'Doc', 'pdoc00149.txt')
        with open(filename) as handle:
            record = Prodoc.read(handle)
        self.assertEqual(record.accession, 'PDOC00149')
        self.assertEqual(len(record.prosite_refs), 1)
        self.assertEqual(record.prosite_refs[0], ('PS00165', 'DEHYDRATASE_SER_THR'))
        self.assertEqual(record.text, '*********************************************************************\n* Serine/threonine dehydratases pyridoxal-phosphate attachment site *\n*********************************************************************\n\nSerine and threonine  dehydratases [1,2]  are  functionally  and  structurally\nrelated pyridoxal-phosphate dependent enzymes:\n\n - L-serine dehydratase (EC 4.3.1.17) and D-serine  dehydratase  (EC 4.3.1.18)\n   catalyze the dehydratation of L-serine (respectively D-serine) into ammonia\n   and pyruvate.\n - Threonine dehydratase  (EC 4.3.1.19) (TDH) catalyzes  the  dehydratation of\n   threonine into  alpha-ketobutarate  and  ammonia.  In Escherichia coli  and\n   other microorganisms,  two  classes  of  TDH  are  known  to  exist. One is\n   involved in  the  biosynthesis of isoleucine, the other in hydroxamino acid\n   catabolism.\n\nThreonine synthase  (EC 4.2.3.1) is  also  a  pyridoxal-phosphate  enzyme,  it\ncatalyzes the  transformation of  homoserine-phosphate into threonine.  It has\nbeen shown [3] that  threonine  synthase  is  distantly related to the serine/\nthreonine dehydratases.\n\nIn all these enzymes, the pyridoxal-phosphate group is  attached  to a  lysine\nresidue.  The sequence around  this residue is sufficiently conserved to allow\nthe derivation  of  a  pattern  specific  to serine/threonine dehydratases and\nthreonine synthases.\n\n-Consensus pattern: [DESH]-x(4,5)-[STVG]-{EVKD}-[AS]-[FYI]-K-[DLIFSA]-[RLVMF]-\n                    [GA]-[LIVMGA]\n                    [The K is the pyridoxal-P attachment site]\n-Sequences known to belong to this class detected by the pattern: ALL.\n-Other sequence(s) detected in Swiss-Prot: 17.\n\n-Note: Some   bacterial L-serine dehydratases - such as those from Escherichia\n coli - are iron-sulfur proteins [4] and do not belong to this family.\n\n-Last update: December 2004 / Pattern and text revised.\n\n')
        self.assertEqual(len(record.references), 4)
        self.assertEqual(record.references[0].number, '1')
        self.assertEqual(record.references[0].authors, 'Ogawa H., Gomi T., Konishi K., Date T., Nakashima H., Nose K., Matsuda Y., Peraino C., Pitot H.C., Fujioka M.')
        self.assertEqual(record.references[0].citation, '"Human liver serine dehydratase. cDNA cloning and sequence homology\nwith hydroxyamino acid dehydratases from other sources."\nJ. Biol. Chem. 264:15818-15823(1989).\nPubMed=2674117')
        self.assertEqual(record.references[1].number, '2')
        self.assertEqual(record.references[1].authors, 'Datta P., Goss T.J., Omnaas J.R., Patil R.V.')
        self.assertEqual(record.references[1].citation, '"Covalent structure of biodegradative threonine dehydratase of\nEscherichia coli: homology with other dehydratases."\nProc. Natl. Acad. Sci. U.S.A. 84:393-397(1987).\nPubMed=3540965')
        self.assertEqual(record.references[2].number, '3')
        self.assertEqual(record.references[2].authors, 'Parsot C.')
        self.assertEqual(record.references[2].citation, '"Evolution of biosynthetic pathways: a common ancestor for threonine\nsynthase, threonine dehydratase and D-serine dehydratase."\nEMBO J. 5:3013-3019(1986).\nPubMed=3098560')
        self.assertEqual(record.references[3].number, '4')
        self.assertEqual(record.references[3].authors, 'Grabowski R., Hofmeister A.E.M., Buckel W.')
        self.assertEqual(record.references[3].citation, '"Bacterial L-serine dehydratases: a new family of enzymes containing\niron-sulfur clusters."\nTrends Biochem. Sci. 18:297-300(1993).\nPubMed=8236444')

    def test_read_pdoc00340(self):
        if False:
            return 10
        'Reading Prodoc record PDOC00340.'
        filename = os.path.join('Prosite', 'Doc', 'pdoc00340.txt')
        with open(filename) as handle:
            record = Prodoc.read(handle)
        self.assertEqual(record.accession, 'PDOC00340')
        self.assertEqual(len(record.prosite_refs), 3)
        self.assertEqual(record.prosite_refs[0], ('PS00406', 'ACTINS_1'))
        self.assertEqual(record.prosite_refs[1], ('PS00432', 'ACTINS_2'))
        self.assertEqual(record.prosite_refs[2], ('PS01132', 'ACTINS_ACT_LIKE'))
        self.assertEqual(record.text, '*********************\n* Actins signatures *\n*********************\n\nActins [1 to 4] are highly conserved contractile  proteins that are present in\nall eukaryotic cells. In vertebrates there are three groups of actin isoforms:\nalpha, beta and gamma.  The alpha actins are found in muscle tissues and are a\nmajor constituent of the contractile apparatus.  The beta and gamma actins co-\nexists in most cell  types as  components of the cytoskeleton and as mediators\nof internal cell motility.  In plants [5]  there  are  many isoforms which are\nprobably involved  in  a  variety of  functions such as cytoplasmic streaming,\ncell shape determination,  tip growth,  graviperception, cell wall deposition,\netc.\n\nActin exists either in a monomeric form (G-actin) or in a polymerized form (F-\nactin). Each actin monomer  can  bind a molecule of ATP;  when  polymerization\noccurs, the ATP is hydrolyzed.\n\nActin is a protein of from 374 to 379 amino acid  residues.  The  structure of\nactin has been highly conserved in the course of evolution.\n\nRecently some  divergent  actin-like  proteins have been identified in several\nspecies. These proteins are:\n\n - Centractin  (actin-RPV)  from mammals, fungi (yeast ACT5, Neurospora crassa\n   ro-4) and  Pneumocystis  carinii  (actin-II).  Centractin  seems  to  be  a\n   component of  a  multi-subunit  centrosomal complex involved in microtubule\n   based vesicle motility. This subfamily is also known as ARP1.\n - ARP2  subfamily  which  includes  chicken ACTL, yeast ACT2, Drosophila 14D,\n   C.elegans actC.\n - ARP3  subfamily  which includes actin 2 from mammals, Drosophila 66B, yeast\n   ACT4 and fission yeast act2.\n - ARP4  subfamily  which includes yeast ACT3 and Drosophila 13E.\n\nWe developed  three  signature  patterns. The first two are specific to actins\nand span  positions  54 to 64 and 357 to 365. The last signature picks up both\nactins and  the actin-like proteins and corresponds to positions 106 to 118 in\nactins.\n\n-Consensus pattern: [FY]-[LIV]-[GV]-[DE]-E-[ARV]-[QLAH]-x(1,2)-[RKQ](2)-[GD]\n-Sequences known to belong to this class detected by the pattern: ALL,  except\n for the actin-like proteins and 10 actins.\n-Other sequence(s) detected in Swiss-Prot: NONE.\n\n-Consensus pattern: W-[IVC]-[STAK]-[RK]-x-[DE]-Y-[DNE]-[DE]\n-Sequences known to belong to this class detected by the pattern: ALL,  except\n for the actin-like proteins and 9 actins.\n-Other sequence(s) detected in Swiss-Prot: NONE.\n\n-Consensus pattern: [LM]-[LIVMA]-T-E-[GAPQ]-x-[LIVMFYWHQPK]-[NS]-[PSTAQ]-x(2)-\n                    N-[KR]\n-Sequences known to belong to this class detected by the pattern: ALL,  except\n for 5 actins.\n-Other sequence(s) detected in Swiss-Prot: NONE.\n\n-Last update: December 2004 / Patterns and text revised.\n\n')
        self.assertEqual(len(record.references), 5)
        self.assertEqual(record.references[0].number, '1')
        self.assertEqual(record.references[0].authors, 'Sheterline P., Clayton J., Sparrow J.C.')
        self.assertEqual(record.references[0].citation, '(In) Actins, 3rd Edition, Academic Press Ltd, London, (1996).')
        self.assertEqual(record.references[1].number, '2')
        self.assertEqual(record.references[1].authors, 'Pollard T.D., Cooper J.A.')
        self.assertEqual(record.references[1].citation, 'Annu. Rev. Biochem. 55:987-1036(1986).')
        self.assertEqual(record.references[2].number, '3')
        self.assertEqual(record.references[2].authors, 'Pollard T.D.')
        self.assertEqual(record.references[2].citation, '"Actin."\nCurr. Opin. Cell Biol. 2:33-40(1990).\nPubMed=2183841')
        self.assertEqual(record.references[3].number, '4')
        self.assertEqual(record.references[3].authors, 'Rubenstein P.A.')
        self.assertEqual(record.references[3].citation, '"The functional importance of multiple actin isoforms."\nBioEssays 12:309-315(1990).\nPubMed=2203335')
        self.assertEqual(record.references[4].number, '5')
        self.assertEqual(record.references[4].authors, 'Meagher R.B., McLean B.G.')
        self.assertEqual(record.references[4].citation, 'Cell Motil. Cytoskeleton 16:164-166(1990).')

    def test_read_pdoc00424(self):
        if False:
            i = 10
            return i + 15
        'Reading Prodoc record PDOC00424.'
        filename = os.path.join('Prosite', 'Doc', 'pdoc00424.txt')
        with open(filename) as handle:
            record = Prodoc.read(handle)
        self.assertEqual(record.accession, 'PDOC00424')
        self.assertEqual(len(record.prosite_refs), 1)
        self.assertEqual(record.prosite_refs[0], ('PS00488', 'PAL_HISTIDASE'))
        self.assertEqual(record.text, '**********************************************************\n* Phenylalanine and histidine ammonia-lyases active site *\n**********************************************************\n\nPhenylalanine ammonia-lyase (EC 4.3.1.5) (PAL) is  a  key  enzyme of plant and\nfungi  phenylpropanoid  metabolism  which is involved in the biosynthesis of a\nwide  variety  of secondary metabolites such  as  flavanoids,   furanocoumarin\nphytoalexins and  cell  wall  components.  These compounds have many important\nroles in plants during normal growth and in responses to environmental stress.\nPAL catalyzes  the  removal  of  an  ammonia  group from phenylalanine to form\ntrans-cinnamate.\n\nHistidine ammonia-lyase (EC 4.3.1.3) (histidase)  catalyzes  the first step in\nhistidine degradation, the removal of  an  ammonia  group  from  histidine  to\nproduce urocanic acid.\n\nThe two types of enzymes are functionally and  structurally related [1].  They\nare the only enzymes  which are known to have the modified amino acid dehydro-\nalanine (DHA) in their active site. A serine residue has been shown [2,3,4] to\nbe the  precursor  of  this  essential electrophilic moiety. The region around\nthis active  site  residue  is  well  conserved and can be used as a signature\npattern.\n\n-Consensus pattern: [GS]-[STG]-[LIVM]-[STG]-[SAC]-S-G-[DH]-L-x-[PN]-L-[SA]-\n                    x(2,3)-[SAGVTL]\n                    [S is the active site residue]\n-Sequences known to belong to this class detected by the pattern: ALL.\n-Other sequence(s) detected in Swiss-Prot: NONE.\n-Last update: April 2006 / Pattern revised.\n\n')
        self.assertEqual(len(record.references), 4)
        self.assertEqual(record.references[0].number, '1')
        self.assertEqual(record.references[0].authors, 'Taylor R.G., Lambert M.A., Sexsmith E., Sadler S.J., Ray P.N., Mahuran D.J., McInnes R.R.')
        self.assertEqual(record.references[0].citation, '"Cloning and expression of rat histidase. Homology to two bacterial\nhistidases and four phenylalanine ammonia-lyases."\nJ. Biol. Chem. 265:18192-18199(1990).\nPubMed=2120224')
        self.assertEqual(record.references[1].number, '2')
        self.assertEqual(record.references[1].authors, 'Langer M., Reck G., Reed J., Retey J.')
        self.assertEqual(record.references[1].citation, '"Identification of serine-143 as the most likely precursor of\ndehydroalanine in the active site of histidine ammonia-lyase. A study\nof the overexpressed enzyme by site-directed mutagenesis."\nBiochemistry 33:6462-6467(1994).\nPubMed=8204579')
        self.assertEqual(record.references[2].number, '3')
        self.assertEqual(record.references[2].authors, 'Schuster B., Retey J.')
        self.assertEqual(record.references[2].citation, '"Serine-202 is the putative precursor of the active site\ndehydroalanine of phenylalanine ammonia lyase. Site-directed\nmutagenesis studies on the enzyme from parsley (Petroselinum crispum\nL.)."\nFEBS Lett. 349:252-254(1994).\nPubMed=8050576')
        self.assertEqual(record.references[3].number, '4')
        self.assertEqual(record.references[3].authors, 'Taylor R.G., McInnes R.R.')
        self.assertEqual(record.references[3].citation, '"Site-directed mutagenesis of conserved serines in rat histidase.\nIdentification of serine 254 as an essential active site residue."\nJ. Biol. Chem. 269:27473-27477(1994).\nPubMed=7961661')

    def test_read_pdoc00472(self):
        if False:
            for i in range(10):
                print('nop')
        'Reading Prodoc record PDOC00472.'
        filename = os.path.join('Prosite', 'Doc', 'pdoc00472.txt')
        with open(filename) as handle:
            record = Prodoc.read(handle)
        self.assertEqual(record.accession, 'PDOC00472')
        self.assertEqual(len(record.prosite_refs), 1)
        self.assertEqual(record.prosite_refs[0], ('PS00546', 'CYSTEINE_SWITCH'))
        self.assertEqual(record.text, "*****************************\n* Matrixins cysteine switch *\n*****************************\n\nMammalian extracellular matrix metalloproteinases (EC 3.4.24.-), also known as\nmatrixins [1] (see <PDOC00129>), are zinc-dependent enzymes. They are secreted\nby cells  in an inactive form (zymogen) that differs from the mature enzyme by\nthe presence  of  an  N-terminal propeptide. A highly conserved octapeptide is\nfound two  residues  downstream  of the C-terminal end of the propeptide. This\nregion has been shown to be  involved  in  autoinhibition  of matrixins [2,3];\na cysteine  within the octapeptide chelates  the  active  site  zinc ion, thus\ninhibiting the  enzyme.  This  region has been called the 'cysteine switch' or\n'autoinhibitor region'.\n\nA cysteine switch has been found in the following zinc proteases:\n\n - MMP-1 (EC 3.4.24.7) (interstitial collagenase).\n - MMP-2 (EC 3.4.24.24) (72 Kd gelatinase).\n - MMP-3 (EC 3.4.24.17) (stromelysin-1).\n - MMP-7 (EC 3.4.24.23) (matrilysin).\n - MMP-8 (EC 3.4.24.34) (neutrophil collagenase).\n - MMP-9 (EC 3.4.24.35) (92 Kd gelatinase).\n - MMP-10 (EC 3.4.24.22) (stromelysin-2).\n - MMP-11 (EC 3.4.24.-) (stromelysin-3).\n - MMP-12 (EC 3.4.24.65) (macrophage metalloelastase).\n - MMP-13 (EC 3.4.24.-) (collagenase 3).\n - MMP-14 (EC 3.4.24.-) (membrane-type matrix metalliproteinase 1).\n - MMP-15 (EC 3.4.24.-) (membrane-type matrix metalliproteinase 2).\n - MMP-16 (EC 3.4.24.-) (membrane-type matrix metalliproteinase 3).\n - Sea urchin hatching enzyme (EC 3.4.24.12) (envelysin) [4].\n - Chlamydomonas reinhardtii gamete lytic enzyme (GLE) [5].\n\n-Consensus pattern: P-R-C-[GN]-x-P-[DR]-[LIVSAPKQ]\n                    [C chelates the zinc ion]\n-Sequences known to belong to this class detected by the pattern: ALL,  except\n for cat MMP-7 and mouse MMP-11.\n-Other sequence(s) detected in Swiss-Prot: NONE.\n-Last update: November 1997 / Pattern and text revised.\n\n")
        self.assertEqual(len(record.references), 5)
        self.assertEqual(record.references[0].number, '1')
        self.assertEqual(record.references[0].authors, 'Woessner J.F. Jr.')
        self.assertEqual(record.references[0].citation, '"Matrix metalloproteinases and their inhibitors in connective tissue\nremodeling."\nFASEB J. 5:2145-2154(1991).\nPubMed=1850705')
        self.assertEqual(record.references[1].number, '2')
        self.assertEqual(record.references[1].authors, 'Sanchez-Lopez R., Nicholson R., Gesnel M.C., Matrisian L.M., Breathnach R.')
        self.assertEqual(record.references[1].citation, 'J. Biol. Chem. 263:11892-11899(1988).')
        self.assertEqual(record.references[2].number, '3')
        self.assertEqual(record.references[2].authors, 'Park A.J., Matrisian L.M., Kells A.F., Pearson R., Yuan Z.Y., Navre M.')
        self.assertEqual(record.references[2].citation, '"Mutational analysis of the transin (rat stromelysin) autoinhibitor\nregion demonstrates a role for residues surrounding the \'cysteine\nswitch\'."\nJ. Biol. Chem. 266:1584-1590(1991).\nPubMed=1988438')
        self.assertEqual(record.references[3].number, '4')
        self.assertEqual(record.references[3].authors, 'Lepage T., Gache C.')
        self.assertEqual(record.references[3].citation, '"Early expression of a collagenase-like hatching enzyme gene in the\nsea urchin embryo."\nEMBO J. 9:3003-3012(1990).\nPubMed=2167841')
        self.assertEqual(record.references[4].number, '5')
        self.assertEqual(record.references[4].authors, 'Kinoshita T., Fukuzawa H., Shimada T., Saito T., Matsuda Y.')
        self.assertEqual(record.references[4].citation, '"Primary structure and expression of a gamete lytic enzyme in\nChlamydomonas reinhardtii: similarity of functional domains to matrix\nmetalloproteases."\nProc. Natl. Acad. Sci. U.S.A. 89:4693-4697(1992).\nPubMed=1584806')

    def test_read_pdoc00640(self):
        if False:
            i = 10
            return i + 15
        'Reading Prodoc record PDOC00640.'
        filename = os.path.join('Prosite', 'Doc', 'pdoc00640.txt')
        with open(filename) as handle:
            record = Prodoc.read(handle)
        self.assertEqual(record.accession, 'PDOC00640')
        self.assertEqual(len(record.prosite_refs), 1)
        self.assertEqual(record.prosite_refs[0], ('PS00812', 'GLYCOSYL_HYDROL_F8'))
        self.assertEqual(record.text, '******************************************\n* Glycosyl hydrolases family 8 signature *\n******************************************\n\nThe microbial degradation  of cellulose and  xylans requires  several types of\nenzymes such as endoglucanases (EC 3.2.1.4),  cellobiohydrolases (EC 3.2.1.91)\n(exoglucanases), or xylanases (EC 3.2.1.8) [1,2].  Fungi and bacteria produces\na spectrum of cellulolytic  enzymes (cellulases)  and  xylanases which, on the\nbasis of sequence similarities,  can be classified into families. One of these\nfamilies is known as the cellulase family D [3] or as  the glycosyl hydrolases\nfamily 8  [4,E1].  The  enzymes  which  are  currently known to belong to this\nfamily are listed below.\n\n - Acetobacter xylinum endonuclease cmcAX.\n - Bacillus strain KSM-330 acidic endonuclease K (Endo-K).\n - Cellulomonas josui endoglucanase 2 (celB).\n - Cellulomonas uda endoglucanase.\n - Clostridium cellulolyticum endoglucanases C (celcCC).\n - Clostridium thermocellum endoglucanases A (celA).\n - Erwinia chrysanthemi minor endoglucanase y (celY).\n - Bacillus circulans beta-glucanase (EC 3.2.1.73).\n - Escherichia coli hypothetical protein yhjM.\n\nThe most conserved region in  these enzymes is  a stretch of about 20 residues\nthat contains  two conserved aspartate. The first asparatate is thought [5] to\nact as the nucleophile in the catalytic mechanism. We have used this region as\na signature pattern.\n\n-Consensus pattern: A-[ST]-D-[AG]-D-x(2)-[IM]-A-x-[SA]-[LIVM]-[LIVMG]-x-A-\n                    x(3)-[FW]\n                    [The first D is an active site residue]\n-Sequences known to belong to this class detected by the pattern: ALL.\n-Other sequence(s) detected in Swiss-Prot: NONE.\n\n-Expert(s) to contact by email:\n           Henrissat B.; bernie@afmb.cnrs-mrs.fr\n\n-Last update: November 1997 / Text revised.\n\n')
        self.assertEqual(len(record.references), 6)
        self.assertEqual(record.references[0].number, '1')
        self.assertEqual(record.references[0].authors, 'Beguin P.')
        self.assertEqual(record.references[0].citation, '"Molecular biology of cellulose degradation."\nAnnu. Rev. Microbiol. 44:219-248(1990).\nPubMed=2252383; DOI=10.1146/annurev.mi.44.100190.001251')
        self.assertEqual(record.references[1].number, '2')
        self.assertEqual(record.references[1].authors, 'Gilkes N.R., Henrissat B., Kilburn D.G., Miller R.C. Jr., Warren R.A.J.')
        self.assertEqual(record.references[1].citation, '"Domains in microbial beta-1, 4-glycanases: sequence conservation,\nfunction, and enzyme families."\nMicrobiol. Rev. 55:303-315(1991).\nPubMed=1886523')
        self.assertEqual(record.references[2].number, '3')
        self.assertEqual(record.references[2].authors, 'Henrissat B., Claeyssens M., Tomme P., Lemesle L., Mornon J.-P.')
        self.assertEqual(record.references[2].citation, '"Cellulase families revealed by hydrophobic cluster analysis."\nGene 81:83-95(1989).\nPubMed=2806912')
        self.assertEqual(record.references[3].number, '4')
        self.assertEqual(record.references[3].authors, 'Henrissat B.')
        self.assertEqual(record.references[3].citation, '"A classification of glycosyl hydrolases based on amino acid sequence\nsimilarities."\nBiochem. J. 280:309-316(1991).\nPubMed=1747104')
        self.assertEqual(record.references[4].number, '5')
        self.assertEqual(record.references[4].authors, 'Alzari P.M., Souchon H., Dominguez R.')
        self.assertEqual(record.references[4].citation, '"The crystal structure of endoglucanase CelA, a family 8 glycosyl\nhydrolase from Clostridium thermocellum."\nStructure 4:265-275(1996).\nPubMed=8805535')
        self.assertEqual(record.references[5].number, 'E1')
        self.assertEqual(record.references[5].authors, '')
        self.assertEqual(record.references[5].citation, 'http://www.expasy.org/cgi-bin/lists?glycosid.txt')

    def test_read_pdoc00787(self):
        if False:
            return 10
        'Reading Prodoc record PDOC00787.'
        filename = os.path.join('Prosite', 'Doc', 'pdoc00787.txt')
        with open(filename) as handle:
            record = Prodoc.read(handle)
        self.assertEqual(record.accession, 'PDOC00787')
        self.assertEqual(len(record.prosite_refs), 1)
        self.assertEqual(record.prosite_refs[0], ('PS01027', 'GLYCOSYL_HYDROL_F39'))
        self.assertEqual(record.text, '******************************************************\n* Glycosyl hydrolases family 39 putative active site *\n******************************************************\n\nIt has  been  shown  [1,E1]  that  the  following  glycosyl  hydrolases can be\nclassified into a single family on the basis of sequence similarities:\n\n - Mammalian lysosomal alpha-L-iduronidase (EC 3.2.1.76).\n - Caldocellum  saccharolyticum  and  Thermoanaerobacter saccharolyticum beta-\n   xylosidase (EC 3.2.1.37) (gene xynB).\n\nThe best  conserved  regions  in  these  enzymes is  located in the N-terminal\nsection. It   contains  a  glutamic  acid  residue  which,  on  the  basis  of\nsimilarities with other  families of glycosyl hydrolases [2], probably acts as\nthe proton donor in the catalytic mechanism. We use this region as a signature\npattern.\n\n-Consensus pattern: W-x-F-E-x-W-N-E-P-[DN]\n                    [The second E may be the active site residue]\n-Sequences known to belong to this class detected by the pattern: ALL.\n-Other sequence(s) detected in Swiss-Prot: NONE.\n\n-Expert(s) to contact by email:\n           Henrissat B.; bernie@afmb.cnrs-mrs.fr\n\n-Last update: May 2004 / Text revised.\n\n')
        self.assertEqual(len(record.references), 3)
        self.assertEqual(record.references[0].number, '1')
        self.assertEqual(record.references[0].authors, 'Henrissat B., Bairoch A.')
        self.assertEqual(record.references[0].citation, '"New families in the classification of glycosyl hydrolases based on\namino acid sequence similarities."\nBiochem. J. 293:781-788(1993).\nPubMed=8352747')
        self.assertEqual(record.references[1].number, '2')
        self.assertEqual(record.references[1].authors, 'Henrissat B., Callebaut I., Fabrega S., Lehn P., Mornon J.-P., Davies G.')
        self.assertEqual(record.references[1].citation, '"Conserved catalytic machinery and the prediction of a common fold for\nseveral families of glycosyl hydrolases."\nProc. Natl. Acad. Sci. U.S.A. 92:7090-7094(1995).\nPubMed=7624375')
        self.assertEqual(record.references[2].number, 'E1')
        self.assertEqual(record.references[2].authors, '')
        self.assertEqual(record.references[2].citation, 'http://www.expasy.org/cgi-bin/lists?glycosid.txt')

    def test_read_pdoc0933(self):
        if False:
            while True:
                i = 10
        'Reading Prodoc record PDOC00933.'
        filename = os.path.join('Prosite', 'Doc', 'pdoc00933.txt')
        with open(filename) as handle:
            record = Prodoc.read(handle)
        self.assertEqual(record.accession, 'PDOC00933')
        self.assertEqual(len(record.prosite_refs), 1)
        self.assertEqual(record.prosite_refs[0], ('PS01213', 'GLOBIN_FAM_2'))
        self.assertEqual(record.text, '**********************************************\n* Protozoan/cyanobacterial globins signature *\n**********************************************\n\nGlobins are heme-containing  proteins involved in  binding and/or transporting\noxygen [1]. Almost all globins belong to a large family (see <PDOC00793>), the\nonly exceptions  are  the  following proteins which form a family of their own\n[2,3,4]:\n\n - Monomeric  hemoglobins  from the protozoan Paramecium caudatum, Tetrahymena\n   pyriformis and Tetrahymena thermophila.\n - Cyanoglobins  from  the  cyanobacteria Nostoc commune and Synechocystis PCC\n   6803.\n - Globins  LI637  and  LI410  from  the chloroplast of the alga Chlamydomonas\n   eugametos.\n - Mycobacterium tuberculosis globins glbN and glbO.\n\nThese proteins  contain a conserved histidine which could be involved in heme-\nbinding. As a signature pattern, we use a conserved region that ends with this\nresidue.\n\n-Consensus pattern: F-[LF]-x(4)-[GE]-G-[PAT]-x(2)-[YW]-x-[GSE]-[KRQAE]-x(1,5)-\n                    [LIVM]-x(3)-H\n                    [The H may be a heme ligand]\n-Sequences known to belong to this class detected by the pattern: ALL.\n-Other sequence(s) detected in Swiss-Prot: NONE.\n-Last update: April 2006 / Pattern revised.\n\n')
        self.assertEqual(len(record.references), 4)
        self.assertEqual(record.references[0].number, '1')
        self.assertEqual(record.references[0].authors, 'Concise Encyclopedia Biochemistry, Second Edition, Walter de Gruyter, Berlin New-York (1988).')
        self.assertEqual(record.references[0].citation, '')
        self.assertEqual(record.references[1].number, '2')
        self.assertEqual(record.references[1].authors, 'Takagi T.')
        self.assertEqual(record.references[1].citation, 'Curr. Opin. Struct. Biol. 3:413-418(1993).')
        self.assertEqual(record.references[2].number, '3')
        self.assertEqual(record.references[2].authors, 'Couture M., Chamberland H., St-Pierre B., Lafontaine J., Guertin M.')
        self.assertEqual(record.references[2].citation, '"Nuclear genes encoding chloroplast hemoglobins in the unicellular\ngreen alga Chlamydomonas eugametos."\nMol. Gen. Genet. 243:185-197(1994).\nPubMed=8177215')
        self.assertEqual(record.references[3].number, '4')
        self.assertEqual(record.references[3].authors, 'Couture M., Das T.K., Savard P.Y., Ouellet Y., Wittenberg J.B., Wittenberg B.A., Rousseau D.L., Guertin M.')
        self.assertEqual(record.references[3].citation, '"Structural investigations of the hemoglobin of the cyanobacterium\nSynechocystis PCC6803 reveal a unique distal heme pocket."\nEur. J. Biochem. 267:4770-4780(2000).\nPubMed=10903511')

class TestProdocParse(unittest.TestCase):
    """Tests for the Prodoc parse function."""

    def test_parse_pdoc(self):
        if False:
            while True:
                i = 10
        'Parsing an excerpt of prosite.doc.'
        filename = os.path.join('Prosite', 'Doc', 'prosite.excerpt.doc')
        with open(filename) as handle:
            records = Prodoc.parse(handle)
            record = next(records)
            self.assertEqual(record.accession, 'PDOC00000')
            self.assertEqual(len(record.prosite_refs), 0)
            self.assertEqual(record.text, '**********************************\n*** PROSITE documentation file ***\n**********************************\n\nRelease 20.43 of 10-Feb-2009.\n\nPROSITE is developed by the Swiss Institute of Bioinformatics (SIB) under\nthe responsability of Amos Bairoch and Nicolas Hulo.\n\nThis release was prepared by: Nicolas Hulo, Virginie Bulliard, Petra\nLangendijk-Genevaux and Christian Sigrist with the help of Edouard\nde Castro, Lorenzo Cerutti, Corinne Lachaize and Amos Bairoch.\n\n\nSee: http://www.expasy.org/prosite/\nEmail: prosite@expasy.org\n\nAcknowledgements:\n\n - To all those mentioned in this document who have reviewed the entry(ies)\n   for which they are listed as experts. With specific thanks to Rein Aasland,\n   Mark Boguski, Peer Bork, Josh Cherry, Andre Chollet, Frank Kolakowski,\n   David Landsman, Bernard Henrissat, Eugene Koonin, Steve Henikoff, Manuel\n   Peitsch and Jonathan Reizer.\n - Jim Apostolopoulos is the author of the PDOC00699 entry.\n - Brigitte Boeckmann is the author of the PDOC00691, PDOC00703, PDOC00829,\n   PDOC00796, PDOC00798, PDOC00799, PDOC00906, PDOC00907, PDOC00908,\n   PDOC00912, PDOC00913, PDOC00924, PDOC00928, PDOC00929, PDOC00955,\n   PDOC00961, PDOC00966, PDOC00988 and PDOC50020 entries.\n - Jean-Louis Boulay is the author of the PDOC01051, PDOC01050, PDOC01052,\n   PDOC01053 and PDOC01054 entries.\n - Ryszard Brzezinski is the author of the PDOC60000 entry.\n - Elisabeth Coudert is the author of the PDOC00373 entry.\n - Kirill Degtyarenko is the author of the PDOC60001 entry.\n - Christian Doerig is the author of the PDOC01049 entry.\n - Kay Hofmann is the author of the PDOC50003, PDOC50006, PDOC50007 and\n   PDOC50017 entries.\n - Chantal Hulo is the author of the PDOC00987 entry.\n - Karine Michoud is the author of the PDOC01044 and PDOC01042 entries.\n - Yuri Panchin is the author of the PDOC51013 entry.\n - S. Ramakumar is the author of the PDOC51052, PDOC60004, PDOC60010,\n   PDOC60011, PDOC60015, PDOC60016, PDOC60018, PDOC60020, PDOC60021,\n   PDOC60022, PDOC60023, PDOC60024, PDOC60025, PDOC60026, PDOC60027,\n   PDOC60028, PDOC60029 and PDOC60030 entries.\n - Keith Robison is the author of the PDOC00830 and PDOC00861 entries.\n\n   ------------------------------------------------------------------------\n   PROSITE is copyright.   It  is  produced  by  the  Swiss  Institute   of\n   Bioinformatics (SIB). There are no restrictions on its use by non-profit\n   institutions as long as its  content is in no way modified. Usage by and\n   for commercial  entities requires a license agreement.   For information\n   about  the  licensing  scheme   send  an  email to license@isb-sib.ch or\n   see: http://www.expasy.org/prosite/prosite_license.htm.\n   ------------------------------------------------------------------------\n\n')
            record = next(records)
            self.assertEqual(record.accession, 'PDOC00001')
            self.assertEqual(len(record.prosite_refs), 1)
            self.assertEqual(record.prosite_refs[0], ('PS00001', 'ASN_GLYCOSYLATION'))
            self.assertEqual(record.text, '************************\n* N-glycosylation site *\n************************\n\nIt has been known for a long time [1] that potential N-glycosylation sites are\nspecific to the consensus sequence Asn-Xaa-Ser/Thr.  It must be noted that the\npresence of the consensus  tripeptide  is  not sufficient  to conclude that an\nasparagine residue is glycosylated, due to  the fact that the  folding of  the\nprotein plays an important  role in the  regulation of N-glycosylation [2]. It\nhas been shown [3] that  the  presence of proline between Asn and Ser/Thr will\ninhibit N-glycosylation; this  has  been confirmed by a recent [4] statistical\nanalysis of glycosylation sites, which also  shows that about 50% of the sites\nthat have a proline C-terminal to Ser/Thr are not glycosylated.\n\nIt must also  be noted that there  are  a few  reported cases of glycosylation\nsites with the pattern Asn-Xaa-Cys; an  experimentally demonstrated occurrence\nof such a non-standard site is found in the plasma protein C [5].\n\n-Consensus pattern: N-{P}-[ST]-{P}\n                    [N is the glycosylation site]\n-Last update: May 1991 / Text revised.\n\n')
            self.assertEqual(record.references[0].number, '1')
            self.assertEqual(record.references[0].authors, 'Marshall R.D.')
            self.assertEqual(record.references[0].citation, '"Glycoproteins."\nAnnu. Rev. Biochem. 41:673-702(1972).\nPubMed=4563441; DOI=10.1146/annurev.bi.41.070172.003325')
            self.assertEqual(record.references[1].number, '2')
            self.assertEqual(record.references[1].authors, 'Pless D.D., Lennarz W.J.')
            self.assertEqual(record.references[1].citation, '"Enzymatic conversion of proteins to glycoproteins."\nProc. Natl. Acad. Sci. U.S.A. 74:134-138(1977).\nPubMed=264667')
            self.assertEqual(record.references[2].number, '3')
            self.assertEqual(record.references[2].authors, 'Bause E.')
            self.assertEqual(record.references[2].citation, '"Structural requirements of N-glycosylation of proteins. Studies with\nproline peptides as conformational probes."\nBiochem. J. 209:331-336(1983).\nPubMed=6847620')
            self.assertEqual(record.references[3].number, '4')
            self.assertEqual(record.references[3].authors, 'Gavel Y., von Heijne G.')
            self.assertEqual(record.references[3].citation, '"Sequence differences between glycosylated and non-glycosylated\nAsn-X-Thr/Ser acceptor sites: implications for protein engineering."\nProtein Eng. 3:433-442(1990).\nPubMed=2349213')
            self.assertEqual(record.references[4].number, '5')
            self.assertEqual(record.references[4].authors, 'Miletich J.P., Broze G.J. Jr.')
            self.assertEqual(record.references[4].citation, '"Beta protein C is not glycosylated at asparagine 329. The rate of\ntranslation may influence the frequency of usage at\nasparagine-X-cysteine sites."\nJ. Biol. Chem. 265:11397-11404(1990).\nPubMed=1694179')
            record = next(records)
            self.assertEqual(record.accession, 'PDOC00004')
            self.assertEqual(len(record.prosite_refs), 1)
            self.assertEqual(record.prosite_refs[0], ('PS00004', 'CAMP_PHOSPHO_SITE'))
            self.assertEqual(record.text, '****************************************************************\n* cAMP- and cGMP-dependent protein kinase phosphorylation site *\n****************************************************************\n\nThere has been a  number of studies  relative to the  specificity of cAMP- and\ncGMP-dependent protein kinases [1,2,3].  Both types of kinases appear to share\na preference  for  the  phosphorylation  of serine or threonine residues found\nclose to at least  two consecutive N-terminal  basic residues. It is important\nto note that there are quite a number of exceptions to this rule.\n\n-Consensus pattern: [RK](2)-x-[ST]\n                    [S or T is the phosphorylation site]\n-Last update: June 1988 / First entry.\n\n')
            self.assertEqual(record.references[0].number, '1')
            self.assertEqual(record.references[0].authors, 'Fremisco J.R., Glass D.B., Krebs E.G.')
            self.assertEqual(record.references[0].citation, 'J. Biol. Chem. 255:4240-4245(1980).')
            self.assertEqual(record.references[1].number, '2')
            self.assertEqual(record.references[1].authors, 'Glass D.B., Smith S.B.')
            self.assertEqual(record.references[1].citation, '"Phosphorylation by cyclic GMP-dependent protein kinase of a synthetic\npeptide corresponding to the autophosphorylation site in the enzyme."\nJ. Biol. Chem. 258:14797-14803(1983).\nPubMed=6317673')
            self.assertEqual(record.references[2].number, '3')
            self.assertEqual(record.references[2].authors, 'Glass D.B., el-Maghrabi M.R., Pilkis S.J.')
            self.assertEqual(record.references[2].citation, '"Synthetic peptides corresponding to the site phosphorylated in\n6-phosphofructo-2-kinase/fructose-2,6-bisphosphatase as substrates of\ncyclic nucleotide-dependent protein kinases."\nJ. Biol. Chem. 261:2987-2993(1986).\nPubMed=3005275')
            record = next(records)
            self.assertEqual(record.accession, 'PDOC60030')
            self.assertEqual(len(record.prosite_refs), 1)
            self.assertEqual(record.prosite_refs[0], ('PS60030', 'BACTERIOCIN_IIA'))
            self.assertEqual(record.text, "******************************************\n* Bacteriocin class IIa family signature *\n******************************************\n\nMany Gram-positive  bacteria  produce  ribosomally  synthesized  antimicrobial\npeptides, often  termed  bacteriocins. One important and well studied class of\nbacteriocins is the class IIa or pediocin-like bacteriocins produced by lactic\nacid bacteria.  All  class  IIa  bacteriocins  are produced by food-associated\nstrains, isolated  from  a  variety of food products of industrial and natural\norigins, including  meat  products,  dairy  products and vegetables. Class IIa\nbacteriocins are all cationic, display anti-Listeria activity, and kill target\ncells by permeabilizing the cell membrane [1-3].\n\nClass IIa  bacteriocins  contain  between  37  and 48 residues. Based on their\nprimary structures,  the  peptide  chains  of  class  IIa  bacteriocins may be\ndivided roughly into two regions: a hydrophilic, cationic and highly conserved\nN-terminal region,  and  a  less  conserved hydrophobic/amphiphilic C-terminal\nregion. The  N-terminal  region  contains  the conserved Y-G-N-G-V/L 'pediocin\nbox' motif  and  two conserved cysteine residues joined by a disulfide bridge.\nIt forms  a  three-stranded antiparallel beta-sheet supported by the conserved\ndisulfide bridge  (see <PDB:1OG7>). This cationic N-terminal beta-sheet domain\nmediates binding of the class IIa bacteriocin to the target cell membrane. The\nC-terminal region forms a hairpin-like domain (see <PDB:1OG7>) that penetrates\ninto the  hydrophobic  part  of  the  target  cell membrane, thereby mediating\nleakage through  the  membrane.  The  two domains are joined by a hinge, which\nenables movement of the domains relative to each other [2,3].\n\nSome proteins  known  to belong to the class IIa bacteriocin family are listed\nbelow:\n\n - Pediococcus acidilactici pediocin PA-1.\n - Leuconostoc mesenteroides mesentericin Y105.\n - Carnobacterium piscicola carnobacteriocin B2.\n - Lactobacillus sake sakacin P.\n - Enterococcus faecium enterocin A.\n - Enterococcus faecium enterocin P.\n - Leuconostoc gelidum leucocin A.\n - Lactobacillus curvatus curvacin A.\n - Listeria innocua listeriocin 743A.\n\nThe pattern  we  developed  for  the  class  IIa bacteriocin family covers the\n'pediocin box' motif.\n\n-Conserved pattern: Y-G-N-G-[VL]-x-C-x(4)-C\n-Sequences known to belong to this class detected by the pattern: ALL.\n-Other sequence(s) detected in Swiss-Prot: NONE.\n\n-Expert(s) to contact by email:\n           Ramakumar S.; ramak@physics.iisc.ernet.in\n\n-Last update: March 2006 / First entry.\n\n")
            self.assertEqual(record.references[0].number, '1')
            self.assertEqual(record.references[0].authors, 'Ennahar S., Sonomoto K., Ishizaki A.')
            self.assertEqual(record.references[0].citation, '"Class IIa bacteriocins from lactic acid bacteria: antibacterial\nactivity and food preservation."\nJ. Biosci. Bioeng. 87:705-716(1999).\nPubMed=16232543')
            self.assertEqual(record.references[1].number, '2')
            self.assertEqual(record.references[1].authors, 'Johnsen L., Fimland G., Nissen-Meyer J.')
            self.assertEqual(record.references[1].citation, '"The C-terminal domain of pediocin-like antimicrobial peptides (class\nIIa bacteriocins) is involved in specific recognition of the\nC-terminal part of cognate immunity proteins and in determining the\nantimicrobial spectrum."\nJ. Biol. Chem. 280:9243-9250(2005).\nPubMed=15611086; DOI=10.1074/jbc.M412712200')
            self.assertEqual(record.references[2].number, '3')
            self.assertEqual(record.references[2].authors, 'Fimland G., Johnsen L., Dalhus B., Nissen-Meyer J.')
        self.assertEqual(record.references[2].citation, '"Pediocin-like antimicrobial peptides (class IIa bacteriocins) and\ntheir immunity proteins: biosynthesis, structure, and mode of\naction."\nJ. Pept. Sci. 11:688-696(2005).\nPubMed=16059970; DOI=10.1002/psc.699')
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)