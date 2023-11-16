"""Parse various position frequency matrix format files."""
import re
from Bio import motifs

class Record(list):
    """Class to store the information in a position frequency matrix table.

    The record inherits from a list containing the individual motifs.
    """

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a string representation of the motifs in the Record object.'
        return '\n'.join((str(motif) for motif in self))

def read(handle, pfm_format):
    if False:
        print('Hello World!')
    'Read motif(s) from a file in various position frequency matrix formats.\n\n    Return the record of PFM(s).\n    Call the appropriate routine based on the format passed.\n    '
    pfm_format = pfm_format.lower().replace('_', '-')
    if pfm_format == 'pfm-four-columns':
        record = _read_pfm_four_columns(handle)
        return record
    elif pfm_format == 'pfm-four-rows':
        record = _read_pfm_four_rows(handle)
        return record
    else:
        raise ValueError("Unknown Position Frequency matrix format '%s'" % pfm_format)

def _read_pfm_four_columns(handle):
    if False:
        i = 10
        return i + 15
    'Read motifs in position frequency matrix format (4 columns) from a file handle.\n\n    # cisbp\n    Pos A   C   G   T\n    1   0.00961538461538462 0.00961538461538462 0.00961538461538462 0.971153846153846\n    2   0.00961538461538462 0.00961538461538462 0.00961538461538462 0.971153846153846\n    3   0.971153846153846   0.00961538461538462 0.00961538461538462 0.00961538461538462\n    4   0.00961538461538462 0.00961538461538462 0.00961538461538462 0.971153846153846\n    5   0.00961538461538462 0.971153846153846   0.00961538461538462 0.00961538461538462\n    6   0.971153846153846   0.00961538461538462 0.00961538461538462 0.00961538461538462\n    7   0.00961538461538462 0.971153846153846   0.00961538461538462 0.00961538461538462\n    8   0.00961538461538462 0.00961538461538462 0.00961538461538462 0.971153846153846\n\n    # c2h2 zfs\n    Gene    ENSG00000197372\n    Pos A   C   G   T\n    1   0.341303    0.132427    0.117054    0.409215\n    2   0.283785    0.077066    0.364552    0.274597\n    3   0.491055    0.078208    0.310520    0.120217\n    4   0.492621    0.076117    0.131007    0.300256\n    5   0.250645    0.361464    0.176504    0.211387\n    6   0.276694    0.498070    0.197793    0.027444\n    7   0.056317    0.014631    0.926202    0.002850\n    8   0.004470    0.007769    0.983797    0.003964\n    9   0.936213    0.058787    0.002387    0.002613\n    10  0.004352    0.004030    0.002418    0.989200\n    11  0.013277    0.008165    0.001991    0.976567\n    12  0.968132    0.002263    0.002868    0.026737\n    13  0.397623    0.052017    0.350783    0.199577\n    14  0.000000    0.000000    1.000000    0.000000\n    15  1.000000    0.000000    0.000000    0.000000\n    16  0.000000    0.000000    1.000000    0.000000\n    17  0.000000    0.000000    1.000000    0.000000\n    18  1.000000    0.000000    0.000000    0.000000\n    19  0.000000    1.000000    0.000000    0.000000\n    20  1.000000    0.000000    0.000000    0.000000\n\n    # c2h2 zfs\n    Gene    FBgn0000210\n    Motif   M1734_0.90\n    Pos A   C   G   T\n    1   0.25    0.0833333   0.0833333   0.583333\n    2   0.75    0.166667    0.0833333   0\n    3   0.833333    0   0   0.166667\n    4   1   0   0   0\n    5   0   0.833333    0.0833333   0.0833333\n    6   0.333333    0   0   0.666667\n    7   0.833333    0   0   0.166667\n    8   0.5 0   0.333333    0.166667\n    9   0.5 0.0833333   0.166667    0.25\n    10  0.333333    0.25    0.166667    0.25\n    11  0.166667    0.25    0.416667    0.166667\n\n    # flyfactorsurvey (cluster buster)\n    >AbdA_Cell_FBgn0000014\n    1   3   0   14\n    0   0   0   18\n    16  0   0   2\n    18  0   0   0\n    1   0   0   17\n    0   0   6   12\n    15  1   2   0\n\n    # homer\n    >ATGACTCATC AP-1(bZIP)/ThioMac-PU.1-ChIP-Seq(GSE21512)/Homer    6.049537    -1.782996e+03   0   9805.3,5781.0,3085.1,2715.0,0.00e+00\n    0.419   0.275   0.277   0.028\n    0.001   0.001   0.001   0.997\n    0.010   0.002   0.965   0.023\n    0.984   0.003   0.001   0.012\n    0.062   0.579   0.305   0.054\n    0.026   0.001   0.001   0.972\n    0.043   0.943   0.001   0.012\n    0.980   0.005   0.001   0.014\n    0.050   0.172   0.307   0.471\n    0.149   0.444   0.211   0.195\n\n    # hocomoco\n    > AHR_si\n    40.51343240527031  18.259112547756697  56.41253757072521  38.77363485291994\n    10.877470982533044  11.870876719950774  34.66312982331297  96.54723985087516\n    21.7165707818416  43.883079837598544  20.706746561638717  67.6523201955933\n    2.5465132509466635  1.3171620263517245  145.8637051322628  4.231336967110781\n    0.0  150.35847450464382  1.4927836298652875  2.1074592421627525\n    3.441039751299748  0.7902972158110341  149.37613720253387  0.3512432070271259\n    0.0  3.441039751299748  0.7024864140542533  149.81519121131782\n    0.0  0.0  153.95871737667187  0.0\n    43.07922333291745  66.87558226865211  16.159862546986584  27.844049228115868\n\n    # neph\n    UW.Motif.0001   atgactca\n    0.772949    0.089579    0.098612    0.038860\n    0.026652    0.004653    0.025056    0.943639\n    0.017663    0.023344    0.918728    0.040264\n    0.919596    0.025414    0.029759    0.025231\n    0.060312    0.772259    0.104968    0.062462\n    0.037406    0.020643    0.006667    0.935284\n    0.047316    0.899024    0.026928    0.026732\n    0.948639    0.019497    0.005737    0.026128\n\n    # tiffin\n    T   A   G   C\n    30  0   28  40\n    0   0   0   99\n    0   55  14  29\n    0   99  0   0\n    20  78  0   0\n    0   52  7   39\n    19  46  11  22\n    0   60  38  0\n    0   33  0   66\n    73  0   25  0\n    99  0   0   0\n    '
    record = Record()
    motif_name = None
    motif_nbr = 0
    motif_nbr_added = 0
    default_nucleotide_order = ['A', 'C', 'G', 'T']
    nucleotide_order = default_nucleotide_order
    nucleotide_counts = {'A': [], 'C': [], 'G': [], 'T': []}
    for line in handle:
        line = line.strip()
        if line:
            columns = line.split()
            nbr_columns = len(columns)
            if line.startswith('#'):
                continue
            elif line.startswith('>'):
                if motif_nbr != 0 and motif_nbr_added != motif_nbr:
                    motif = motifs.Motif(alphabet='GATC', counts=nucleotide_counts)
                    motif.name = motif_name
                    record.append(motif)
                    motif_nbr_added = motif_nbr
                motif_name = line[1:].strip()
                nucleotide_order = default_nucleotide_order
            elif columns[0] == 'Gene':
                if motif_nbr != 0 and motif_nbr_added != motif_nbr:
                    motif = motifs.Motif(alphabet='GATC', counts=nucleotide_counts)
                    motif.name = motif_name
                    record.append(motif)
                    motif_nbr_added = motif_nbr
                motif_name = columns[1]
                nucleotide_order = default_nucleotide_order
            elif columns[0] == 'Motif':
                if motif_nbr != 0 and motif_nbr_added != motif_nbr:
                    motif = motifs.Motif(alphabet='GATC', counts=nucleotide_counts)
                    motif.name = motif_name
                    record.append(motif)
                    motif_nbr_added = motif_nbr
                motif_name = columns[1]
                nucleotide_order = default_nucleotide_order
            elif columns[0] == 'Pos':
                if nbr_columns == 5:
                    if motif_nbr != 0 and motif_nbr_added != motif_nbr:
                        motif = motifs.Motif(alphabet='GATC', counts=nucleotide_counts)
                        motif.name = motif_name
                        record.append(motif)
                        motif_nbr_added = motif_nbr
                    nucleotide_order = default_nucleotide_order
                    if set(columns[1:]) == set(default_nucleotide_order):
                        nucleotide_order = columns[1:]
            elif columns[0] in default_nucleotide_order:
                if nbr_columns == 4:
                    nucleotide_order = default_nucleotide_order
                    if set(columns) == set(default_nucleotide_order):
                        nucleotide_order = columns
            else:
                if nbr_columns == 4:
                    matrix_columns = columns
                elif nbr_columns == 5:
                    matrix_columns = columns[1:]
                else:
                    continue
                if motif_nbr == motif_nbr_added:
                    nucleotide_counts = {'A': [], 'C': [], 'G': [], 'T': []}
                    motif_nbr += 1
                [nucleotide_counts[nucleotide].append(float(nucleotide_count)) for (nucleotide, nucleotide_count) in zip(nucleotide_order, matrix_columns)]
        else:
            if motif_nbr != 0 and motif_nbr_added != motif_nbr:
                motif = motifs.Motif(alphabet='GATC', counts=nucleotide_counts)
                motif.name = motif_name
                record.append(motif)
                motif_nbr_added = motif_nbr
            motif_name = None
            nucleotide_order = default_nucleotide_order
    if motif_nbr != 0 and motif_nbr_added != motif_nbr:
        motif = motifs.Motif(alphabet='GATC', counts=nucleotide_counts)
        motif.name = motif_name
        record.append(motif)
    return record

def _read_pfm_four_rows(handle):
    if False:
        i = 10
        return i + 15
    'Read motifs in position frequency matrix format (4 rows) from a file handle.\n\n    # hdpi\n    A   0   5   6   5   1   0\n    C   1   1   0   0   0   4\n    G   5   0   0   0   3   0\n    T   0   0   0   1   2   2\n\n    # yetfasco\n    A   0.5 0.0 0.0 0.25    0.25    0.25    0.25    0.25    0.25    0.25    0.25    0.25    0.5 0.0 0.0833333334583333\n    T   0.0 0.0 0.0 0.25    0.25    0.25    0.25    0.25    0.25    0.25    0.25    0.25    0.0 0.0 0.0833333334583333\n    G   0.0 1.0 0.0 0.25    0.25    0.25    0.25    0.25    0.25    0.25    0.25    0.25    0.0 1.0 0.249999999875\n    C   0.5 0.0 1.0 0.25    0.25    0.25    0.25    0.25    0.25    0.25    0.25    0.25    0.5 0.0 0.583333333208333\n\n    # flyfactorsurvey ZFP finger\n    A |     92    106    231    135      0      1    780     28      0    700    739     94     60    127    130\n    C |    138     82    129     81    774      1      3      1      0      6     17     49    193    122    148\n    G |    270    398     54    164      7    659      1    750    755     65      1     41    202    234    205\n    T |    290    204    375    411      9    127      6     11     36     20     31    605    335    307    308\n\n    # scertf pcm\n    A | 9 1 1 97 1 94\n    T | 80 1 97 1 1 2\n    C | 9 97 1 1 1 2\n    G | 2 1 1 1 97 2\n\n    # scertf pfm\n    A | 0.090 0.010 0.010 0.970 0.010 0.940\n    C | 0.090 0.970 0.010 0.010 0.010 0.020\n    G | 0.020 0.010 0.010 0.010 0.970 0.020\n    T | 0.800 0.010 0.970 0.010 0.010 0.020\n\n    # idmmpmm\n    > abd-A\n    0.218451749734889 0.0230646871686108 0.656680805938494 0.898197242841994 0.040694591728526 0.132953340402969 0.74907211028632 0.628313891834571\n    0.0896076352067868 0.317338282078473 0.321580063626723 0.0461293743372216 0.0502386002120891 0.040694591728526 0.0284994697773065 0.0339342523860021\n    0.455991516436904 0.0691940615058324 0.0108695652173913 0.0217391304347826 0.0284994697773065 0.0284994697773065 0.016304347826087 0.160127253446448\n    0.235949098621421 0.590402969247084 0.0108695652173913 0.0339342523860021 0.880567338282079 0.797852598091198 0.206124072110286 0.17762460233298\n\n    # JASPAR\n        >MA0001.1 AGL3\n        A  [ 0  3 79 40 66 48 65 11 65  0 ]\n        C  [94 75  4  3  1  2  5  2  3  3 ]\n        G  [ 1  0  3  4  1  0  5  3 28 88 ]\n        T  [ 2 19 11 50 29 47 22 81  1  6 ]\n\n    or::\n\n        >MA0001.1 AGL3\n        0  3 79 40 66 48 65 11 65  0\n        94 75  4  3  1  2  5  2  3  3\n        1  0  3  4  1  0  5  3 28 88\n        2 19 11 50 29 47 22 81  1  6\n    '
    record = Record()
    name_pattern = re.compile('^>\\s*(.+)\\s*')
    row_pattern_with_nucleotide_letter = re.compile('\\s*([ACGT])\\s*[\\[|]*\\s*([0-9.\\-eE\\s]+)\\s*\\]*\\s*')
    row_pattern_without_nucleotide_letter = re.compile('\\s*([0-9.\\-eE\\s]+)\\s*')
    motif_name = None
    nucleotide_counts = {}
    row_count = 0
    nucleotides = ['A', 'C', 'G', 'T']
    for line in handle:
        line = line.strip()
        name_match = name_pattern.match(line)
        row_match_with_nucleotide_letter = row_pattern_with_nucleotide_letter.match(line)
        row_match_without_nucleotide_letter = row_pattern_without_nucleotide_letter.match(line)
        if name_match:
            motif_name = name_match.group(1)
        elif row_match_with_nucleotide_letter:
            (nucleotide, counts_str) = row_match_with_nucleotide_letter.group(1, 2)
            current_nucleotide_counts = counts_str.split()
            nucleotide_counts[nucleotide] = [float(current_nucleotide_count) for current_nucleotide_count in current_nucleotide_counts]
            row_count += 1
            if row_count == 4:
                motif = motifs.Motif(alphabet='GATC', counts=nucleotide_counts)
                if motif_name:
                    motif.name = motif_name
                record.append(motif)
                motif_name = None
                nucleotide_counts = {}
                row_count = 0
        elif row_match_without_nucleotide_letter:
            current_nucleotide_counts = row_match_without_nucleotide_letter.group(1).split()
            nucleotide_counts[nucleotides[row_count]] = [float(current_nucleotide_count) for current_nucleotide_count in current_nucleotide_counts]
            row_count += 1
            if row_count == 4:
                motif = motifs.Motif(alphabet='GATC', counts=nucleotide_counts)
                if motif_name:
                    motif.name = motif_name
                record.append(motif)
                motif_name = None
                nucleotide_counts = {}
                row_count = 0
    return record

def write(motifs):
    if False:
        while True:
            i = 10
    'Return the representation of motifs in Cluster Buster position frequency matrix format.'
    lines = []
    for m in motifs:
        line = f'>{m.name}\n'
        lines.append(line)
        for ACGT_counts in zip(m.counts['A'], m.counts['C'], m.counts['G'], m.counts['T']):
            lines.append('{:0.0f}\t{:0.0f}\t{:0.0f}\t{:0.0f}\n'.format(*ACGT_counts))
    text = ''.join(lines)
    return text