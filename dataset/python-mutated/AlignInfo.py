"""Extract information from alignment objects.

In order to try and avoid huge alignment objects with tons of functions,
functions which return summary type information about alignments should
be put into classes in this module.
"""
import math
import sys
from collections import Counter
from Bio.Seq import Seq

class SummaryInfo:
    """Calculate summary info about the alignment.

    This class should be used to calculate information summarizing the
    results of an alignment. This may either be straight consensus info
    or more complicated things.
    """

    def __init__(self, alignment):
        if False:
            return 10
        'Initialize with the alignment to calculate information on.\n\n        ic_vector attribute. A list of ic content for each column number.\n        '
        self.alignment = alignment
        self.ic_vector = []

    def dumb_consensus(self, threshold=0.7, ambiguous='X', require_multiple=False):
        if False:
            for i in range(10):
                print('nop')
        "Output a fast consensus sequence of the alignment.\n\n        This doesn't do anything fancy at all. It will just go through the\n        sequence residue by residue and count up the number of each type\n        of residue (ie. A or G or T or C for DNA) in all sequences in the\n        alignment. If the percentage of the most common residue type is\n        greater then the passed threshold, then we will add that residue type,\n        otherwise an ambiguous character will be added.\n\n        This could be made a lot fancier (ie. to take a substitution matrix\n        into account), but it just meant for a quick and dirty consensus.\n\n        Arguments:\n         - threshold - The threshold value that is required to add a particular\n           atom.\n         - ambiguous - The ambiguous character to be added when the threshold is\n           not reached.\n         - require_multiple - If set as True, this will require that more than\n           1 sequence be part of an alignment to put it in the consensus (ie.\n           not just 1 sequence and gaps).\n\n        "
        consensus = ''
        con_len = self.alignment.get_alignment_length()
        for n in range(con_len):
            atom_dict = Counter()
            num_atoms = 0
            for record in self.alignment:
                try:
                    c = record[n]
                except IndexError:
                    continue
                if c != '-' and c != '.':
                    atom_dict[c] += 1
                    num_atoms += 1
            max_atoms = []
            max_size = 0
            for atom in atom_dict:
                if atom_dict[atom] > max_size:
                    max_atoms = [atom]
                    max_size = atom_dict[atom]
                elif atom_dict[atom] == max_size:
                    max_atoms.append(atom)
            if require_multiple and num_atoms == 1:
                consensus += ambiguous
            elif len(max_atoms) == 1 and max_size / num_atoms >= threshold:
                consensus += max_atoms[0]
            else:
                consensus += ambiguous
        return Seq(consensus)

    def gap_consensus(self, threshold=0.7, ambiguous='X', require_multiple=False):
        if False:
            return 10
        'Output a fast consensus sequence of the alignment, allowing gaps.\n\n        Same as dumb_consensus(), but allows gap on the output.\n\n        Things to do:\n         - Let the user define that with only one gap, the result\n           character in consensus is gap.\n         - Let the user select gap character, now\n           it takes the same as input.\n\n        '
        consensus = ''
        con_len = self.alignment.get_alignment_length()
        for n in range(con_len):
            atom_dict = Counter()
            num_atoms = 0
            for record in self.alignment:
                try:
                    c = record[n]
                except IndexError:
                    continue
                atom_dict[c] += 1
                num_atoms += 1
            max_atoms = []
            max_size = 0
            for atom in atom_dict:
                if atom_dict[atom] > max_size:
                    max_atoms = [atom]
                    max_size = atom_dict[atom]
                elif atom_dict[atom] == max_size:
                    max_atoms.append(atom)
            if require_multiple and num_atoms == 1:
                consensus += ambiguous
            elif len(max_atoms) == 1 and max_size / num_atoms >= threshold:
                consensus += max_atoms[0]
            else:
                consensus += ambiguous
        return Seq(consensus)

    def replacement_dictionary(self, skip_chars=None, letters=None):
        if False:
            while True:
                i = 10
        "Generate a replacement dictionary to plug into a substitution matrix.\n\n        This should look at an alignment, and be able to generate the number\n        of substitutions of different residues for each other in the\n        aligned object.\n\n        Will then return a dictionary with this information::\n\n            {('A', 'C') : 10, ('C', 'A') : 12, ('G', 'C') : 15 ....}\n\n        This also treats weighted sequences. The following example shows how\n        we calculate the replacement dictionary. Given the following\n        multiple sequence alignment::\n\n            GTATC  0.5\n            AT--C  0.8\n            CTGTC  1.0\n\n        For the first column we have::\n\n            ('A', 'G') : 0.5 * 0.8 = 0.4\n            ('C', 'G') : 0.5 * 1.0 = 0.5\n            ('A', 'C') : 0.8 * 1.0 = 0.8\n\n        We then continue this for all of the columns in the alignment, summing\n        the information for each substitution in each column, until we end\n        up with the replacement dictionary.\n\n        Arguments:\n         - skip_chars - Not used; setting it to anything other than None\n           will raise a ValueError\n         - letters - An iterable (e.g. a string or list of characters to include.\n        "
        if skip_chars is not None:
            raise ValueError("argument skip_chars has been deprecated; instead, please use 'letters' to specify the characters you want to include")
        rep_dict = {(letter1, letter2): 0 for letter1 in letters for letter2 in letters}
        for rec_num1 in range(len(self.alignment)):
            for rec_num2 in range(rec_num1 + 1, len(self.alignment)):
                self._pair_replacement(self.alignment[rec_num1].seq, self.alignment[rec_num2].seq, self.alignment[rec_num1].annotations.get('weight', 1.0), self.alignment[rec_num2].annotations.get('weight', 1.0), rep_dict, letters)
        return rep_dict

    def _pair_replacement(self, seq1, seq2, weight1, weight2, dictionary, letters):
        if False:
            print('Hello World!')
        'Compare two sequences and generate info on the replacements seen (PRIVATE).\n\n        Arguments:\n         - seq1, seq2 - The two sequences to compare.\n         - weight1, weight2 - The relative weights of seq1 and seq2.\n         - dictionary - The dictionary containing the starting replacement\n           info that we will modify.\n         - letters - A list of characters to include when calculating replacements.\n\n        '
        for (residue1, residue2) in zip(seq1, seq2):
            if residue1 in letters and residue2 in letters:
                dictionary[residue1, residue2] += weight1 * weight2

    def _get_all_letters(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a string containing the expected letters in the alignment (PRIVATE).'
        set_letters = set()
        for record in self.alignment:
            set_letters.update(record.seq)
        list_letters = sorted(set_letters)
        all_letters = ''.join(list_letters)
        return all_letters

    def pos_specific_score_matrix(self, axis_seq=None, chars_to_ignore=None):
        if False:
            for i in range(10):
                print('nop')
        'Create a position specific score matrix object for the alignment.\n\n        This creates a position specific score matrix (pssm) which is an\n        alternative method to look at a consensus sequence.\n\n        Arguments:\n         - chars_to_ignore - A list of all characters not to include in\n           the pssm.\n         - axis_seq - An optional argument specifying the sequence to\n           put on the axis of the PSSM. This should be a Seq object. If nothing\n           is specified, the consensus sequence, calculated with default\n           parameters, will be used.\n\n        Returns:\n         - A PSSM (position specific score matrix) object.\n\n        '
        all_letters = self._get_all_letters()
        if not all_letters:
            raise ValueError('_get_all_letters returned empty string')
        if chars_to_ignore is None:
            chars_to_ignore = []
        if not isinstance(chars_to_ignore, list):
            raise TypeError('chars_to_ignore should be a list.')
        gap_char = '-'
        chars_to_ignore.append(gap_char)
        for char in chars_to_ignore:
            all_letters = all_letters.replace(char, '')
        if axis_seq:
            left_seq = axis_seq
            if len(axis_seq) != self.alignment.get_alignment_length():
                raise ValueError('Axis sequence length does not equal the get_alignment_length')
        else:
            left_seq = self.dumb_consensus()
        pssm_info = []
        for residue_num in range(len(left_seq)):
            score_dict = dict.fromkeys(all_letters, 0)
            for record in self.alignment:
                try:
                    this_residue = record.seq[residue_num]
                except IndexError:
                    this_residue = None
                if this_residue and this_residue not in chars_to_ignore:
                    weight = record.annotations.get('weight', 1.0)
                    try:
                        score_dict[this_residue] += weight
                    except KeyError:
                        raise ValueError('Residue %s not found' % this_residue) from None
            pssm_info.append((left_seq[residue_num], score_dict))
        return PSSM(pssm_info)

    def information_content(self, start=0, end=None, e_freq_table=None, log_base=2, chars_to_ignore=None, pseudo_count=0):
        if False:
            i = 10
            return i + 15
        "Calculate the information content for each residue along an alignment.\n\n        Arguments:\n         - start, end - The starting an ending points to calculate the\n           information content. These points should be relative to the first\n           sequence in the alignment, starting at zero (ie. even if the 'real'\n           first position in the seq is 203 in the initial sequence, for\n           the info content, we need to use zero). This defaults to the entire\n           length of the first sequence.\n         - e_freq_table - A dictionary specifying the expected frequencies\n           for each letter (e.g. {'G' : 0.4, 'C' : 0.4, 'T' : 0.1, 'A' : 0.1}).\n           Gap characters should not be included, since these should not have\n           expected frequencies.\n         - log_base - The base of the logarithm to use in calculating the\n           information content. This defaults to 2 so the info is in bits.\n         - chars_to_ignore - A listing of characters which should be ignored\n           in calculating the info content. Defaults to none.\n\n        Returns:\n         - A number representing the info content for the specified region.\n\n        Please see the Biopython manual for more information on how information\n        content is calculated.\n\n        "
        if end is None:
            end = len(self.alignment[0].seq)
        if chars_to_ignore is None:
            chars_to_ignore = []
        if start < 0 or end > len(self.alignment[0].seq):
            raise ValueError('Start (%s) and end (%s) are not in the range %s to %s' % (start, end, 0, len(self.alignment[0].seq)))
        random_expected = None
        all_letters = self._get_all_letters()
        for char in chars_to_ignore:
            all_letters = all_letters.replace(char, '')
        info_content = {}
        for residue_num in range(start, end):
            freq_dict = self._get_letter_freqs(residue_num, self.alignment, all_letters, chars_to_ignore, pseudo_count, e_freq_table, random_expected)
            column_score = self._get_column_info_content(freq_dict, e_freq_table, log_base, random_expected)
            info_content[residue_num] = column_score
        total_info = sum(info_content.values())
        self.ic_vector = []
        for (i, k) in enumerate(info_content):
            self.ic_vector.append(info_content[i + start])
        return total_info

    def _get_letter_freqs(self, residue_num, all_records, letters, to_ignore, pseudo_count=0, e_freq_table=None, random_expected=None):
        if False:
            i = 10
            return i + 15
        'Determine the frequency of specific letters in the alignment (PRIVATE).\n\n        Arguments:\n         - residue_num - The number of the column we are getting frequencies\n           from.\n         - all_records - All of the SeqRecords in the alignment.\n         - letters - The letters we are interested in getting the frequency\n           for.\n         - to_ignore - Letters we are specifically supposed to ignore.\n         - pseudo_count - Optional argument specifying the Pseudo count (k)\n           to add in order to prevent a frequency of 0 for a letter.\n         - e_freq_table - An optional argument specifying a dictionary with\n           the expected frequencies for each letter.\n         - random_expected - Optional argument that specify the frequency to use\n           when e_freq_table is not defined.\n\n        This will calculate the frequencies of each of the specified letters\n        in the alignment at the given frequency, and return this as a\n        dictionary where the keys are the letters and the values are the\n        frequencies. Pseudo count can be added to prevent a null frequency\n        '
        freq_info = dict.fromkeys(letters, 0)
        total_count = 0
        gap_char = '-'
        if pseudo_count < 0:
            raise ValueError('Positive value required for pseudo_count, %s provided' % pseudo_count)
        for record in all_records:
            try:
                if record.seq[residue_num] not in to_ignore:
                    weight = record.annotations.get('weight', 1.0)
                    freq_info[record.seq[residue_num]] += weight
                    total_count += weight
            except KeyError:
                raise ValueError('Residue %s not found in letters %s' % (record.seq[residue_num], letters)) from None
        if e_freq_table:
            for key in freq_info:
                if key != gap_char and key not in e_freq_table:
                    raise ValueError('%s not found in expected frequency table' % key)
        if total_count == 0:
            for letter in freq_info:
                if freq_info[letter] != 0:
                    raise ValueError('freq_info[letter] is not 0')
        else:
            for letter in freq_info:
                if pseudo_count and (random_expected or e_freq_table):
                    if e_freq_table:
                        ajust_freq = e_freq_table[letter]
                    else:
                        ajust_freq = random_expected
                    ajusted_letter_count = freq_info[letter] + ajust_freq * pseudo_count
                    ajusted_total = total_count + pseudo_count
                    freq_info[letter] = ajusted_letter_count / ajusted_total
                else:
                    freq_info[letter] = freq_info[letter] / total_count
        return freq_info

    def _get_column_info_content(self, obs_freq, e_freq_table, log_base, random_expected):
        if False:
            i = 10
            return i + 15
        'Calculate the information content for a column (PRIVATE).\n\n        Arguments:\n         - obs_freq - The frequencies observed for each letter in the column.\n         - e_freq_table - An optional argument specifying a dictionary with\n           the expected frequencies for each letter.\n         - log_base - The base of the logarithm to use in calculating the\n           info content.\n\n        '
        gap_char = '-'
        if e_freq_table:
            for key in obs_freq:
                if key != gap_char and key not in e_freq_table:
                    raise ValueError(f'Frequency table provided does not contain observed letter {key}')
        total_info = 0.0
        for letter in obs_freq:
            inner_log = 0.0
            if letter != gap_char:
                if e_freq_table:
                    inner_log = obs_freq[letter] / e_freq_table[letter]
                else:
                    inner_log = obs_freq[letter] / random_expected
            if inner_log > 0:
                letter_info = obs_freq[letter] * math.log(inner_log) / math.log(log_base)
                total_info += letter_info
        return total_info

    def get_column(self, col):
        if False:
            for i in range(10):
                print('nop')
        'Return column of alignment.'
        return self.alignment[:, col]

class PSSM:
    """Represent a position specific score matrix.

    This class is meant to make it easy to access the info within a PSSM
    and also make it easy to print out the information in a nice table.

    Let's say you had an alignment like this::

        GTATC
        AT--C
        CTGTC

    The position specific score matrix (when printed) looks like::

          G A T C
        G 1 1 0 1
        T 0 0 3 0
        A 1 1 0 0
        T 0 0 2 0
        C 0 0 0 3

    You can access a single element of the PSSM using the following::

        your_pssm[sequence_number][residue_count_name]

    For instance, to get the 'T' residue for the second element in the
    above alignment you would need to do:

    your_pssm[1]['T']
    """

    def __init__(self, pssm):
        if False:
            for i in range(10):
                print('nop')
        'Initialize with pssm data to represent.\n\n        The pssm passed should be a list with the following structure:\n\n        list[0] - The letter of the residue being represented (for instance,\n        from the example above, the first few list[0]s would be GTAT...\n        list[1] - A dictionary with the letter substitutions and counts.\n        '
        self.pssm = pssm

    def __getitem__(self, pos):
        if False:
            for i in range(10):
                print('nop')
        return self.pssm[pos][1]

    def __str__(self):
        if False:
            i = 10
            return i + 15
        out = ' '
        all_residues = sorted(self.pssm[0][1])
        for res in all_residues:
            out += '   %s' % res
        out += '\n'
        for item in self.pssm:
            out += '%s ' % item[0]
            for res in all_residues:
                out += ' %.1f' % item[1][res]
            out += '\n'
        return out

    def get_residue(self, pos):
        if False:
            print('Hello World!')
        'Return the residue letter at the specified position.'
        return self.pssm[pos][0]

def print_info_content(summary_info, fout=None, rep_record=0):
    if False:
        for i in range(10):
            print('nop')
    '3 column output: position, aa in representative sequence, ic_vector value.'
    fout = fout or sys.stdout
    if not summary_info.ic_vector:
        summary_info.information_content()
    rep_sequence = summary_info.alignment[rep_record]
    for (pos, (aa, ic)) in enumerate(zip(rep_sequence, summary_info.ic_vector)):
        fout.write('%d %s %.3f\n' % (pos, aa, ic))