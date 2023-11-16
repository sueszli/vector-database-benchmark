"""Code to parse BIG GenePop files.

The difference between this class and the standard Bio.PopGen.GenePop.Record
class is that this one does not read the whole file to memory.
It provides an iterator interface, slower but consuming much mess memory.
Should be used with big files (Thousands of markers and individuals).

See http://wbiomed.curtin.edu.au/genepop/ , the format is documented
here: http://wbiomed.curtin.edu.au/genepop/help_input.html .

Classes:
 - FileRecord           Holds GenePop data.

Functions:


"""
from Bio.PopGen.GenePop import get_indiv

def read(fname):
    if False:
        while True:
            i = 10
    'Parse a file containing a GenePop file.\n\n    fname is a file name that contains a GenePop record.\n    '
    record = FileRecord(fname)
    return record

class FileRecord:
    """Hold information from a GenePop record.

    Attributes:
    - marker_len         The marker length (2 or 3 digit code per allele).
    - comment_line       Comment line.
    - loci_list          List of loci names.

    Methods:
    - get_individual     Returns the next individual of the current population.
    - skip_population    Skips the current population.

    skip_population skips the individuals of the current population, returns
    True if there are more populations.

    get_individual returns an individual of the current population (or None
    if the list ended).

    Each individual is a pair composed by individual name and a list of alleles
    (2 per marker or 1 for haploid data). Examples::

        ('Ind1', [(1,2),    (3,3), (200,201)]
        ('Ind2', [(2,None), (3,3), (None,None)]
        ('Other1', [(1,1),  (4,3), (200,200)]

    """

    def __init__(self, fname):
        if False:
            while True:
                i = 10
        'Initialize the class.'
        self.comment_line = ''
        self.loci_list = []
        self.fname = fname
        self.start_read()

    def __str__(self):
        if False:
            print('Hello World!')
        'Return (reconstructs) a GenePop textual representation.\n\n        This might take a lot of memory.\n        Marker length will be 3.\n        '
        marker_len = 3
        rep = [self.comment_line + '\n']
        rep.append('\n'.join(self.loci_list) + '\n')
        current_pop = self.current_pop
        current_ind = self.current_ind
        self._handle.seek(0)
        self.skip_header()
        rep.append('Pop\n')
        more = True
        while more:
            res = self.get_individual()
            if res is True:
                rep.append('Pop\n')
            elif res is False:
                more = False
            else:
                (name, markers) = res
                rep.append(name)
                rep.append(',')
                for marker in markers:
                    rep.append(' ')
                    for al in marker:
                        if al is None:
                            al = '0'
                        aStr = str(al)
                        while len(aStr) < marker_len:
                            aStr = ''.join(['0', aStr])
                        rep.append(aStr)
                rep.append('\n')
        self.seek_position(current_pop, current_ind)
        return ''.join(rep)

    def start_read(self):
        if False:
            print('Hello World!')
        'Start parsing a file containing a GenePop file.'
        self._handle = open(self.fname)
        self.comment_line = self._handle.readline().rstrip()
        sample_loci_line = self._handle.readline().rstrip().replace(',', '')
        all_loci = sample_loci_line.split(' ')
        self.loci_list.extend(all_loci)
        for line in self._handle:
            line = line.rstrip()
            if line.upper() == 'POP':
                break
            self.loci_list.append(line)
        else:
            raise ValueError('No population data found, file probably not GenePop related')
        self.current_pop = 0
        self.current_ind = 0

    def skip_header(self):
        if False:
            while True:
                i = 10
        'Skip the Header. To be done after a re-open.'
        self.current_pop = 0
        self.current_ind = 0
        for line in self._handle:
            if line.rstrip().upper() == 'POP':
                return

    def seek_position(self, pop, indiv):
        if False:
            i = 10
            return i + 15
        'Seek a certain position in the file.\n\n        Arguments:\n         - pop - pop position (0 is first)\n         - indiv - individual in pop\n\n        '
        self._handle.seek(0)
        self.skip_header()
        while pop > 0:
            self.skip_population()
            pop -= 1
        while indiv > 0:
            self.get_individual()
            indiv -= 1

    def skip_population(self):
        if False:
            print('Hello World!')
        'Skip the current population. Returns true if there is another pop.'
        for line in self._handle:
            if line == '':
                return False
            line = line.rstrip()
            if line.upper() == 'POP':
                self.current_pop += 1
                self.current_ind = 0
                return True

    def get_individual(self):
        if False:
            print('Hello World!')
        'Get the next individual.\n\n        Returns individual information if there are more individuals\n        in the current population.\n        Returns True if there are no more individuals in the current\n        population, but there are more populations. Next read will\n        be of the following pop.\n        Returns False if at end of file.\n        '
        for line in self._handle:
            line = line.rstrip()
            if line.upper() == 'POP':
                self.current_pop += 1
                self.current_ind = 0
                return True
            else:
                self.current_ind += 1
                (indiv_name, allele_list, ignore) = get_indiv(line)
                return (indiv_name, allele_list)
        return False

    def remove_population(self, pos, fname):
        if False:
            return 10
        'Remove a population (by position).\n\n        Arguments:\n         - pos - position\n         - fname - file to be created with population removed\n\n        '
        old_rec = read(self.fname)
        with open(fname, 'w') as f:
            f.write(self.comment_line + '\n')
            for locus in old_rec.loci_list:
                f.write(locus + '\n')
            curr_pop = 0
            l_parser = old_rec.get_individual()
            start_pop = True
            while l_parser:
                if curr_pop == pos:
                    old_rec.skip_population()
                    curr_pop += 1
                elif l_parser is True:
                    curr_pop += 1
                    start_pop = True
                else:
                    if start_pop:
                        f.write('POP\n')
                        start_pop = False
                    (name, markers) = l_parser
                    f.write(name + ',')
                    for marker in markers:
                        f.write(' ')
                        for al in marker:
                            if al is None:
                                al = '0'
                            aStr = str(al)
                            while len(aStr) < 3:
                                aStr = ''.join(['0', aStr])
                            f.write(aStr)
                    f.write('\n')
                l_parser = old_rec.get_individual()

    def remove_locus_by_position(self, pos, fname):
        if False:
            i = 10
            return i + 15
        'Remove a locus by position.\n\n        Arguments:\n         - pos - position\n         - fname - file to be created with locus removed\n\n        '
        old_rec = read(self.fname)
        with open(fname, 'w') as f:
            f.write(self.comment_line + '\n')
            loci_list = old_rec.loci_list
            del loci_list[pos]
            for locus in loci_list:
                f.write(locus + '\n')
            l_parser = old_rec.get_individual()
            f.write('POP\n')
            while l_parser:
                if l_parser is True:
                    f.write('POP\n')
                else:
                    (name, markers) = l_parser
                    f.write(name + ',')
                    marker_pos = 0
                    for marker in markers:
                        if marker_pos == pos:
                            marker_pos += 1
                            continue
                        marker_pos += 1
                        f.write(' ')
                        for al in marker:
                            if al is None:
                                al = '0'
                            aStr = str(al)
                            while len(aStr) < 3:
                                aStr = ''.join(['0', aStr])
                            f.write(aStr)
                    f.write('\n')
                l_parser = old_rec.get_individual()

    def remove_loci_by_position(self, positions, fname):
        if False:
            i = 10
            return i + 15
        'Remove a set of loci by position.\n\n        Arguments:\n         - positions - positions\n         - fname - file to be created with locus removed\n\n        '
        old_rec = read(self.fname)
        with open(fname, 'w') as f:
            f.write(self.comment_line + '\n')
            loci_list = old_rec.loci_list
            positions.sort()
            positions.reverse()
            posSet = set()
            for pos in positions:
                del loci_list[pos]
                posSet.add(pos)
            for locus in loci_list:
                f.write(locus + '\n')
            l_parser = old_rec.get_individual()
            f.write('POP\n')
            while l_parser:
                if l_parser is True:
                    f.write('POP\n')
                else:
                    (name, markers) = l_parser
                    f.write(name + ',')
                    marker_pos = 0
                    for marker in markers:
                        if marker_pos in posSet:
                            marker_pos += 1
                            continue
                        marker_pos += 1
                        f.write(' ')
                        for al in marker:
                            if al is None:
                                al = '0'
                            aStr = str(al)
                            while len(aStr) < 3:
                                aStr = ''.join(['0', aStr])
                            f.write(aStr)
                    f.write('\n')
                l_parser = old_rec.get_individual()

    def remove_locus_by_name(self, name, fname):
        if False:
            return 10
        'Remove a locus by name.\n\n        Arguments:\n         - name - name\n         - fname - file to be created with locus removed\n\n        '
        for (i, locus) in enumerate(self.loci_list):
            if locus == name:
                self.remove_locus_by_position(i, fname)
                return

    def remove_loci_by_name(self, names, fname):
        if False:
            print('Hello World!')
        'Remove a loci list (by name).\n\n        Arguments:\n         - names - names\n         - fname - file to be created with loci removed\n\n        '
        positions = []
        for (i, locus) in enumerate(self.loci_list):
            if locus in names:
                positions.append(i)
        self.remove_loci_by_position(positions, fname)