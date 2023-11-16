"""Print the results of restriction enzyme analysis.

PrintFormat prints the results from restriction analysis in 3 different
format: list, column or map.

The easiest way to use it is:

    >>> from Bio.Restriction.PrintFormat import PrintFormat
    >>> from Bio.Restriction.Restriction import RestrictionBatch
    >>> from Bio.Seq import Seq
    >>> pBs_mcs = Seq('GGTACCGGGCCCCCCCTCGAGGTCGACGGTATCGATAAGCTTGATATCGAATTC')
    >>> restriction_batch = RestrictionBatch(['EcoRI', 'BamHI', 'ApaI'])
    >>> result = restriction_batch.search(pBs_mcs)
    >>> my_map = PrintFormat()
    >>> my_map.print_that(result, 'My pBluescript mcs analysis:\\n',
    ...               'No site:\\n')
    My pBluescript mcs analysis:
    ApaI       :  12.
    EcoRI      :  50.
    No site:
    BamHI     
    <BLANKLINE>
    >>> my_map.sequence = pBs_mcs
    >>> my_map.print_as("map")
    >>> my_map.print_that(result)
               12 ApaI
               |                                                
               |                                     50 EcoRI
               |                                     |          
    GGTACCGGGCCCCCCCTCGAGGTCGACGGTATCGATAAGCTTGATATCGAATTC
    ||||||||||||||||||||||||||||||||||||||||||||||||||||||
    CCATGGCCCGGGGGGGAGCTCCAGCTGCCATAGCTATTCGAACTATAGCTTAAG
    1                                                   54
    <BLANKLINE>
    <BLANKLINE>
       Enzymes which do not cut the sequence.
    <BLANKLINE>
    BamHI     
    <BLANKLINE>
    >>>

Some of the methods of PrintFormat are meant to be overridden by derived
class.

Use the following parameters to control the appearance:

- ConsoleWidth : width of the console used default to 80.
                 should never be less than 60.
- NameWidth    : space attributed to the name in PrintList method.
- Indent       : Indent of the second line.
- MaxSize      : Maximal size of the sequence (default=6:
                 -> 99 999 bp + 1 trailing ','
                 people are unlikely to ask for restriction map of sequences
                 bigger than 100.000 bp. This is needed to determine the
                 space to be reserved for sites location.

                 - MaxSize = 5  =>   9.999 bp
                 - MaxSize = 6  =>  99.999 bp
                 - MaxSize = 7  => 999.999 bp

Example output::

    <------------ ConsoleWidth --------------->
    <- NameWidth ->
    EcoRI         :   1, 45, 50, 300, 400, 650,
                          700, 1200, 2500.
                      <-->
                        Indent

"""
import re

class PrintFormat:
    """PrintFormat allow the printing of results of restriction analysis."""
    ConsoleWidth = 80
    NameWidth = 10
    MaxSize = 6
    Cmodulo = ConsoleWidth % NameWidth
    PrefWidth = ConsoleWidth - Cmodulo
    Indent = 4
    linesize = PrefWidth - NameWidth

    def print_as(self, what='list'):
        if False:
            print('Hello World!')
        "Print the results as specified.\n\n        Valid format are:\n            'list'      -> alphabetical order\n            'number'    -> number of sites in the sequence\n            'map'       -> a map representation of the sequence with the sites.\n\n        If you want more flexibility over-ride the virtual method make_format.\n        "
        if what == 'map':
            self.make_format = self._make_map
        elif what == 'number':
            self.make_format = self._make_number
        else:
            self.make_format = self._make_list

    def format_output(self, dct, title='', s1=''):
        if False:
            i = 10
            return i + 15
        'Summarise results as a nicely formatted string.\n\n        Arguments:\n         - dct is a dictionary as returned by a RestrictionBatch.search()\n         - title is the title of the map.\n           It must be a formatted string, i.e. you must include the line break.\n         - s1 is the title separating the list of enzymes that have sites from\n           those without sites.\n         - s1 must be a formatted string as well.\n\n        The format of print_that is a list.\n        '
        if not dct:
            dct = self.results
        (ls, nc) = ([], [])
        for (k, v) in dct.items():
            if v:
                ls.append((k, v))
            else:
                nc.append(k)
        return self.make_format(ls, title, nc, s1)

    def print_that(self, dct, title='', s1=''):
        if False:
            return 10
        'Print the output of the format_output method (OBSOLETE).\n\n        Arguments:\n         - dct is a dictionary as returned by a RestrictionBatch.search()\n         - title is the title of the map.\n           It must be a formatted string, i.e. you must include the line break.\n         - s1 is the title separating the list of enzymes that have sites from\n           those without sites.\n         - s1 must be a formatted string as well.\n\n        This method prints the output of A.format_output() and it is here\n        for backwards compatibility.\n        '
        print(self.format_output(dct, title, s1))

    def make_format(self, cut=(), title='', nc=(), s1=''):
        if False:
            while True:
                i = 10
        'Virtual method used for formatting results.\n\n        Virtual method.\n        Here to be pointed to one of the _make_* methods.\n        You can as well create a new method and point make_format to it.\n        '
        return self._make_list(cut, title, nc, s1)

    def _make_list(self, ls, title, nc, s1):
        if False:
            print('Hello World!')
        'Summarise a list of positions by enzyme (PRIVATE).\n\n        Return a string of form::\n\n            title.\n\n            enzyme1     :   position1, position2.\n            enzyme2     :   position1, position2, position3.\n\n        Arguments:\n         - ls is a tuple or list of cutting enzymes.\n         - title is the title.\n         - nc is a tuple or list of non cutting enzymes.\n         - s1 is the sentence before the non cutting enzymes.\n        '
        return self._make_list_only(ls, title) + self._make_nocut_only(nc, s1)

    def _make_map(self, ls, title, nc, s1):
        if False:
            i = 10
            return i + 15
        'Summarise mapping information as a string (PRIVATE).\n\n        Return a string of form::\n\n            | title.\n            |\n            |     enzyme1, position\n            |     |\n            | AAAAAAAAAAAAAAAAAAAAA...\n            | |||||||||||||||||||||\n            | TTTTTTTTTTTTTTTTTTTTT...\n\n        Arguments:\n         - ls is a list of cutting enzymes.\n         - title is the title.\n         - nc is a list of non cutting enzymes.\n         - s1 is the sentence before the non cutting enzymes.\n        '
        return self._make_map_only(ls, title) + self._make_nocut_only(nc, s1)

    def _make_number(self, ls, title, nc, s1):
        if False:
            print('Hello World!')
        'Format cutting position information as a string (PRIVATE).\n\n        Returns a string in the form::\n\n            title.\n\n            enzyme which cut 1 time:\n\n            enzyme1     :   position1.\n\n            enzyme which cut 2 times:\n\n            enzyme2     :   position1, position2.\n            ...\n\n        Arguments:\n         - ls is a list of cutting enzymes.\n         - title is the title.\n         - nc is a list of non cutting enzymes.\n         - s1 is the sentence before the non cutting enzymes.\n        '
        return self._make_number_only(ls, title) + self._make_nocut_only(nc, s1)

    def _make_nocut(self, ls, title, nc, s1):
        if False:
            while True:
                i = 10
        'Summarise non-cutting enzymes (PRIVATE).\n\n        Return a formatted string of the non cutting enzymes.\n\n        ls is a list of cutting enzymes -> will not be used.\n        Here for compatibility with make_format.\n\n        Arguments:\n         - title is the title.\n         - nc is a list of non cutting enzymes.\n         - s1 is the sentence before the non cutting enzymes.\n        '
        return title + self._make_nocut_only(nc, s1)

    def _make_nocut_only(self, nc, s1, ls=(), title=''):
        if False:
            while True:
                i = 10
        'Summarise non-cutting enzymes (PRIVATE).\n\n        Return a formatted string of the non cutting enzymes.\n\n        Arguments:\n         - nc is a tuple or list of non cutting enzymes.\n         - s1 is the sentence before the non cutting enzymes.\n        '
        if not nc:
            return s1
        st = ''
        stringsite = s1 or '\n   Enzymes which do not cut the sequence.\n\n'
        Join = ''.join
        for key in sorted(nc):
            st = Join((st, str.ljust(str(key), self.NameWidth)))
            if len(st) > self.linesize:
                stringsite = Join((stringsite, st, '\n'))
                st = ''
        stringsite = Join((stringsite, st, '\n'))
        return stringsite

    def _make_list_only(self, ls, title, nc=(), s1=''):
        if False:
            return 10
        'Summarise list of positions per enzyme (PRIVATE).\n\n        Return a string of form::\n\n            title.\n\n            enzyme1     :   position1, position2.\n            enzyme2     :   position1, position2, position3.\n            ...\n\n        Arguments:\n         - ls is a tuple or list of results.\n         - title is a string.\n         - Non cutting enzymes are not included.\n        '
        if not ls:
            return title
        return self.__next_section(ls, title)

    def _make_number_only(self, ls, title, nc=(), s1=''):
        if False:
            while True:
                i = 10
        'Summarise number of cuts as a string (PRIVATE).\n\n        Return a string of form::\n\n            title.\n\n            enzyme which cut 1 time:\n\n            enzyme1     :   position1.\n\n            enzyme which cut 2 times:\n\n            enzyme2     :   position1, position2.\n            ...\n\n        Arguments:\n         - ls is a list of results.\n         - title is a string.\n         - Non cutting enzymes are not included.\n        '
        if not ls:
            return title
        ls.sort(key=lambda x: len(x[1]))
        iterator = iter(ls)
        cur_len = 1
        new_sect = []
        for (name, sites) in iterator:
            length = len(sites)
            if length > cur_len:
                title += '\n\nenzymes which cut %i times :\n\n' % cur_len
                title = self.__next_section(new_sect, title)
                (new_sect, cur_len) = ([(name, sites)], length)
                continue
            new_sect.append((name, sites))
        title += '\n\nenzymes which cut %i times :\n\n' % cur_len
        return self.__next_section(new_sect, title)

    def _make_map_only(self, ls, title, nc=(), s1=''):
        if False:
            while True:
                i = 10
        'Make string describing cutting map (PRIVATE).\n\n        Return a string of form::\n\n            | title.\n            |\n            |     enzyme1, position\n            |     |\n            | AAAAAAAAAAAAAAAAAAAAA...\n            | |||||||||||||||||||||\n            | TTTTTTTTTTTTTTTTTTTTT...\n\n        Arguments:\n         - ls is a list of results.\n         - title is a string.\n         - Non cutting enzymes are not included.\n        '
        if not ls:
            return title
        resultKeys = sorted((str(x) for (x, y) in ls))
        map = title or ''
        enzymemap = {}
        for (enzyme, cut) in ls:
            for c in cut:
                if c in enzymemap:
                    enzymemap[c].append(str(enzyme))
                else:
                    enzymemap[c] = [str(enzyme)]
        mapping = sorted(enzymemap.keys())
        cutloc = {}
        (x, counter, length) = (0, 0, len(self.sequence))
        for x in range(60, length, 60):
            counter = x - 60
            loc = []
            cutloc[counter] = loc
            remaining = []
            for key in mapping:
                if key <= x:
                    loc.append(key)
                else:
                    remaining.append(key)
            mapping = remaining
        cutloc[x] = mapping
        sequence = str(self.sequence)
        revsequence = str(self.sequence.complement(inplace=False))
        a = '|'
        (base, counter) = (0, 0)
        emptyline = ' ' * 60
        Join = ''.join
        for base in range(60, length, 60):
            counter = base - 60
            line = emptyline
            for key in cutloc[counter]:
                s = ''
                if key == base:
                    for n in enzymemap[key]:
                        s = ' '.join((s, n))
                    chunk = line[0:59]
                    lineo = Join((chunk, str(key), s, '\n'))
                    line2 = Join((chunk, a, '\n'))
                    linetot = Join((lineo, line2))
                    map = Join((map, linetot))
                    break
                for n in enzymemap[key]:
                    s = ' '.join((s, n))
                k = key % 60
                lineo = Join((line[0:k - 1], str(key), s, '\n'))
                line = Join((line[0:k - 1], a, line[k:]))
                line2 = Join((line[0:k - 1], a, line[k:], '\n'))
                linetot = Join((lineo, line2))
                map = Join((map, linetot))
            mapunit = '\n'.join((sequence[counter:base], a * 60, revsequence[counter:base], Join((str.ljust(str(counter + 1), 15), ' ' * 30, str.rjust(str(base), 15), '\n\n'))))
            map = Join((map, mapunit))
        line = ' ' * 60
        for key in cutloc[base]:
            s = ''
            if key == length:
                for n in enzymemap[key]:
                    s = Join((s, ' ', n))
                chunk = line[0:length - 1]
                lineo = Join((chunk, str(key), s, '\n'))
                line2 = Join((chunk, a, '\n'))
                linetot = Join((lineo, line2))
                map = Join((map, linetot))
                break
            for n in enzymemap[key]:
                s = Join((s, ' ', n))
            k = key % 60
            lineo = Join((line[0:k - 1], str(key), s, '\n'))
            line = Join((line[0:k - 1], a, line[k:]))
            line2 = Join((line[0:k - 1], a, line[k:], '\n'))
            linetot = Join((lineo, line2))
            map = Join((map, linetot))
        mapunit = ''
        mapunit = Join((sequence[base:length], '\n'))
        mapunit = Join((mapunit, a * (length - base), '\n'))
        mapunit = Join((mapunit, revsequence[base:length], '\n'))
        mapunit = Join((mapunit, Join((str.ljust(str(base + 1), 15), ' ' * (length - base - 30), str.rjust(str(length), 15), '\n\n'))))
        map = Join((map, mapunit))
        return map

    def __next_section(self, ls, into):
        if False:
            for i in range(10):
                print('nop')
        'Next section (PRIVATE).\n\n        Arguments:\n         - ls is a tuple/list of tuple (string, [int, int]).\n         - into is a string to which the formatted ls will be added.\n\n        Format ls as a string of lines:\n        The form is::\n\n            enzyme1     :   position1.\n            enzyme2     :   position2, position3.\n\n        then add the formatted ls to tot\n        return tot.\n        '
        indentation = '\n' + (self.NameWidth + self.Indent) * ' '
        linesize = self.linesize - self.MaxSize
        pat = re.compile('([\\w,\\s()]){1,%i}[,\\.]' % linesize)
        (several, Join) = ('', ''.join)
        for (name, sites) in sorted(ls):
            stringsite = ''
            output = Join((', '.join((str(site) for site in sites)), '.'))
            if len(output) > linesize:
                output = [x.group() for x in re.finditer(pat, output)]
                stringsite = indentation.join(output)
            else:
                stringsite = output
            into = Join((into, str(name).ljust(self.NameWidth), ' :  ', stringsite, '\n'))
        return into