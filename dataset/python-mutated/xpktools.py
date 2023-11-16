"""Tools to manipulate data from nmrview .xpk peaklist files."""
HEADERLEN = 6

class XpkEntry:
    """Provide dictionary access to single entry from nmrview .xpk file.

    This class is suited for handling single lines of non-header data
    from an nmrview .xpk file. This class provides methods for extracting
    data by the field name which is listed in the last line of the
    peaklist header.

    Parameters
    ----------
    xpkentry : str
        The line from an nmrview .xpk file.
    xpkheadline : str
        The line from the header file that gives the names of the entries.
        This is typically the sixth line of the header, 1-origin.

    Attributes
    ----------
    fields : dict
        Dictionary of fields where key is in header line, value is an entry.
        Variables are accessed by either their name in the header line as in
        self.field["H1.P"] will return the H1.P entry for example.
        self.field["entrynum"] returns the line number (1st field of line)

    """

    def __init__(self, entry, headline):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the class.'
        datlist = entry.split()
        headlist = headline.split()
        self.fields = dict(zip(headlist, datlist[1:]))
        try:
            self.fields['entrynum'] = datlist[0]
        except IndexError:
            pass

class Peaklist:
    """Provide access to header lines and data from a nmrview xpk file.

    Header file lines and file data are available as attributes.

    Parameters
    ----------
    infn : str
        The input nmrview filename.

    Attributes
    ----------
    firstline  : str
        The first line in the header.
    axislabels : str
        The axis labels.
    dataset    : str
        The label of the dataset.
    sw         : str
        The sw coordinates.
    sf         : str
        The sf coordinates.
    datalabels : str
        The labels of the entries.

    data : list
        File data after header lines.

    Examples
    --------
    >>> from Bio.NMR.xpktools import Peaklist
    >>> peaklist = Peaklist('../Doc/examples/nmr/noed.xpk')
    >>> peaklist.firstline
    'label dataset sw sf '
    >>> peaklist.dataset
    'test.nv'
    >>> peaklist.sf
    '{599.8230 } { 60.7860 } { 60.7860 }'
    >>> peaklist.datalabels
    ' H1.L  H1.P  H1.W  H1.B  H1.E  H1.J  15N2.L  15N2.P  15N2.W  15N2.B  15N2.E  15N2.J  N15.L  N15.P  N15.W  N15.B  N15.E  N15.J  vol  int  stat '

    """

    def __init__(self, infn):
        if False:
            while True:
                i = 10
        'Initialize the class.'
        with open(infn) as infile:
            self.firstline = infile.readline().split('\n')[0]
            self.axislabels = infile.readline().split('\n')[0]
            self.dataset = infile.readline().split('\n')[0]
            self.sw = infile.readline().split('\n')[0]
            self.sf = infile.readline().split('\n')[0]
            self.datalabels = infile.readline().split('\n')[0]
            self.data = [line.split('\n')[0] for line in infile]

    def residue_dict(self, index):
        if False:
            print('Hello World!')
        "Return a dict of lines in 'data' indexed by residue number or a nucleus.\n\n        The nucleus should be given as the input argument in the same form as\n        it appears in the xpk label line (H1, 15N for example)\n\n        Parameters\n        ----------\n        index : str\n            The nucleus to index data by.\n\n        Returns\n        -------\n        resdict : dict\n            Mappings of index nucleus to data line.\n\n        Examples\n        --------\n        >>> from Bio.NMR.xpktools import Peaklist\n        >>> peaklist = Peaklist('../Doc/examples/nmr/noed.xpk')\n        >>> residue_d = peaklist.residue_dict('H1')\n        >>> sorted(residue_d.keys())\n        ['10', '3', '4', '5', '6', '7', '8', '9', 'maxres', 'minres']\n        >>> residue_d['10']\n        ['8  10.hn   7.663   0.021   0.010   ++   0.000   10.n   118.341   0.324   0.010   +E   0.000   10.n   118.476   0.324   0.010   +E   0.000  0.49840 0.49840 0']\n\n        "
        maxres = -1
        minres = -1
        self.dict = {}
        for line in self.data:
            ind = XpkEntry(line, self.datalabels).fields[index + '.L']
            key = ind.split('.')[0]
            res = int(key)
            if maxres == -1:
                maxres = res
            if minres == -1:
                minres = res
            maxres = max([maxres, res])
            minres = min([minres, res])
            res = str(res)
            try:
                self.dict[res].append(line)
            except KeyError:
                self.dict[res] = [line]
        self.dict['maxres'] = maxres
        self.dict['minres'] = minres
        return self.dict

    def write_header(self, outfn):
        if False:
            for i in range(10):
                print('nop')
        'Write header lines from input file to handle ``outfn``.'
        with open(outfn, 'w') as outfile:
            outfile.write(self.firstline)
            outfile.write('\n')
            outfile.write(self.axislabels)
            outfile.write('\n')
            outfile.write(self.dataset)
            outfile.write('\n')
            outfile.write(self.sw)
            outfile.write('\n')
            outfile.write(self.sf)
            outfile.write('\n')
            outfile.write(self.datalabels)
            outfile.write('\n')

def replace_entry(line, fieldn, newentry):
    if False:
        while True:
            i = 10
    'Replace an entry in a string by the field number.\n\n    No padding is implemented currently.  Spacing will change if\n    the original field entry and the new field entry are of\n    different lengths.\n    '
    start = _find_start_entry(line, fieldn)
    leng = len(line[start:].split()[0])
    newline = line[:start] + str(newentry) + line[start + leng:]
    return newline

def _find_start_entry(line, n):
    if False:
        print('Hello World!')
    'Find the starting character for entry ``n`` in a space delimited ``line`` (PRIVATE).\n\n    n is counted starting with 1.\n    The n=1 field by definition begins at the first character.\n\n    Returns\n    -------\n    starting character : str\n        The starting character for entry ``n``.\n\n    '
    if n == 1:
        return 0
    c = 1
    leng = len(line)
    if line[0] == ' ':
        infield = False
        field = 0
    else:
        infield = True
        field = 1
    while c < leng and field < n:
        if infield:
            if line[c] == ' ' and line[c - 1] != ' ':
                infield = False
            elif line[c] != ' ':
                infield = True
                field += 1
        c += 1
    return c - 1

def data_table(fn_list, datalabel, keyatom):
    if False:
        return 10
    'Generate a data table from a list of input xpk files.\n\n    Parameters\n    ----------\n    fn_list : list\n        List of .xpk file names.\n    datalabel : str\n        The data element reported.\n    keyatom : str\n        The name of the nucleus used as an index for the data table.\n\n    Returns\n    -------\n    outlist : list\n       List of table rows indexed by ``keyatom``.\n\n    '
    outlist = []
    (dict_list, label_line_list) = _read_dicts(fn_list, keyatom)
    minr = dict_list[0]['minres']
    maxr = dict_list[0]['maxres']
    for dictionary in dict_list:
        if maxr < dictionary['maxres']:
            maxr = dictionary['maxres']
        if minr > dictionary['minres']:
            minr = dictionary['minres']
    res = minr
    while res <= maxr:
        count = 0
        key = str(res)
        line = key
        for dictionary in dict_list:
            label = label_line_list[count]
            if key in dictionary:
                line = line + '\t' + XpkEntry(dictionary[key][0], label).fields[datalabel]
            else:
                line += '\t*'
            count += 1
        line += '\n'
        outlist.append(line)
        res += 1
    return outlist

def _read_dicts(fn_list, keyatom):
    if False:
        while True:
            i = 10
    'Read multiple files into a list of residue dictionaries (PRIVATE).'
    dict_list = []
    datalabel_list = []
    for fn in fn_list:
        peaklist = Peaklist(fn)
        dictionary = peaklist.residue_dict(keyatom)
        dict_list.append(dictionary)
        datalabel_list.append(peaklist.datalabels)
    return [dict_list, datalabel_list]
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest()