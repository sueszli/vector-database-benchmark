"""Convert a series of Rebase files into a Restriction_Dictionary.py module.

The Rebase files are in the emboss format:
 - ``emboss_e.###`` - contains information about the restriction sites.
 - ``emboss_r.###`` - contains general information about the enzymes.
 - ``emboss_s.###`` - contains information about the suppliers.

Here ``###`` is the 3 digit number REBASE release number (e.g. 312). The first
digit is the last digit of the year (e.g. 3 for 2013) and the two last the
month (e.g. 12 for December).

These files are available by FTP from ftp://ftp.neb.com/pub/rebase/ which
should allow automated fetching via ``rebase_update.py``.
In addition there are links on this HTML page which allows manual download
and renaming of the files: http://rebase.neb.com/rebase/rebase.f37.html
This Python file is intended to be used via the scripts in
`Scripts/Restriction/*.py` only.
"""
import os
import itertools
import time
import sys
import shutil
import optparse
from Bio.Seq import Seq
from Bio.Data.IUPACData import ambiguous_dna_values as amb_dna
import Bio.Restriction.Restriction
from Bio.Restriction.Restriction import AbstractCut, RestrictionType, NoCut
from Bio.Restriction.Restriction import OneCut, TwoCuts, Meth_Dep, Meth_Undep
from Bio.Restriction.Restriction import Palindromic, NonPalindromic, Unknown
from Bio.Restriction.Restriction import Blunt, Ov5, Ov3
from Bio.Restriction.Restriction import NotDefined, Defined, Ambiguous
from Bio.Restriction.Restriction import Commercially_available, Not_available
from rebase_update import release_number, get_files
enzymedict = {}
suppliersdict = {}
classdict = {}
typedict = {}

def parse_enzyme_records(handle):
    if False:
        for i in range(10):
            print('nop')
    'Parse ENZYME records.\n\n    This function is for parsing ENZYME files containing multiple\n    records.\n\n    Arguments:\n     - handle   - handle to the file.\n\n    '
    while True:
        record = read_enzyme_record(handle)
        if not record:
            break
        yield record

def read_enzyme_record(handle):
    if False:
        print('Hello World!')
    'Read a single Enzyme record.\n\n    Enzyme record read format is adapted from Bio.ExPASy.Enzyme, but must be\n    able to read an accession field that is not used by Bio.ExPASy.Enzyme.\n    '
    record = None
    for line in handle:
        (key, value) = (line[:2], line[5:].rstrip())
        if key == 'ID':
            record = {'ID': value}
        elif key == 'AC':
            record['AC'] = value
        elif key == '//':
            if record:
                return record
            else:
                continue
    if record:
        raise ValueError('Unexpected end of stream')

def load_enzyme_ids(file) -> dict[str, int]:
    if False:
        print('Hello World!')
    'Load enzyme identifiers from bairoch-format file.'
    with open(file) as in_file:
        return {record['ID']: int(record['AC'].removeprefix('RB').removesuffix(';')) for record in parse_enzyme_records(in_file)}

def double_quote_repr(value):
    if False:
        print('Hello World!')
    'Return string representation of value, preferring double quotes.\n\n    Used to produce Python code with double quotes.\n\n    Only special cases strings, tuples and lists so far.\n    '
    if isinstance(value, str):
        if '"' not in value:
            return f'"{value}"'
    elif isinstance(value, tuple):
        if len(value) == 1:
            return '(%s,)' % double_quote_repr(list(value)[0])
        else:
            return '(%s)' % ', '.join((double_quote_repr(_) for _ in value))
    elif isinstance(value, list):
        return '[%s]' % ', '.join((double_quote_repr(_) for _ in value))
    return repr(value)

class OverhangError(ValueError):
    """Exception for dealing with overhang."""

def regex(site):
    if False:
        while True:
            i = 10
    "Construct a regular expression (string) from a DNA sequence.\n\n    Examples\n    --------\n    >>> regex('ABCGN')\n    'A[CGT]CG.'\n\n    "
    reg_ex = str(site)
    for base in reg_ex:
        if base in ('A', 'T', 'C', 'G', 'a', 'c', 'g', 't'):
            pass
        if base in ('N', 'n'):
            reg_ex = '.'.join(reg_ex.split('N'))
            reg_ex = '.'.join(reg_ex.split('n'))
        if base in ('R', 'Y', 'W', 'M', 'S', 'K', 'H', 'D', 'B', 'V'):
            expand = '[' + amb_dna[base.upper()] + ']'
            reg_ex = expand.join(reg_ex.split(base))
    return reg_ex

def is_palindrome(sequence):
    if False:
        while True:
            i = 10
    'Check whether the sequence is a palindrome or not.'
    return sequence == sequence.reverse_complement()

class newenzyme:
    """Construct the attributes of the enzyme corresponding to 'name'."""

    def __init__(cls, name, id):
        if False:
            while True:
                i = 10
        "Set up the enzyme's attributes."
        cls.id = id
        if id:
            cls.uri = f'https://identifiers.org/rebase:{cls.id}'
        else:
            cls.uri = None
        cls.opt_temp = 37
        cls.inact_temp = 65
        cls.substrat = 'DNA'
        target = enzymedict[name]
        cls.site = target[0]
        cls.size = target[1]
        cls.suppl = tuple(target[9])
        cls.freq = target[11]
        cls.ovhg = target[13]
        cls.ovhgseq = target[14]
        cls.bases = ()
        if target[10]:
            cls.bases += ('Palindromic',)
        else:
            cls.bases += ('NonPalindromic',)
        if not target[2]:
            cls.bases += ('NoCut', 'Unknown', 'NotDefined')
            cls.fst5 = None
            cls.fst3 = None
            cls.scd5 = None
            cls.scd3 = None
            cls.ovhg = None
            cls.ovhgseq = None
        else:
            if target[2] == 2:
                cls.bases += ('OneCut',)
                cls.fst5 = target[4]
                cls.fst3 = target[5]
                cls.scd5 = None
                cls.scd3 = None
            else:
                cls.bases += ('TwoCuts',)
                cls.fst5 = target[4]
                cls.fst3 = target[5]
                cls.scd5 = target[6]
                cls.scd3 = target[7]
            if target[3]:
                cls.bases += ('Blunt', 'Defined')
                cls.ovhg = 0
            elif isinstance(cls.ovhg, int):
                if cls.ovhg > 0:
                    cls.bases += ('Ov3', 'Ambiguous')
                elif cls.ovhg < 0:
                    cls.bases += ('Ov5', 'Ambiguous')
            elif cls.fst5 - (cls.fst3 + cls.size) < 0:
                cls.bases += ('Ov5', 'Defined')
                cls.ovhg = -len(cls.ovhg)
            else:
                cls.bases += ('Ov3', 'Defined')
                cls.ovhg = +len(cls.ovhg)
        if target[8]:
            cls.bases += ('Meth_Dep',)
            cls.compsite = target[12]
        else:
            cls.bases += ('Meth_Undep',)
            cls.compsite = target[12]
        if cls.suppl:
            cls.bases += ('Commercially_available',)
        else:
            cls.bases += ('Not_available',)
        cls.bases += ('AbstractCut', 'RestrictionType')
        cls.__name__ = name
        cls.results = None
        cls.dna = None
        cls.__bases__ = cls.bases
        cls.charac = (cls.fst5, cls.fst3, cls.scd5, cls.scd3, cls.site)
        if not target[2] and cls.suppl:
            supp = ', '.join((suppliersdict[s][0] for s in cls.suppl))
            print('WARNING : It seems that %s is both commercially available\n\tand its characteristics are unknown. \n\tThis seems counter-intuitive.\n\tThere is certainly an error either in ranacompiler or\n\tin this REBASE release.\n\tThe supplier is : %s.' % (name, supp))

class TypeCompiler:
    """Build the different types possible for Restriction Enzymes."""

    def __init__(self):
        if False:
            while True:
                i = 10
        'TypeCompiler() -> new TypeCompiler instance.'

    def buildtype(self):
        if False:
            while True:
                i = 10
        'Build new types that will be needed for constructing the enzymes.'
        baT = (AbstractCut, RestrictionType)
        cuT = (NoCut, OneCut, TwoCuts)
        meT = (Meth_Dep, Meth_Undep)
        paT = (Palindromic, NonPalindromic)
        ovT = (Unknown, Blunt, Ov5, Ov3)
        deT = (NotDefined, Defined, Ambiguous)
        coT = (Commercially_available, Not_available)
        types = [(p, c, o, d, m, co, baT[0], baT[1]) for p in paT for c in cuT for o in ovT for d in deT for m in meT for co in coT]
        n = 1
        for ty in types:
            dct = {}
            for t in ty:
                dct.update(t.__dict__)
                dct['results'] = []
                dct['substrat'] = 'DNA'
                dct['dna'] = None
                if t == NoCut:
                    dct.update({'fst5': None, 'fst3': None, 'scd5': None, 'scd3': None, 'ovhg': None, 'ovhgseq': None})
                elif t == OneCut:
                    dct.update({'scd5': None, 'scd3': None})

            class klass(type):
                """Dynamically defined restriction enzyme class."""

                def __new__(cls):
                    if False:
                        i = 10
                        return i + 15
                    return type.__new__(cls, f'type{n:d}', ty, dct)

                def __init__(cls):
                    if False:
                        while True:
                            i = 10
                    super().__init__(f'type{n:d}', ty, dct)
            yield klass()
            n += 1
start = f'#!/usr/bin/env python\n#      Copyright (C) 2004. Frederic Sohm.\n#\n# This code is part of the Biopython distribution and governed by its\n# license.  Please see the LICENSE file that should have been included\n# as part of this package.\n#\n# THIS FILE IS AUTOMATICALLY GENERATED - DO NOT EDIT IT BY HAND!\n#\n# Instead, use the tool Scripts/Restriction/ranacompiler.py which in turn\n# uses Scripts/Restriction/rebase_update.py\n\n"""Restriction Analysis Libraries.\n\nUsed REBASE emboss files version {release_number} ({time.gmtime().tm_year}).\n"""\n\n'

class DictionaryBuilder:
    """Builds ``Restriction_Dictionary.py`` from Rebase files.

    If the emboss files used for the construction need to be updated this
    class will download them if the ftp connection is correctly set.
    """

    def build_dict(self):
        if False:
            i = 10
            return i + 15
        'Construct dictionary and build files containing new dictionaries.'
        (emboss_e, emboss_r, emboss_s, enzyme_id_dict) = self.lastrebasefile()
        self.information_mixer(emboss_r, emboss_e, emboss_s)
        emboss_r.close()
        emboss_e.close()
        emboss_s.close()
        tdct = {}
        for klass in TypeCompiler().buildtype():
            exec(klass.__name__ + '= klass')
            exec("tdct['" + klass.__name__ + "'] = klass")
        for name in enzymedict:
            try:
                enzyme_id = enzyme_id_dict[name]
            except KeyError:
                print(f'Could not find REBASE enzyme ID for {name}: omitting')
                enzyme_id = None
            cls = newenzyme(name, enzyme_id)
            bases = cls.bases
            clsbases = tuple((eval(x) for x in bases))
            typestuff = ''
            for t in tdct.values():
                if t.__bases__ == clsbases:
                    typestuff = t
                    typename = t.__name__
                continue
            dct = dict(cls.__dict__)
            del dct['bases']
            del dct['__bases__']
            del dct['__name__']
            classdict[name] = dct
            commonattr = ['fst5', 'fst3', 'scd5', 'scd3', 'substrat', 'ovhg', 'ovhgseq', 'results', 'dna']
            if typename in typedict:
                typedict[typename][1].append(name)
            else:
                enzlst = []
                tydct = dict(typestuff.__dict__)
                tydct = {k: v for (k, v) in tydct.items() if k in commonattr}
                enzlst.append(name)
                typedict[typename] = (bases, enzlst)
            for letter in cls.__dict__['suppl']:
                suppliersdict[letter][1].append(name)
        if not classdict or not suppliersdict or (not typedict):
            print('One of the new dictionaries is empty.')
            print('Check the integrity of the emboss file before continuing.')
            print('Update aborted.')
            sys.exit()
        print('\nThe new database contains %i enzymes.\n' % len(classdict))
        update = os.getcwd()
        with open(os.path.join(update, 'Restriction_Dictionary.py'), 'w') as results:
            print('Writing the dictionary containing the new Restriction classes...')
            results.write(start)
            results.write('rest_dict = {}\n')
            results.write('\n')
            for name in sorted(classdict):
                results.write('rest_dict[%s] = {\n' % double_quote_repr(name))
                for (key, value) in sorted(classdict[name].items()):
                    results.write('    %s: %s,\n' % (double_quote_repr(key), double_quote_repr(value)))
                results.write('}\n\n')
            print('OK.\n')
            print('Writing the dictionary containing the suppliers data...')
            results.write('\n')
            results.write('# Turn black code style off\n# fmt: off\n')
            results.write('\n')
            results.write('suppliers = {}\n')
            results.write('\n')
            for name in sorted(suppliersdict):
                results.write('suppliers[%s] = (\n' % double_quote_repr(name))
                for value in suppliersdict[name]:
                    results.write('    %s,\n' % double_quote_repr(value))
                results.write(')\n\n')
            print('OK.\n')
            print('Writing the dictionary containing the Restriction types...')
            results.write('\n')
            results.write('typedict = {}\n')
            results.write('\n')
            for name in sorted(typedict):
                results.write('typedict[%s] = (\n' % double_quote_repr(name))
                for value in typedict[name]:
                    results.write('    %s,\n' % double_quote_repr(value))
                results.write(')\n\n')
            results.write('# Turn black code style on\n# fmt: on\n')
            print('OK.\n')

    def install_dict(self):
        if False:
            i = 10
            return i + 15
        'Install the newly created dictionary in the site-packages folder.\n\n        May need super user privilege on some architectures.\n        '
        print('\n ' + '*' * 78 + ' \n')
        print('\n\t\tInstalling Restriction_Dictionary.py')
        try:
            import Bio.Restriction.Restriction_Dictionary as rd
        except ImportError:
            print('\n Unable to locate the previous Restriction_Dictionary.py module\n Aborting installation.')
            sys.exit()
        old = os.path.join(os.path.split(rd.__file__)[0], 'Restriction_Dictionary.py')
        update_folder = os.getcwd()
        shutil.copyfile(old, os.path.join(update_folder, 'Restriction_Dictionary.old'))
        new = os.path.join(update_folder, 'Restriction_Dictionary.py')
        try:
            exec(compile(open(new).read(), new, 'exec'))
            print('\n\tThe new file seems ok. Proceeding with the installation.')
        except SyntaxError:
            sys.exit('ERROR: new dictionary file is corrupted. Aborting installation.')
        try:
            shutil.copyfile(new, old)
            print('\n\t Everything ok. If you need it a version of the old\n\t dictionary have been saved in the Updates folder under\n\t the name Restriction_Dictionary.old.')
            print('\n ' + '*' * 78 + ' \n')
        except OSError:
            print('\n ' + '*' * 78 + ' \n')
            print('\n\t WARNING : Impossible to install the new dictionary.\n\t Are you sure you have write permission to the folder :\n\n\t %s ?\n\n' % os.path.split(old)[0])
            return self.no_install()

    def no_install(self):
        if False:
            for i in range(10):
                print('nop')
        'Build the new dictionary but do not install the dictionary.'
        print('\n ' + '*' * 78 + '\n')
        try:
            import Bio.Restriction.Restriction_Dictionary as rd
        except ImportError:
            print('\n Unable to locate the previous Restriction_Dictionary.py module\n Aborting installation.')
            sys.exit()
        old = os.path.join(os.path.split(rd.__file__)[0], 'Restriction_Dictionary.py')
        update = os.getcwd()
        shutil.copyfile(old, os.path.join(update, 'Restriction_Dictionary.old'))
        places = (update, os.path.split(Bio.Restriction.Restriction.__file__)[0])
        print("\t\tCompilation of the new dictionary : OK.\n\t\tInstallation : No.\n\n You will find the newly created 'Restriction_Dictionary.py' file\n in the folder : \n\n\t%s\n\n Make a copy of 'Restriction_Dictionary.py' and place it with \n the other Restriction libraries.\n\n note : \n This folder should be :\n\n\t%s\n" % places)
        print('\n ' + '*' * 78 + '\n')

    def lastrebasefile(self):
        if False:
            print('Hello World!')
        'Check the emboss files are up to date and download them if not.'
        embossnames = ('emboss_e', 'emboss_r', 'emboss_s')
        emboss_now = ['.'.join((x, release_number)) for x in embossnames]
        bairoch_now = f'bairoch.{release_number}'
        update_needed = False
        dircontent = os.listdir(os.getcwd())
        base = os.getcwd()
        for name in emboss_now + [bairoch_now]:
            if name not in dircontent:
                update_needed = True
        if not update_needed:
            print('\n Using the bairoch file : %s' % bairoch_now)
            enzyme_id_dict = load_enzyme_ids(bairoch_now)
            print('\n Using the emboss files : %s' % ', '.join(emboss_now))
            return tuple((open(os.path.join(base, n)) for n in emboss_now)) + (enzyme_id_dict,)
        else:
            print('\n The rebase files are missing or more than one month old.\n Would you like to update them before proceeding?(y/n)')
            r = input(' update [n] >>> ')
            if r in ['y', 'yes', 'Y', 'Yes']:
                get_files()
                print('\n Update complete. Creating the dictionaries.\n')
                print('\n Using the files : %s' % ', '.join(emboss_now))
                return tuple((open(os.path.join(base, n)) for n in emboss_now))
            else:

                class NotFoundError(Exception):
                    pass
                for name in embossnames:
                    try:
                        for file in dircontent:
                            if file.startswith(name):
                                break
                        else:
                            raise NotFoundError
                    except NotFoundError:
                        print(f'\nNo {name} file found. Upgrade is impossible.\n')
                        sys.exit()
                    continue
        last = [0]
        for file in dircontent:
            fs = file.split('.')
            try:
                if fs[0] in embossnames and int(fs[1]) > int(last[-1]):
                    if last[0]:
                        last.append(fs[1])
                    else:
                        last[0] = fs[1]
                else:
                    continue
            except ValueError:
                continue
        last.sort()
        last = last[::-1]
        if int(last[-1]) < 100:
            (last[0], last[-1]) = (last[-1], last[0])
        for number in last:
            files = [name + f'.{number}' for name in embossnames]
            strmess = '\nLast EMBOSS files found are :\n'
            try:
                for file in files:
                    if os.path.isfile(os.path.join(base, file)):
                        strmess += f'\t{file}.\n'
                    else:
                        raise ValueError
                print(strmess)
                emboss_e = open(os.path.join(base, f'emboss_e.{number}'))
                emboss_r = open(os.path.join(base, f'emboss_r.{number}'))
                emboss_s = open(os.path.join(base, f'emboss_s.{number}'))
                return (emboss_e, emboss_r, emboss_s)
            except ValueError:
                continue

    def parseline(self, line):
        if False:
            print('Hello World!')
        'Parse a line from the Rebase emboss_e.xxx file.'
        line = [line[0]] + [line[1].upper()] + [int(i) for i in line[2:9]] + line[9:]
        name = line[0].replace('-', '_').replace('.', '_')
        site = line[1]
        dna = Seq(site)
        size = line[2]
        fst5 = line[5]
        fst3 = line[6]
        scd5 = line[7]
        scd3 = line[8]
        ovhg1 = fst5 - fst3
        ovhg2 = scd5 - scd3
        if fst5 < 0:
            fst5 += 1
        if fst3 < 0:
            fst3 += 1
        if scd5 < 0:
            scd5 += 1
        if scd3 < 0:
            scd3 += 1
        if ovhg2 != 0 and ovhg1 != ovhg2:
            print('\nWARNING : %s cut twice with different overhang length each time.\n\tUnable to deal with this behaviour. \n\tThis enzyme will not be included in the database. Sorry.' % name)
            print('\tChecking...')
            raise OverhangError
        if 0 <= fst5 <= size and 0 <= fst3 <= size:
            if fst5 < fst3:
                ovhg1 = ovhgseq = site[fst5:fst3]
            elif fst5 > fst3:
                ovhg1 = ovhgseq = site[fst3:fst5]
            else:
                ovhg1 = ovhgseq = ''
            for base in 'NRYWMSKHDBV':
                if base in ovhg1:
                    ovhgseq = ovhg1
                    if fst5 < fst3:
                        ovhg1 = -len(ovhg1)
                    else:
                        ovhg1 = len(ovhg1)
                    break
                else:
                    continue
        elif 0 <= fst5 <= size:
            if fst5 < fst3:
                ovhgseq = site[fst5:] + (fst3 - size) * 'N'
            elif fst5 > fst3:
                ovhgseq = abs(fst3) * 'N' + site[:fst5]
            else:
                ovhg1 = ovhgseq = ''
        elif 0 <= fst3 <= size:
            if fst5 < fst3:
                ovhgseq = abs(fst5) * 'N' + site[:fst3]
            elif fst5 > fst3:
                ovhgseq = site[fst3:] + (fst5 - size) * 'N'
            else:
                raise ValueError('Error in #1')
        elif fst3 < 0 and size < fst5:
            ovhgseq = abs(fst3) * 'N' + site + (fst5 - size) * 'N'
        elif fst5 < 0 and size < fst3:
            ovhgseq = abs(fst5) * 'N' + site + (fst3 - size) * 'N'
        else:
            ovhgseq = 'N' * abs(ovhg1)
        for x in (5, 7):
            if line[x] < 0:
                line[x] += 1
        for x in (6, 8):
            if line[x] > 0:
                line[x] -= size
            elif line[x] < 0:
                line[x] = line[x] - size + 1
        rg = ''
        if is_palindrome(dna):
            line.append(True)
            rg = ''.join(['(?=(?P<', name, '>', regex(site.upper()), '))'])
        else:
            line.append(False)
            sense = ''.join(['(?=(?P<', name, '>', regex(site.upper()), '))'])
            antisense = ''.join(['(?=(?P<', name, '_as>', regex(dna.reverse_complement()), '))'])
            rg = sense + '|' + antisense
        freq = 1
        for base in site.upper():
            freq *= 4.0 / len(amb_dna[base])
        line.append(freq)
        line.append(rg)
        line.append(ovhg1)
        line.append(ovhgseq)
        return line

    def removestart(self, file):
        if False:
            while True:
                i = 10
        'Remove the header of the file.'
        return list(itertools.dropwhile(lambda line: line.startswith('#'), file))

    def getblock(self, file, index):
        if False:
            i = 10
            return i + 15
        'Get a data block from the emboss_r file.'
        take = itertools.takewhile
        block = list(take(lambda line: not line.startswith('//'), file[index:]))
        index += len(block) + 1
        return (block, index)

    def get(self, block):
        if False:
            while True:
                i = 10
        'Get name, methylation information and suppliers.'
        bl3 = block[3].strip()
        if not bl3:
            bl3 = False
        return (block[0].strip(), bl3, block[5].strip())

    def information_mixer(self, file1, file2, file3):
        if False:
            return 10
        'Combine extracted data from the three emboss_x.xxx files.'
        methfile = self.removestart(file1)
        sitefile = self.removestart(file2)
        supplier = self.removestart(file3)
        (i1, i2) = (0, 0)
        oldblock = None
        try:
            while True:
                (block, i1) = self.getblock(methfile, i1)
                bl = self.get(block)
                line = sitefile[i2].strip().split()
                name = line[0]
                if name == bl[0]:
                    line.append(bl[1])
                    line.append(bl[2])
                else:
                    bl = self.get(oldblock)
                    if line[0] == bl[0]:
                        line.append(bl[1])
                        line.append(bl[2])
                        i2 += 1
                    else:
                        raise TypeError
                oldblock = block
                i2 += 1
                try:
                    line = self.parseline(line)
                except OverhangError:
                    n = name
                    if not bl[2]:
                        print(f'Anyway, {n} is not commercially available.\n')
                    else:
                        print(f'Unfortunately, {n} is commercially available.\n')
                    continue
                name = name.replace('-', '_').replace('.', '_')
                if name in enzymedict:
                    print(f'\nWARNING : {name} has two different sites.\n')
                    other = line[0].replace('-', '_').replace('.', '_')
                    dna = Seq(line[1])
                    sense1 = regex(dna)
                    antisense1 = regex(dna.reverse_complement())
                    dna = Seq(enzymedict[other][0])
                    sense2 = regex(dna)
                    antisense2 = regex(dna.reverse_complement())
                    sense = f'(?=(?P<{other}>{sense1})|{sense2})'
                    antisense = f'(?=(?P<{other}_as>{antisense1}|{antisense2}))'
                    reg = sense + '|' + antisense
                    line[1] = line[1] + '|' + enzymedict[other][0]
                    line[-1] = reg
                enzymedict[name] = line[1:]
        except IndexError:
            pass
        for i in supplier:
            t = i.strip().split(' ', 1)
            suppliersdict[t[0]] = (t[1], [])

def standalone():
    if False:
        i = 10
        return i + 15
    'Set up for running as main.'
    parser = optparse.OptionParser()
    add = parser.add_option
    add('-i', '--install', action='store_true', dest='i', default=False, help='compile and install the newly created file. default behaviour (without switch): Compile the enzymes and store them in the Updates folder')
    (options, args) = parser.parse_args()
    return (options, args)
if __name__ == '__main__':
    (options, args) = standalone()
    Builder = DictionaryBuilder()
    Builder.build_dict()
    if options.i:
        Builder.install_dict()
    else:
        Builder.no_install()
    sys.exit()