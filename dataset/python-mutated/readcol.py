"""
Taken from
From: https://github.com/keflavich/agpy/blob/master/agpy/readcol.py
License: Copyright (c) 2009 Adam Ginsburg

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

readcol.py by Adam Ginsburg (keflavich@gmail.com)

readcol is meant to emulate IDL's readcol.pro, but is more capable and
flexible.  It is not a particularly "pythonic" program since it is not modular.
For a modular ascii table reader, http://cxc.harvard.edu/contrib/asciitable/ is
probably better.  This single-function code is probably more intuitive to an
end-user, though.
"""
import string, re, sys
import numpy
from collections import OrderedDict
try:
    from scipy.stats import mode
    hasmode = True
except ImportError:
    hasmode = False
except ValueError:
    hasmode = False

def readcol(filename, skipline=0, skipafter=0, names=False, fsep=None, twod=True, fixedformat=None, asdict=False, comment='#', verbose=True, nullval=None, asStruct=False, namecomment=True, removeblanks=False, header_badchars=None, asRecArray=False):
    if False:
        print('Hello World!')
    '\n    The default return is a two dimensional float array.  If you want a list of\n    columns output instead of a 2D array, pass \'twod=False\'.  In this case,\n    each column\'s data type will be automatically detected.\n    \n    Example usage:\n    CASE 1) a table has the format:\n     X    Y    Z\n    0.0  2.4  8.2\n    1.0  3.4  5.6\n    0.7  3.2  2.1\n    ...\n    names,(x,y,z)=readcol("myfile.tbl",names=True,twod=False)\n    or\n    x,y,z=readcol("myfile.tbl",skipline=1,twod=False)\n    or \n    names,xx = readcol("myfile.tbl",names=True)\n    or\n    xxdict = readcol("myfile.tbl",asdict=True)\n    or\n    xxstruct = readcol("myfile.tbl",asStruct=True)\n\n    CASE 2) no title is contained into the table, then there is\n    no need to skipline:\n    x,y,z=readcol("myfile.tbl")\n    \n    CASE 3) there is a names column and then more descriptive text:\n     X      Y     Z\n    (deg) (deg) (km/s) \n    0.0    2.4   8.2\n    1.0    3.4.  5.6\n    ...\n    then use:\n    names,x,y,z=readcol("myfile.tbl",names=True,skipline=1,twod=False)\n    or\n    x,y,z=readcol("myfile.tbl",skipline=2,twod=False)\n\n    INPUTS:\n        fsep - field separator, e.g. for comma separated value (csv) files\n        skipline - number of lines to ignore at the start of the file\n        names - read / don\'t read in the first line as a list of column names\n                can specify an integer line number too, though it will be \n                the line number after skipping lines\n        twod - two dimensional or one dimensional output\n        nullval - if specified, all instances of this value will be replaced\n           with a floating NaN\n        asdict - zips names with data to create a dict with column headings \n            tied to column data.  If asdict=True, names will be set to True\n        asStruct - same as asdict, but returns a structure instead of a dictionary\n            (i.e. you call struct.key instead of struct[\'key\'])\n        fixedformat - if you have a fixed format file, this is a python list of \n            column lengths.  e.g. the first table above would be [3,5,5].  Note\n            that if you specify the wrong fixed format, you will get junk; if your\n            format total is greater than the line length, the last entries will all\n            be blank but readcol will not report an error.\n        namecomment - assumed that "Name" row is on a comment line.  If it is not - \n            e.g., it is the first non-comment line, change this to False\n        removeblanks - remove all blank entries from split lines.  This can cause lost\n            data if you have blank entries on some lines.\n        header_badchars - remove these characters from a header before parsing it\n            (helpful for IPAC tables that are delimited with | )\n\n    If you get this error: "scipy could not be imported.  Your table must have\n    full rows." it means readcol cannot automatically guess which columns\n    contain data.  If you have scipy and columns of varying length, readcol will\n    read in all of the rows with length=mode(row lengths).\n    '
    with open(filename, 'r') as f:
        f = f.readlines()
        null = [f.pop(0) for i in range(skipline)]
        commentfilter = make_commentfilter(comment)
        if not asStruct:
            asStruct = asRecArray
        if namecomment is False and (names or asdict or asStruct):
            while 1:
                line = f.pop(0)
                if line[0] != comment:
                    nameline = line
                    if header_badchars:
                        for c in header_badchars:
                            nameline = nameline.replace(c, ' ')
                    nms = nameline.split(fsep)
                    break
                elif len(f) == 0:
                    raise Exception('No uncommented lines found.')
        elif names or asdict or asStruct:
            if type(names) == type(1):
                nameline = f.pop(names)
            else:
                nameline = f.pop(0)
            if nameline[0] == comment:
                nameline = nameline[1:]
            if header_badchars:
                for c in header_badchars:
                    nameline = nameline.replace(c, ' ')
            nms = list([name.strip() for name in nameline.split(fsep)])
        null = [f.pop(0) for i in range(skipafter)]
        if fixedformat:
            myreadff = lambda x: readff(x, fixedformat)
            splitarr = list(map(myreadff, f))
            splitarr = list(filter(commentfilter, splitarr))
        else:
            fstrip = list(map(str.strip, f))
            fseps = [fsep for i in range(len(f))]
            splitarr = list(map(str.split, fstrip, fseps))
            if removeblanks:
                for i in range(splitarr.count([''])):
                    splitarr.remove([''])
            splitarr = list(filter(commentfilter, splitarr))
            nperline = list(map(len, splitarr))
            if hasmode:
                (ncols, nrows) = mode(nperline)
                if nrows != len(splitarr):
                    if verbose:
                        print("Removing %i rows that don't match most common length %i.                           \n%i rows read into array." % (len(splitarr) - nrows, ncols, nrows))
                    for i in range(len(splitarr) - 1, -1, -1):
                        if nperline[i] != ncols:
                            splitarr.pop(i)
        try:
            x = numpy.asarray(splitarr, dtype='float')
        except ValueError:
            if verbose:
                print('WARNING: reading as string array because %s array failed' % 'float')
            try:
                x = numpy.asarray(splitarr, dtype='S')
            except ValueError:
                if hasmode:
                    raise Exception('ValueError when converting data to array.' + '  You have scipy.mode on your system, so this is ' + 'probably not an issue of differing row lengths.')
                else:
                    raise Exception('Conversion to array error.  You probably ' + 'have different row lengths and scipy.mode was not ' + 'imported.')
        if nullval is not None:
            x[x == nullval] = numpy.nan
            x = get_autotype(x)
        if asdict or asStruct:
            mydict = OrderedDict(zip(nms, x.T))
            for (k, v) in mydict.items():
                mydict[k] = get_autotype(v)
            if asdict:
                return mydict
            elif asRecArray:
                return Struct(mydict).as_recarray()
            elif asStruct:
                return Struct(mydict)
        elif names and twod:
            return (nms, x)
        elif names:
            return (nms, [get_autotype(x.T[i]) for i in range(x.shape[1])])
        elif twod:
            return x
        else:
            return [get_autotype(x.T[i]) for i in range(x.shape[1])]

def get_autotype(arr):
    if False:
        for i in range(10):
            print('nop')
    '\n    Attempts to return a numpy array converted to the most sensible dtype\n    Value errors will be caught and simply return the original array\n    Tries to make dtype int, then float, then no change\n    '
    try:
        narr = arr.astype('float')
        if (narr < sys.maxsize).all() and (narr % 1).sum() == 0:
            return narr.astype('int')
        else:
            return narr
    except ValueError:
        return arr

class Struct(object):
    """
    Simple struct intended to take a dictionary of column names -> columns
    and turn it into a struct by removing special characters
    """

    def __init__(self, namedict):
        if False:
            return 10
        R = re.compile('\\W')
        for k in namedict.keys():
            v = namedict.pop(k)
            if k[0].isdigit():
                k = 'n' + k
            namedict[R.sub('', k)] = v
        self.__dict__ = namedict

    def add_column(self, name, data):
        if False:
            while True:
                i = 10
        '\n        Add a new column (attribute) to the struct\n        (will overwrite anything with the same name)\n        '
        self.__dict__[name] = data

    def as_recarray(self):
        if False:
            for i in range(10):
                print('nop')
        ' Convert into numpy recordarray '
        dtype = [(k, v.dtype) for (k, v) in self.__dict__.iteritems()]
        R = numpy.recarray(len(self.__dict__[k]), dtype=dtype)
        for key in self.__dict__:
            R[key] = self.__dict__[key]
        return R

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        return self.__dict__[key]

def readff(s, format):
    if False:
        i = 10
        return i + 15
    '\n    Fixed-format reader\n    Pass in a single line string (s) and a format list, \n    which needs to be a python list of string lengths \n    '
    F = numpy.array([0] + format).cumsum()
    bothF = zip(F[:-1], F[1:])
    strarr = [s[l:u] for (l, u) in bothF]
    return strarr

def make_commentfilter(comment):
    if False:
        for i in range(10):
            print('nop')
    if comment is not None:

        def commentfilter(a):
            if False:
                print('Hello World!')
            try:
                return comment.find(a[0][0])
            except:
                return -1
        return commentfilter
    else:
        return lambda x: -1