"""
Process f2py template files (`filename.pyf.src` -> `filename.pyf`)

Usage: python generate_pyf.py filename.pyf.src -o filename.pyf
"""
import os
import sys
import re
import subprocess
import argparse
"\nprocess_file(filename)\n\n  takes templated file .xxx.src and produces .xxx file where .xxx\n  is .pyf .f90 or .f using the following template rules:\n\n  '<..>' denotes a template.\n\n  All function and subroutine blocks in a source file with names that\n  contain '<..>' will be replicated according to the rules in '<..>'.\n\n  The number of comma-separated words in '<..>' will determine the number of\n  replicates.\n\n  '<..>' may have two different forms, named and short. For example,\n\n  named:\n   <p=d,s,z,c> where anywhere inside a block '<p>' will be replaced with\n   'd', 's', 'z', and 'c' for each replicate of the block.\n\n   <_c>  is already defined: <_c=s,d,c,z>\n   <_t>  is already defined: <_t=real,double precision,complex,double complex>\n\n  short:\n   <s,d,c,z>, a short form of the named, useful when no <p> appears inside\n   a block.\n\n  In general, '<..>' contains a comma separated list of arbitrary\n  expressions. If these expression must contain a comma|leftarrow|rightarrow,\n  then prepend the comma|leftarrow|rightarrow with a backslash.\n\n  If an expression matches '\\<index>' then it will be replaced\n  by <index>-th expression.\n\n  Note that all '<..>' forms in a block must have the same number of\n  comma-separated entries.\n\n Predefined named template rules:\n  <prefix=s,d,c,z>\n  <ftype=real,double precision,complex,double complex>\n  <ftypereal=real,double precision,\\0,\\1>\n  <ctype=float,double,complex_float,complex_double>\n  <ctypereal=float,double,\\0,\\1>\n"
routine_start_re = re.compile('(\\n|\\A)((     (\\$|\\*))|)\\s*(subroutine|function)\\b', re.I)
routine_end_re = re.compile('\\n\\s*end\\s*(subroutine|function)\\b.*(\\n|\\Z)', re.I)
function_start_re = re.compile('\\n     (\\$|\\*)\\s*function\\b', re.I)

def parse_structure(astr):
    if False:
        print('Hello World!')
    ' Return a list of tuples for each function or subroutine each\n    tuple is the start and end of a subroutine or function to be\n    expanded.\n    '
    spanlist = []
    ind = 0
    while True:
        m = routine_start_re.search(astr, ind)
        if m is None:
            break
        start = m.start()
        if function_start_re.match(astr, start, m.end()):
            while True:
                i = astr.rfind('\n', ind, start)
                if i == -1:
                    break
                start = i
                if astr[i:i + 7] != '\n     $':
                    break
        start += 1
        m = routine_end_re.search(astr, m.end())
        ind = end = m and m.end() - 1 or len(astr)
        spanlist.append((start, end))
    return spanlist
template_re = re.compile('<\\s*(\\w[\\w\\d]*)\\s*>')
named_re = re.compile('<\\s*(\\w[\\w\\d]*)\\s*=\\s*(.*?)\\s*>')
list_re = re.compile('<\\s*((.*?))\\s*>')

def find_repl_patterns(astr):
    if False:
        for i in range(10):
            print('nop')
    reps = named_re.findall(astr)
    names = {}
    for rep in reps:
        name = rep[0].strip() or unique_key(names)
        repl = rep[1].replace('\\,', '@comma@')
        thelist = conv(repl)
        names[name] = thelist
    return names

def find_and_remove_repl_patterns(astr):
    if False:
        while True:
            i = 10
    names = find_repl_patterns(astr)
    astr = re.subn(named_re, '', astr)[0]
    return (astr, names)
item_re = re.compile('\\A\\\\(?P<index>\\d+)\\Z')

def conv(astr):
    if False:
        print('Hello World!')
    b = astr.split(',')
    l = [x.strip() for x in b]
    for i in range(len(l)):
        m = item_re.match(l[i])
        if m:
            j = int(m.group('index'))
            l[i] = l[j]
    return ','.join(l)

def unique_key(adict):
    if False:
        while True:
            i = 10
    ' Obtain a unique key given a dictionary.'
    allkeys = list(adict.keys())
    done = False
    n = 1
    while not done:
        newkey = '__l%s' % n
        if newkey in allkeys:
            n += 1
        else:
            done = True
    return newkey
template_name_re = re.compile('\\A\\s*(\\w[\\w\\d]*)\\s*\\Z')

def expand_sub(substr, names):
    if False:
        while True:
            i = 10
    substr = substr.replace('\\>', '@rightarrow@')
    substr = substr.replace('\\<', '@leftarrow@')
    lnames = find_repl_patterns(substr)
    substr = named_re.sub('<\\1>', substr)

    def listrepl(mobj):
        if False:
            i = 10
            return i + 15
        thelist = conv(mobj.group(1).replace('\\,', '@comma@'))
        if template_name_re.match(thelist):
            return '<%s>' % thelist
        name = None
        for key in lnames.keys():
            if lnames[key] == thelist:
                name = key
        if name is None:
            name = unique_key(lnames)
            lnames[name] = thelist
        return '<%s>' % name
    substr = list_re.sub(listrepl, substr)
    numsubs = None
    base_rule = None
    rules = {}
    for r in template_re.findall(substr):
        if r not in rules:
            thelist = lnames.get(r, names.get(r, None))
            if thelist is None:
                raise ValueError('No replicates found for <%s>' % r)
            if r not in names and (not thelist.startswith('_')):
                names[r] = thelist
            rule = [i.replace('@comma@', ',') for i in thelist.split(',')]
            num = len(rule)
            if numsubs is None:
                numsubs = num
                rules[r] = rule
                base_rule = r
            elif num == numsubs:
                rules[r] = rule
            else:
                print('Mismatch in number of replacements (base <%s=%s>) for <%s=%s>. Ignoring.' % (base_rule, ','.join(rules[base_rule]), r, thelist))
    if not rules:
        return substr

    def namerepl(mobj):
        if False:
            return 10
        name = mobj.group(1)
        return rules.get(name, (k + 1) * [name])[k]
    newstr = ''
    for k in range(numsubs):
        newstr += template_re.sub(namerepl, substr) + '\n\n'
    newstr = newstr.replace('@rightarrow@', '>')
    newstr = newstr.replace('@leftarrow@', '<')
    return newstr

def process_str(allstr):
    if False:
        return 10
    newstr = allstr
    writestr = ''
    struct = parse_structure(newstr)
    oldend = 0
    names = {}
    names.update(_special_names)
    for sub in struct:
        (cleanedstr, defs) = find_and_remove_repl_patterns(newstr[oldend:sub[0]])
        writestr += cleanedstr
        names.update(defs)
        writestr += expand_sub(newstr[sub[0]:sub[1]], names)
        oldend = sub[1]
    writestr += newstr[oldend:]
    return writestr
include_src_re = re.compile('(\\n|\\A)\\s*include\\s*[\'\\"](?P<name>[\\w\\d./\\\\]+\\.src)[\'\\"]', re.I)

def resolve_includes(source):
    if False:
        print('Hello World!')
    d = os.path.dirname(source)
    with open(source) as fid:
        lines = []
        for line in fid:
            m = include_src_re.match(line)
            if m:
                fn = m.group('name')
                if not os.path.isabs(fn):
                    fn = os.path.join(d, fn)
                if os.path.isfile(fn):
                    lines.extend(resolve_includes(fn))
                else:
                    lines.append(line)
            else:
                lines.append(line)
    return lines

def process_file(source):
    if False:
        while True:
            i = 10
    lines = resolve_includes(source)
    return process_str(''.join(lines))
_special_names = find_repl_patterns('\n<_c=s,d,c,z>\n<_t=real,double precision,complex,double complex>\n<prefix=s,d,c,z>\n<ftype=real,double precision,complex,double complex>\n<ctype=float,double,complex_float,complex_double>\n<ftypereal=real,double precision,\\0,\\1>\n<ctypereal=float,double,\\0,\\1>\n')

def main():
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=str, help='Path to the input file')
    parser.add_argument('-o', '--outdir', type=str, help='Path to the output directory')
    args = parser.parse_args()
    if not args.infile.endswith(('.pyf', '.pyf.src', '.f.src')):
        raise ValueError(f'Input file has unknown extension: {args.infile}')
    outdir_abs = os.path.join(os.getcwd(), args.outdir)
    if args.infile.endswith(('.pyf.src', '.f.src')):
        code = process_file(args.infile)
        fname_pyf = os.path.join(args.outdir, os.path.splitext(os.path.split(args.infile)[1])[0])
        with open(fname_pyf, 'w') as f:
            f.write(code)
    else:
        fname_pyf = args.infile
    if args.infile.endswith(('.pyf.src', '.pyf')):
        p = subprocess.Popen([sys.executable, '-m', 'numpy.f2py', fname_pyf, '--build-dir', outdir_abs], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.getcwd())
        (out, err) = p.communicate()
        if not p.returncode == 0:
            raise RuntimeError(f'Writing {args.outfile} with f2py failed!\n{out}\n{{err}}')
if __name__ == '__main__':
    main()