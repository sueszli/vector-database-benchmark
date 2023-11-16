"""Pretty-print Python code to colorized, hyperlinked html.

In python, do:
    py2html.convert_files(['file1.py', 'file2.py', ...]) 
From the shell, do:
    python py2html.py *.py"""
import re, string, time, os
try:
    cmp
except NameError:

    def cmp(x, y):
        if False:
            return 10
        return (x > y) - (x < y)
id = '[a-zA-Z_][a-zA-Z_0-9]*'
(g1, g2, g3, g4) = '\\1 \\2 \\3 \\4'.split()

def b(text):
    if False:
        while True:
            i = 10
    return '<b>%s</b>' % text

def i(text):
    if False:
        return 10
    return '<i>%s</i>' % text

def color(rgb, text):
    if False:
        print('Hello World!')
    return '<font color="%s">%s</font>' % (rgb, text)

def link(url, anchor):
    if False:
        print('Hello World!')
    return '<a href="%s">%s</a>' % (url, anchor)

def hilite(text, bg='ffff00'):
    if False:
        i = 10
        return i + 15
    return '<b style="background-color:%s"><a name="%s">%s</b>' % (bg, text, text)

def modulelink(module, baseurl=''):
    if False:
        i = 10
        return i + 15
    'Hyperlink to a module, either locally or on python.org'
    if module + '.py' not in local_files:
        baseurl = 'http://www.python.org/doc/current/lib/module-'
    return link(baseurl + module + '.html', module)

def importer(m):
    if False:
        return 10
    "Turn text such as 'utils, math, re' into a string of HTML links."
    modules = [modulelink(mod.strip()) for mod in m.group(2).split(',')]
    return m.group(1) + ', '.join(modules) + m.group(3)

def find1(regex, str):
    if False:
        while True:
            i = 10
    return (re.findall(regex, str) or ['&nbsp;'])[0]

def convert_files(filenames, local_filenames=None, tblfile='readme.htm'):
    if False:
        return 10
    'Convert files of python code to colorized HTML.'
    global local_files
    local_files = local_filenames or filenames
    summary_table = {}
    for f in filenames:
        fulltext = '\n'.join(map(string.rstrip, open(f).readlines()))
        text = fulltext
        for (pattern, repl) in replacements:
            text = re.sub(pattern, repl, text)
        text = '<<header("AIMA Python file: %s")>><pre>%s</pre><<footer>>' % (f, text)
        open(f[:-3] + '.htm', 'w').write(text)
        if tblfile:
            ch = find1('Chapters?\\s+([^ \\)"]*)', fulltext)
            module = f.replace('.py', '')
            lines = fulltext.count('\n')
            desc = find1('"""(.*)\\n', fulltext).replace('"""', '')
            summary_table.setdefault(ch, []).append((module, lines, desc))
    if tblfile:
        totallines = 0
        tbl = ['<tr><th>Chapter<th>Module<th>Files<th>Lines<th>Description']
        fmt = '<tr><td align=right>%s<th>%s<td>%s<td align=right>%s<td>%s'
        items = summary_table.items()
        items.sort(num_cmp)
        for (ch, entries) in items:
            for (module, lines, desc) in entries:
                totallines += lines
                files = link(module + '.py', '.py')
                if os.path.exists(module + '.txt'):
                    files += ' ' + link(module + '.txt', '.txt')
                tbl += [fmt % (ch, link(module + '.html', module), files, lines, desc)]
        tbl += [fmt % ('', '', '', totallines, ''), '</table>']
        old = open(tblfile).read()
        new = re.sub('(?s)(<table border=1>)(.*)(</table>)', '\\1' + '\n'.join(tbl) + '\\3', old, 1)
        open(tblfile, 'w').write(new)

def num_cmp(x, y):
    if False:
        print('Hello World!')

    def num(x):
        if False:
            print('Hello World!')
        nums = re.findall('[0-9]+', x or '')
        if nums:
            return int(nums[0])
        return x
    return cmp(num(x[0]), num(y[0]))

def comment(text):
    if False:
        for i in range(10):
            print('nop')
    return i(color('green', text))
replacements = [('&', '&amp;'), ('<', '&lt;'), ('>', '&gt;'), ('(?ms)^#+[#_]{10,} *\\n', '<hr>'), ('(\'[^\']*?\'|"[^"]*?")', comment(g1)), ('(?s)(""".*?"""|' + "'''.*?''')", comment(g1)), ('(#.*)', color('cc33cc', g1)), ('(?m)(^[a-zA-Z][a-zA-Z_0-9, ]+)(\\s+=\\s+)', hilite(g1) + g2), ('(?m)(^\\s*)(def\\s+)(%s)' % id, g1 + b(g2) + hilite(g3)), ('(?m)(^\\s*)(class\\s+)(%s)' % id, g1 + b(g2) + hilite(g3)), ('(from\\s+)([a-z]+)(\\s+import)', importer), ('(import\\s+)([a-z, ]+)(\\s|\\n|$|,)', importer)]
if __name__ == '__main__':
    import sys, glob
    files = []
    for arg in sys.argv[1:]:
        files.extend(glob.glob(arg))
    convert_files(files)