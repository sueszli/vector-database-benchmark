"""Transform gprof(1) output into useful HTML."""
import html
import os
import re
import sys
import webbrowser
header = '<html>\n<head>\n  <title>gprof output (%s)</title>\n</head>\n<body>\n<pre>\n'
trailer = '</pre>\n</body>\n</html>\n'

def add_escapes(filename):
    if False:
        return 10
    with open(filename) as fp:
        for line in fp:
            yield html.escape(line)

def gprof2html(input, output, filename):
    if False:
        return 10
    output.write(header % filename)
    for line in input:
        output.write(line)
        if line.startswith(' time'):
            break
    labels = {}
    for line in input:
        m = re.match('(.*  )(\\w+)\\n', line)
        if not m:
            output.write(line)
            break
        (stuff, fname) = m.group(1, 2)
        labels[fname] = fname
        output.write('%s<a name="flat:%s" href="#call:%s">%s</a>\n' % (stuff, fname, fname, fname))
    for line in input:
        output.write(line)
        if line.startswith('index % time'):
            break
    for line in input:
        m = re.match('(.*  )(\\w+)(( &lt;cycle.*&gt;)? \\[\\d+\\])\\n', line)
        if not m:
            output.write(line)
            if line.startswith('Index by function name'):
                break
            continue
        (prefix, fname, suffix) = m.group(1, 2, 3)
        if fname not in labels:
            output.write(line)
            continue
        if line.startswith('['):
            output.write('%s<a name="call:%s" href="#flat:%s">%s</a>%s\n' % (prefix, fname, fname, fname, suffix))
        else:
            output.write('%s<a href="#call:%s">%s</a>%s\n' % (prefix, fname, fname, suffix))
    for line in input:
        for part in re.findall('(\\w+(?:\\.c)?|\\W+)', line):
            if part in labels:
                part = '<a href="#call:%s">%s</a>' % (part, part)
            output.write(part)
    output.write(trailer)

def main():
    if False:
        i = 10
        return i + 15
    filename = 'gprof.out'
    if sys.argv[1:]:
        filename = sys.argv[1]
    outputfilename = filename + '.html'
    input = add_escapes(filename)
    with open(outputfilename, 'w') as output:
        gprof2html(input, output, filename)
    webbrowser.open('file:' + os.path.abspath(outputfilename))
if __name__ == '__main__':
    main()