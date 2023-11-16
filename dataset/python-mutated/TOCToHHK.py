import os.path
import sys
'\nTOCToHHK.py\n\nConverts an AutoDuck .IDX file into a HTML Help index file.\n'

def main():
    if False:
        return 10
    file = sys.argv[1]
    output = sys.argv[2]
    input = open(file, 'r')
    out = open(output, 'w')
    line = input.readline()
    out.write('\n<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML//EN">\n<HTML>\n<HEAD>\n<meta name="GENERATOR" content="Python AutoDuck TOCToHHK.py">\n<!-- Sitemap 1.0 -->\n</HEAD><BODY>\n<UL>\n')
    while line != '':
        line = line[:-1]
        fields = line.split('\t')
        if '.' in fields[1]:
            keyword = fields[1].split('.')[-1]
        else:
            keyword = fields[1]
        context = fields[0]
        if ' ' in context:
            context = context.replace(' ', '_')
        out.write(f'    <LI><OBJECT type="text/sitemap">\n        <param name="Keyword" value="{keyword}">\n        <param name="Name" value="{fields[1]}">\n        <param name="Local" value="{context}.html">\n        </OBJECT>\n')
        line = input.readline()
    out.write('\n</UL>\n</BODY></HTML>\n')
if __name__ == '__main__':
    main()