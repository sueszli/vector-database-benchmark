import sys

def main(in_fp, out_fp):
    if False:
        print('Hello World!')
    out_fp.write("# -*- coding: utf-8 -*-\nfrom __future__ import unicode_literals\n\n# Warning: Auto-generated file, don't edit.\npinyin_dict = {\n")
    for line in in_fp.readlines():
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        else:
            raw_line = line.split('#')[0].strip()
            new_line = raw_line.replace('U+', '0x')
            new_line = new_line.replace(': ', ": '")
            new_line = "    {new_line}',\n".format(new_line=new_line)
            out_fp.write(new_line)
    out_fp.write('}\n')
if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('python gen_pinyin_dict.py INPUT OUTPUT')
        sys.exit(1)
    in_f = sys.argv[1]
    out_f = sys.argv[2]
    with open(in_f) as in_fp, open(out_f, 'w') as out_fp:
        main(in_fp, out_fp)