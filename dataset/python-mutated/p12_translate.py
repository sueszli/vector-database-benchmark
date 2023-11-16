"""
Topic: str的translate方法清理文本
Desc : 
"""
import unicodedata
import sys

def clean_spaces(s):
    if False:
        while True:
            i = 10
    '普通替换使用replace最快'
    s = s.replace('\r', '')
    s = s.replace('\t', ' ')
    s = s.replace('\x0c', ' ')
    return s

def translate_str():
    if False:
        print('Hello World!')
    s = 'pýtĥöñ\x0cis\tawesome\r\n'
    print(s)
    remap = {ord('\t'): ' ', ord('\x0c'): ' ', ord('\r'): None}
    a = s.translate(remap)
    print(a)
    cmb_chrs = dict.fromkeys((c for c in range(sys.maxunicode) if unicodedata.combining(chr(c))))
    b = unicodedata.normalize('NFD', a)
    print(b)
    print(b.translate(cmb_chrs))
    digitmap = {c: ord('0') + unicodedata.digit(chr(c)) for c in range(sys.maxunicode) if unicodedata.category(chr(c)) == 'Nd'}
    print(len(digitmap))
    x = '١٢٣'
    print(x.translate(digitmap))
    b = unicodedata.normalize('NFD', a)
    print(type(b))
    print(b.encode('ascii', 'ignore').decode('ascii'))
if __name__ == '__main__':
    translate_str()