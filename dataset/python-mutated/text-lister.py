import sys
import fitz

def flags_decomposer(flags):
    if False:
        for i in range(10):
            print('nop')
    'Make font flags human readable.'
    l = []
    if flags & 2 ** 0:
        l.append('superscript')
    if flags & 2 ** 1:
        l.append('italic')
    if flags & 2 ** 2:
        l.append('serifed')
    else:
        l.append('sans')
    if flags & 2 ** 3:
        l.append('monospaced')
    else:
        l.append('proportional')
    if flags & 2 ** 4:
        l.append('bold')
    return ', '.join(l)
doc = fitz.open(sys.argv[1])
page = doc[0]
blocks = page.get_text('dict', flags=11)['blocks']
for b in blocks:
    for l in b['lines']:
        for s in l['spans']:
            print('')
            font_properties = "Font: '%s' (%s), size %g, color #%06x" % (s['font'], flags_decomposer(s['flags']), s['size'], s['color'])
            print("Text: '%s'" % s['text'])
            print(font_properties)