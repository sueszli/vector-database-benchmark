import os.path as op
_vispy_fonts = ('OpenSans',)

def _get_vispy_font_filename(face, bold, italic):
    if False:
        for i in range(10):
            print('nop')
    'Fetch a remote vispy font'
    name = face + '-'
    name += 'Regular' if not bold and (not italic) else ''
    name += 'Bold' if bold else ''
    name += 'Italic' if italic else ''
    name += '.ttf'
    return op.join(op.dirname(__file__), 'data', name)