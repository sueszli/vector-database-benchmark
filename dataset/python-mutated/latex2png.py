import os
import re
import sys
import io
import glob
import tempfile
import shlex
import subprocess
import traceback
from PIL import Image

class Latex:
    BASE = '\n\\documentclass[varwidth]{standalone}\n\\usepackage{fontspec,unicode-math}\n\\usepackage[active,tightpage,displaymath,textmath]{preview}\n\\setmathfont{%s}\n\\begin{document}\n\\thispagestyle{empty}\n%s\n\\end{document}\n'

    def __init__(self, math, dpi=250, font='Latin Modern Math'):
        if False:
            return 10
        'takes list of math code. `returns each element as PNG with DPI=`dpi`'
        self.math = math
        self.dpi = dpi
        self.font = font
        self.prefix_line = self.BASE.split('\n').index('%s')

    def write(self, return_bytes=False):
        if False:
            return 10
        try:
            workdir = tempfile.gettempdir()
            (fd, texfile) = tempfile.mkstemp('.tex', 'eq', workdir, True)
            with os.fdopen(fd, 'w+') as f:
                document = self.BASE % (self.font, '\n'.join(self.math))
                f.write(document)
            (png, error_index) = self.convert_file(texfile, workdir, return_bytes=return_bytes)
            return (png, error_index)
        finally:
            if os.path.exists(texfile):
                try:
                    os.remove(texfile)
                except PermissionError:
                    pass

    def convert_file(self, infile, workdir, return_bytes=False):
        if False:
            print('Hello World!')
        infile = infile.replace('\\', '/')
        try:
            cmd = 'xelatex -interaction nonstopmode -file-line-error -output-directory %s %s' % (workdir.replace('\\', '/'), infile)
            p = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            (sout, serr) = p.communicate()
            (error_index, _) = extract(text=sout, expression='%s:(\\d+)' % os.path.basename(infile))
            if error_index != []:
                error_index = [int(_) - self.prefix_line - 1 for _ in error_index]
            pdffile = infile.replace('.tex', '.pdf')
            (result, _) = extract(text=sout, expression='Output written on %s \\((\\d+)? page' % pdffile)
            if int(result[0]) != len(self.math):
                raise Exception("xelatex rendering error, generated %d formula's page, but the total number of formulas is %d." % (int(result[0]), len(self.math)))
            pngfile = os.path.join(workdir, infile.replace('.tex', '.png'))
            cmd = 'convert -density %i -colorspace gray %s -quality 90 %s' % (self.dpi, pdffile, pngfile)
            if sys.platform == 'win32':
                cmd = 'magick ' + cmd
            p = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (sout, serr) = p.communicate()
            if p.returncode != 0:
                raise Exception('PDFpng error', serr, cmd, os.path.exists(pdffile), os.path.exists(infile))
            if return_bytes:
                if len(self.math) > 1:
                    png = [open(pngfile.replace('.png', '') + '-%i.png' % i, 'rb').read() for i in range(len(self.math))]
                else:
                    png = [open(pngfile.replace('.png', '') + '.png', 'rb').read()]
            elif len(self.math) > 1:
                png = [pngfile.replace('.png', '') + '-%i.png' % i for i in range(len(self.math))]
            else:
                png = [pngfile.replace('.png', '') + '.png']
            return (png, error_index)
        except Exception as e:
            print(e)
        finally:
            basefile = infile.replace('.tex', '')
            tempext = ['.aux', '.pdf', '.log']
            if return_bytes:
                ims = glob.glob(basefile + '*.png')
                for im in ims:
                    os.remove(im)
            for te in tempext:
                tempfile = basefile + te
                if os.path.exists(tempfile):
                    os.remove(tempfile)
__cache = {}

def tex2png(eq, **kwargs):
    if False:
        return 10
    if not eq in __cache:
        __cache[eq] = Latex(eq, **kwargs).write(return_bytes=True)
    return __cache[eq]

def tex2pil(tex, return_error_index=False, **kwargs):
    if False:
        i = 10
        return i + 15
    (pngs, error_index) = Latex(tex, **kwargs).write(return_bytes=True)
    images = [Image.open(io.BytesIO(d)) for d in pngs]
    return (images, error_index) if return_error_index else images

def extract(text, expression=None):
    if False:
        for i in range(10):
            print('nop')
    'extract text from text by regular expression\n\n    Args:\n        text (str): input text\n        expression (str, optional): regular expression. Defaults to None.\n\n    Returns:\n        str: extracted text\n    '
    try:
        pattern = re.compile(expression)
        results = re.findall(pattern, text)
        return (results, True if len(results) != 0 else False)
    except Exception:
        traceback.print_exc()
if __name__ == '__main__':
    if len(sys.argv) > 1:
        src = sys.argv[1]
    else:
        src = '\\begin{equation}\\mathcal{ L}\\nonumber\\end{equation}'
    print('Equation is: %s' % src)
    print(Latex(src).write())