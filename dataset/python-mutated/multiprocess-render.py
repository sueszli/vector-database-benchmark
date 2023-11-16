"""
Demonstrate the use of multiprocessing with PyMuPDF.

Depending on the  number of CPUs, the document is divided in page ranges.
Each range is then worked on by one process.
The type of work would typically be text extraction or page rendering. Each
process must know where to put its results, because this processing pattern
does not include inter-process communication or data sharing.

Compared to sequential processing, speed improvements in range of 100% (ie.
twice as fast) or better can be expected.
"""
from __future__ import print_function, division
import sys
import os
import time
from multiprocessing import Pool, cpu_count
import fitz
mytime = time.clock if str is bytes else time.perf_counter

def render_page(vector):
    if False:
        return 10
    'Render a page range of a document.\n\n    Notes:\n        The PyMuPDF document cannot be part of the argument, because that\n        cannot be pickled. So we are being passed in just its filename.\n        This is no performance issue, because we are a separate process and\n        need to open the document anyway.\n        Any page-specific function can be processed here - rendering is just\n        an example - text extraction might be another.\n        The work must however be self-contained: no inter-process communication\n        or synchronization is possible with this design.\n        Care must also be taken with which parameters are contained in the\n        argument, because it will be passed in via pickling by the Pool class.\n        So any large objects will increase the overall duration.\n    Args:\n        vector: a list containing required parameters.\n    '
    idx = vector[0]
    cpu = vector[1]
    filename = vector[2]
    mat = vector[3]
    doc = fitz.open(filename)
    num_pages = doc.page_count
    seg_size = int(num_pages / cpu + 1)
    seg_from = idx * seg_size
    seg_to = min(seg_from + seg_size, num_pages)
    for i in range(seg_from, seg_to):
        page = doc[i]
        pix = page.get_pixmap(alpha=False, matrix=mat)
    print('Processed page numbers %i through %i' % (seg_from, seg_to - 1))
if __name__ == '__main__':
    t0 = mytime()
    filename = sys.argv[1]
    mat = fitz.Matrix(0.2, 0.2)
    cpu = cpu_count()
    vectors = [(i, cpu, filename, mat) for i in range(cpu)]
    print("Starting %i processes for '%s'." % (cpu, filename))
    pool = Pool()
    pool.map(render_page, vectors, 1)
    t1 = mytime()
    print('Total time %g seconds' % round(t1 - t0, 2))