import os
import shutil
import tempfile
from tessellation import TessellationConfiguration
from tessellation import HyperbolicTessellation

def test_generate_many_tessellations():
    if False:
        return 10
    tmpdir = tempfile.mkdtemp()
    try:
        for p in range(3, 8):
            for q in range(3, 8):
                if (p - 2) * (q - 2) > 4:
                    config = TessellationConfiguration(p, q)
                    tessellation = HyperbolicTessellation(config)
                    tessellation.render(filename=os.path.join(tmpdir, 'tessellation_{}_{}.svg'.format(p, q)), canvas_width=500)
    finally:
        shutil.rmtree(tmpdir)