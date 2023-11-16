"""Test for bitmap output via ReportLab (requires an extra dependency).

The primary purpose of this is to flag if renderPM is missing, but also
to check Bio.Graphics can make a bitmap (e.g. PNG).

The example itself is essentially a repeat from test_GraphicsGeneral.py.
"""
import os
import random
import unittest
from Bio import MissingExternalDependencyError
from Bio import MissingPythonDependencyError
try:
    import reportlab as r
    del r
except Exception:
    raise MissingPythonDependencyError('Install ReportLab if you want to use Bio.Graphics.') from None
try:
    from reportlab.graphics import renderPM
    del renderPM
except Exception:
    raise MissingPythonDependencyError("Install ReportLab's renderPM module if you want to create bitmaps with Bio.Graphics.") from None
try:
    try:
        from PIL import Image as i
    except ImportError:
        import Image as i
    del i
except Exception:
    raise MissingPythonDependencyError('Install Pillow or its predecessor PIL (Python Imaging Library) if you want to create bitmaps with Bio.Graphics.') from None
from reportlab.graphics.renderPM import RenderPMError
from Bio.Graphics.Comparative import ComparativeScatterPlot

def real_test():
    if False:
        print('Hello World!')
    min_two_d_lists = 1
    max_two_d_lists = 7
    min_num_points = 1
    max_num_points = 500
    min_point_num = 0
    max_point_num = 200
    plot_info = []
    num_two_d_lists = random.randrange(min_two_d_lists, max_two_d_lists)
    for two_d_list in range(num_two_d_lists):
        cur_list = []
        num_points = random.randrange(min_num_points, max_num_points)
        for point in range(num_points):
            x_point = random.randrange(min_point_num, max_point_num)
            y_point = random.randrange(min_point_num, max_point_num)
            cur_list.append((x_point, y_point))
        plot_info.append(cur_list)
    compare_plot = ComparativeScatterPlot('png')
    compare_plot.display_info = plot_info
    output_file = os.path.join(os.getcwd(), 'Graphics', 'scatter_test.png')
    try:
        compare_plot.draw_to_file(output_file, 'Testing Scatter Plots')
    except IndexError:
        pass
    except OSError as err:
        if 'encoder zip not available' in str(err):
            raise MissingExternalDependencyError('Check zip encoder installed for PIL and ReportLab renderPM') from None
        else:
            raise
    except RenderPMError as err:
        if str(err).startswith("Can't setFont(") or str(err).startswith('Error in setFont('):
            raise MissingExternalDependencyError('Check the fonts needed by ReportLab if you want bitmaps from Bio.Graphics\n' + str(err)) from None
        elif str(err).startswith('cannot import desired renderPM backend rlPyCairo'):
            raise MissingExternalDependencyError('Reportlab module rlPyCairo unavailable\n' + str(err)) from None
        else:
            raise
    return True
real_test()

class ComparativeTest(unittest.TestCase):
    """Do tests for modules involved with comparing data."""

    def test_simple_scatter_plot(self):
        if False:
            return 10
        'Test creation of a simple PNG scatter plot.'
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)