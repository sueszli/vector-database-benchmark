"""Test the creation of graphics of distribution information.

Provides tests for the Graphics.Distribution classes which provide the
ability to create graphics to show the distribution of information about
BACs/chromosomes/etc.
"""
import os
import random
import unittest
from Bio import MissingExternalDependencyError
try:
    import reportlab as r
    del r
except ImportError:
    raise MissingExternalDependencyError('Install reportlab if you want to use Bio.Graphics.') from None
from Bio.Graphics.Distribution import BarChartDistribution
from Bio.Graphics.Distribution import DistributionPage

def random_distribution(min=-5.0, max=5.0, total_items=50):
    if False:
        i = 10
        return i + 15
    'Create a series of random distribution information.'
    num_items = random.randrange(5, total_items)
    all_info = []
    for item in range(num_items):
        new_item = random.uniform(min, max)
        all_info.append(new_item)
    return all_info

class BarChartTest(unittest.TestCase):
    """Test display of BarChart distributions on a page."""

    def setUp(self):
        if False:
            return 10
        self.simple_page = os.path.join(os.getcwd(), 'Graphics', 'simple_bar.pdf')
        self.multi_page = os.path.join(os.getcwd(), 'Graphics', 'multi_bar.pdf')
        self.num_multi = 5

    def test_simple_page(self):
        if False:
            return 10
        'Test displaying a page with single distribution.'
        dist_info = []
        new_info = random_distribution()
        dist_info.append(new_info)
        distribution = BarChartDistribution(dist_info)
        dist_page = DistributionPage()
        dist_page.distributions.append(distribution)
        dist_page.draw(self.simple_page, 'Test Bar Chart')

    def test_multi_page(self):
        if False:
            return 10
        'Create a page with multiple distributions on it.'
        dist_page = DistributionPage()
        dist_page.number_of_columns = 3
        for multi in range(self.num_multi):
            dist_info = []
            new_info = random_distribution()
            dist_info.append(new_info)
            distribution = BarChartDistribution(dist_info)
            distribution.chart_title = f'Distribution {multi + 1}'
            dist_page.distributions.append(distribution)
        dist_page.draw(self.multi_page, 'Test Multi Bar Chart')
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)